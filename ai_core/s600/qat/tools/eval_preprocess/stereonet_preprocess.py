# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import torch
import copy
import re
import pickle
import argparse
import numpy as np
import torchvision

from PIL import Image
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union, Sequence, Optional, List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="The dir of sceneflow data",
        required=True)
    parser.add_argument(
        "--data-list", type=str, help="The filelist of data", required=True)
    parser.add_argument(
        "--save-path", type=str, help="output path", default="./sceneflow_val")
    args = parser.parse_args()
    return args


def pfm_imread(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def image_pad(
        img: Union[np.ndarray, torch.Tensor],
        layout,
        shape=None,
        divisor=1,
        pad_val=0,
) -> Union[np.ndarray, torch.Tensor]:
    """Pad image to a certain shape.

    Args:
        layout (str): Layout of img, `hwc` or `chw` or `hw`.
        shape (tuple): Expected padding shape, meaning of dimension is the
            same as img, if layout of img is `hwc`, shape must be (pad_h,
            pad_w) or (pad_h, pad_w, c).
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Union[float, Sequence[float]]): Values to be filled in padding
            areas, single value or a list of values with len c.
            E.g. : pad_val = 10, or pad_val = [10, 20, 30].

    Returns:
        ndarray or torch.Tensor: padded image.
    """
    assert layout in ["hwc", "chw", "hw"]
    if isinstance(pad_val, Sequence):
        assert layout in ["hwc", "chw"]
        c_index = layout.index("c")
        assert len(pad_val) == img.shape[c_index]
        pad_val = torch.tensor(pad_val)
        if layout == "hwc":
            pad_val = pad_val.unsqueeze(0).unsqueeze(0)
        elif layout == "chw":
            pad_val = pad_val.unsqueeze(-1).unsqueeze(-1)
        if isinstance(img, np.ndarray):
            pad_val = pad_val.numpy()

    # calculate pad_h and pad_h
    if shape is None:
        shape = img.shape
        if divisor == 1:
            return img
    assert len(shape) in [2, 3]
    if layout == "chw":
        if len(shape) == 2:
            h = max(img.shape[1], shape[0])
            w = max(img.shape[2], shape[1])
        else:
            h = max(img.shape[1], shape[1])
            w = max(img.shape[2], shape[2])
    else:
        h = max(img.shape[0], shape[0])
        w = max(img.shape[1], shape[1])
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    if len(shape) == 3:
        if layout == "hwc":
            shape = (pad_h, pad_w, shape[-1])
        elif layout == "chw":
            shape = (shape[0], pad_h, pad_w)
    else:
        shape = (pad_h, pad_w)

    if len(shape) < len(img.shape):
        if layout == "hwc":
            shape = tuple(shape) + (img.shape[-1], )
        elif layout == "chw":
            shape = (img.shape[0], ) + tuple(shape)
    assert len(shape) == len(img.shape)
    for i in range(len(shape)):
        assert shape[i] >= img.shape[i], ("padded shape must greater than "
                                          "the src shape of img")

    if isinstance(img, Tensor):
        pad = torch.zeros(shape, dtype=img.dtype, device=img.device)
    elif isinstance(img, np.ndarray):
        pad = np.zeros(shape, dtype=img.dtype)
    else:
        raise TypeError
    pad[...] = pad_val

    if len(img.shape) == 2:
        pad[:img.shape[0], :img.shape[1]] = img
    elif layout == "hwc":
        pad[:img.shape[0], :img.shape[1], ...] = img
    else:
        pad[:, :img.shape[1], :img.shape[2]] = img
    return pad


class Pad(object):
    def __init__(self, size=None, divisor=1, pad_val=0):
        """Pad image & mask & seg.

        .. note::
            Affected keys: 'img', 'layout', 'pad_shape', 'gt_seg'.

        Args:
            size (Optional[tuple]): Expected padding size, meaning of dimension
                is the same as img, if layout of img is `hwc`, shape must be
                (pad_h, pad_w) or (pad_h, pad_w, c).
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (Union[float, Sequence[float]]): Values to be filled in
                padding areas for img, single value or a list of values with
                len c. E.g. : pad_val = 10, or pad_val = [10, 20, 30].
        """
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val

    def _pad_img(self, data):
        if data["layout"] == "chw":
            data["before_pad_shape"] = np.array(data["img"].shape[1:])
        else:
            data["before_pad_shape"] = np.array(data["img"].shape[:2])
        padded_img = image_pad(data["img"], data["layout"], self.size,
                               self.divisor, self.pad_val)
        data["img"] = padded_img
        data["padded_img"] = padded_img
        data["pad_shape"] = padded_img.shape

    def _pad_disp(self, data):
        if data["layout"] == "chw":
            size = data["pad_shape"][1:]
        else:
            size = data["pad_shape"][:2]
        padded_disp = image_pad(data["gt_disp"], "hw", size, 1, -1)
        data["gt_disp"] = padded_disp

    def __call__(self, data):
        self._pad_img(data)
        if "gt_disp" in data:
            self._pad_disp(data)
        return data


class ToTensor(object):
    """Convert objects of various python types to torch.Tensor and convert the
    img to yuv444 format if to_yuv is True.

    Supported types are: numpy.ndarray, torch.Tensor, Sequence, int, float.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout', 'gt_bboxes',
        'gt_seg', 'gt_seg_weights', 'gt_flow', 'color_space'.

    Args:
        to_yuv: If true, convert the img to yuv444 format.
        use_yuv_v2: If true, use BgrToYuv444V2 when convert img to yuv format.

    """

    def __init__(self, to_yuv: bool = False, use_yuv_v2: bool = True):
        self.to_yuv = to_yuv
        self.use_yuv_v2 = use_yuv_v2

    @staticmethod
    def _convert_layout(img: Union[torch.Tensor, np.ndarray], layout: str):
        # convert the layout from hwc to chw
        assert layout in ["hwc", "chw"]
        if layout == "chw":
            return img, layout

        if isinstance(img, torch.Tensor):
            img = img.permute((2, 0, 1))  # HWC > CHW
        elif isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC > CHW
        else:
            raise TypeError
        return img, "chw"

    @staticmethod
    def _to_tensor(
            data: Union[torch.Tensor, np.ndarray, Sequence, int, float]):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f"type {type(data)} cannot be converted to tensor.")

    @staticmethod
    def _convert_hw_to_even(data):  # noqa: D205,D400
        """Convert hw of img and img-like labels to even because BgrToYuv444
        requires that the h and w of img are even numbers.
        """
        if data["layout"] == "hwc":
            h, w, c = data["img"].shape
        else:
            c, h, w = data["img"].shape
            if "before_crop_shape" in data:
                data["before_crop_shape"] = (
                    data["before_crop_shape"][2],
                    data["before_crop_shape"][0],
                    data["before_crop_shape"][1],
                )

        crop_h = h if (h % 2) == 0 else h - 1
        crop_w = w if (w % 2) == 0 else w - 1

        if data["layout"] == "hwc":
            data["img"] = data["img"][:crop_h, :crop_w, :]
        else:
            data["img"] = data["img"][:, :crop_h, :crop_w]

        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        return data

    def __call__(self, data):
        # step1: convert the layout from hwc to chw
        data["img"], data["layout"] = self._convert_layout(
            data["img"], data["layout"])
        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        # step2: convert to tensor
        data["img"] = self._to_tensor(data["img"])

        return data


class SceneFlowFromImage(Dataset):
    """SceneFlowFromImage which gets img data and gt from the data_path.

    Args:
        data_path: The dir of sceneflow data.
        data_list: The filelist of data.
        transforms: List of transform.
    """

    def __init__(
            self,
            data_path: str,
            data_list: str,
            transforms: Optional[List] = None,
    ):

        self.data_path = data_path
        self.data_list = data_list
        self.transforms = transforms
        (
            self.left_filenames,
            self.right_filenames,
            self.disp_filenames,
        ) = self.load_path(data_list)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        data = Image.open(filename).convert("RGB")
        return data

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, item):
        left_img = self.load_image(
            os.path.join(self.data_path, self.left_filenames[item]))
        right_img = self.load_image(
            os.path.join(self.data_path, self.right_filenames[item]))
        disparity = self.load_disp(
            os.path.join(self.data_path, self.disp_filenames[item]))
        w, h = left_img.size

        img_l = np.array(left_img)
        img_r = np.array(right_img)
        img = np.concatenate((img_l, img_r), axis=2)

        data = {}
        data["img_name"] = self.left_filenames[item]
        data["layout"] = "hwc"
        data["img_shape"] = img_l.shape
        data["gt_disp"] = disparity
        data["color_space"] = "rgb"
        data["img"] = img
        data["ori_img"] = copy.deepcopy(img)
        if self.transforms is not None:
            data = self.transforms(data)

        return data


def main():
    args = get_args()

    left_path = os.path.join(args.save_path, "left")
    right_path = os.path.join(args.save_path, "right")
    os.makedirs(left_path)
    os.makedirs(right_path)

    transforms = torchvision.transforms.Compose(
        [Pad(divisor=32),
         ToTensor(to_yuv=False, use_yuv_v2=False)])

    sceneflow_dataset = SceneFlowFromImage(
        data_path=args.data_path,
        data_list=args.data_list,
        transforms=transforms)

    infos = {}
    for i in tqdm(range(len(sceneflow_dataset))):
        data = sceneflow_dataset[i]
        _, h, w = data["img"].shape
        data["img"] = data["img"].reshape(2, 3, h, w)

        image1 = data["img"][0:1, :, :, :].permute(0, 2, 3,
                                                   1).squeeze(dim=0).numpy()
        image2 = data["img"][1:2, :, :, :].permute(0, 2, 3,
                                                   1).squeeze(dim=0).numpy()

        left_image = Image.fromarray(image1)
        right_image = Image.fromarray(image2)

        image_name = "_".join(data["img_name"].split("/")[1:])[:-4].replace(
            "_left", "") + "_544_960.png"

        left_image.save(left_path + "/" + image_name)
        right_image.save(right_path + "/" + image_name)

        infos[image_name] = {"metas": data["gt_disp"]}

    gt_anno_file = os.path.join(args.save_path, "val_gt_infos.pkl")

    with open(gt_anno_file, "wb") as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print("Preprocess start!")
    main()
    print("Preprocess end!")
