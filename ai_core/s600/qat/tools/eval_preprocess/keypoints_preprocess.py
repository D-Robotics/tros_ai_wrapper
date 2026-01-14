# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import torch.utils.data as data
import torchvision


class PILToNumpy(object):
    def __call__(self, image):
        return np.array(image)


class ResizeMinMax(object):
    def __init__(self, img_scale):
        assert len(img_scale) == 2
        self.min_side = img_scale[0]
        self.max_side = img_scale[1]

    def __call__(self, data):
        image, ldmk = data["img"], data["gt_ldmk"]
        h, w, c = image.shape
        im_size_min, im_size_max = (h, w) if w > h else (w, h)
        scale = float(self.min_side) / float(im_size_min)
        if np.floor(scale * im_size_max) > self.max_side:
            scale = float(self.max_side) / float(im_size_max)

        new_w, new_h = (int(np.floor(w * scale)), int(np.floor(h * scale)))
        resized_img = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        w_scale = float(new_w / w)
        h_scale = float(new_h / h)
        ldmk[..., 0] = ldmk[..., 0] * w_scale
        ldmk[..., 1] = ldmk[..., 1] * h_scale
        data["img"], data["gt_ldmk"] = resized_img, ldmk
        return data


class Pad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image = data["img"]
        pad = np.zeros((self.size[0], self.size[1], 3), dtype=image.dtype)
        pad[:image.shape[0], :image.shape[1], :] = image
        data["img"] = pad
        return data


class CarfusionData(data.Dataset):
    def __init__(self,
                 data_path: str,
                 anno_json_file: str,
                 transforms: list = None):
        self.data_path = data_path
        self.anno_json_file = anno_json_file
        self.transforms = transforms
        with open(anno_json_file, "r") as f:
            self.anno_dict = json.load(f)
        self.img_list = list(self.anno_dict.keys())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        key = self.img_list[idx]
        image_path = os.path.join(self.data_path, key)
        img_id = image_path.split("/")[-1]
        img_folder = image_path.split("/")[-3]
        image = cv2.imread(image_path)
        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        image = np.array(image)
        h, w, _ = image.shape

        valid_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                               1])  # No.0, No.9 无效
        valid_mask = valid_mask.astype("bool")
        keypoints = self.anno_dict[key]
        keypoints = np.array(keypoints).reshape(14, 3)
        keypoints = keypoints[valid_mask]

        gt_ldmk = keypoints[:, :2]
        gt_ldmk_attr = keypoints[:, 2]

        gt_ldmk_attr[(gt_ldmk[:, 0] < 0) | (gt_ldmk[:, 0] > w)] = 0
        gt_ldmk_attr[(gt_ldmk[:, 1] < 0) | (gt_ldmk[:, 1] > h)] = 0

        data = {
            "img": image,
            "img_name": f"{img_folder}_{img_id}",
            "gt_ldmk": gt_ldmk,
            "gt_ldmk_attr": gt_ldmk_attr,
            "layout": "hwc",
            "img_shape": image.shape,
            "color_space": "rgb",
        }
        if self.transforms:
            data = self.transforms(data)
        return data


transform = torchvision.transforms.Compose([
    # PILToNumpy(),
    ResizeMinMax(img_scale=(128, 128)),
    Pad(size=(128, 128)),
])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-root", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--label-path", type=str, help="path to label", required=True)
parser.add_argument(
    "--save-path",
    type=str,
    help="output path",
    default="./processed_carfusion",
)
args = parser.parse_args()

save_path = args.save_path
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

data_path = args.data_root
label_path = args.label_path

keypoint_data = CarfusionData(
    data_path=data_path,
    anno_json_file=label_path,
    transforms=transform,
)
anno_all = {}
for i, samples in enumerate(keypoint_data):
    if (i + 1) % 500 == 0:
        print("{}th finished!".format(i + 1))
    image = samples["img"]
    image_name = samples["img_name"]
    image_name = image_name[:-4] + ".png"
    anno_all[image_name] = [
        samples["gt_ldmk"].tolist(),
        samples["gt_ldmk_attr"].tolist(),
    ]
    # image = image[:, :, ::-1]
    cv2.imwrite(os.path.join(save_path, image_name), image)
with open(os.path.join(save_path, "processed_anno.json"), "w") as f:
    json.dump(anno_all, f)
print("All images finished!")
