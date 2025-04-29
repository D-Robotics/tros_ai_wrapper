# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import re

import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset


class ResizeMinMax(object):
    def __init__(self, img_scale):
        assert len(img_scale) == 2
        self.min_side = img_scale[0]
        self.max_side = img_scale[1]

    def __call__(self, image):
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
        return resized_img


class Pad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        pad = np.zeros((self.size[0], self.size[1], 3), dtype=image.dtype)
        pad[:image.shape[0], :image.shape[1], :] = image
        return pad


class NuscenesMonoImage(Dataset):
    def __init__(
            self,
            src_data_dir,
            file_path,
            transforms=None,
    ) -> None:
        super(NuscenesMonoImage, self).__init__()
        self.src_data_dir = src_data_dir
        self.transforms = transforms
        self.file_name = []
        f = open(file_path)
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            self.file_name.append(line[0])

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        file_name = self.file_name[index]
        file_path = os.path.join(self.src_data_dir, file_name)
        image = cv2.imread(file_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, file_name


transforms = torchvision.transforms.Compose([
    ResizeMinMax(img_scale=(512, 896)),
    Pad(size=(512, 896)),
])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src-data-dir", type=str, default="./Nuscenes", help="path to dataset")
parser.add_argument(
    "--file-path",
    type=str,
    default="../../script/config/model/data_name_list/nuscenes_names.txt",
    help="path to label",
)
parser.add_argument(
    "--save-path",
    type=str,
    help="output path",
    default="./processed_fcos3d_images")
args = parser.parse_args()

dataset = NuscenesMonoImage(
    src_data_dir=args.src_data_dir,
    file_path=args.file_path,
    transforms=transforms,
)

print(len(dataset))
for i in range(len(dataset)):
    if i % 1000 == 0 or i == len(dataset) - 1:
        print(i)
    image, file_name = dataset[i]
    # print(image.dtype);print(image.shape)
    file_name = file_name.replace("jpg", "png")
    file_path = os.path.join(args.save_path, file_name)
    dir_name = os.path.dirname(file_path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # print(file_path)
    cv2.imwrite(os.path.join(args.save_path, file_name), image)
    # break


# process fcos3d config txt
def txt2configtxt(list):
    res_str = ""
    for i in range(len(list)):
        if i == 0:
            tmp = list[i].split('/')[-1]
            res_str += tmp.replace("jpg", "png") + " "
        # res_str +=list[i].split('/')[-1] +" "
        else:
            a = re.search("(\d+(\.\d+)?)", list[i])
            if i == len(list) - 1:
                res_str += a.group()
            else:
                res_str += a.group() + " "
    return res_str


def processtxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        # print(row.strip('\n'))
        row = row.strip('\n')
        tmp_list = row.split(' ')  # 按‘，'切分每行的数据
        # print(tmp_list)
        data.append(txt2configtxt(tmp_list))
    file.close()
    file_path_txt = args.save_path + "/fcos3d_nuscenes_camconfig.txt"
    f = open(file_path_txt, 'w')
    for line in data:
        print(line)
        f.write(line + '\n')


print("Process fcos3d config txt start!")
processtxt(args.file_path)
print("Process fcos3d config txt finished!")
