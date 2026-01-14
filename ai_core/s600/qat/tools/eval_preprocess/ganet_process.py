# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import sys
from typing import Optional, List

import cv2
import numpy as np
import torch.utils.data as data
import copy
from PIL import Image


def path_join(root, name):
    if root == '':
        return name
    if name[0] == '/':
        return os.path.join(root, name[1:])
    else:
        return os.path.join(root, name)


class CuLaneFromImage(data.Dataset):
    def __init__(
            self,
            data_path: str,
            transforms: Optional[List] = None,
            to_rgb: bool = False,
            train_flag: bool = False,
    ):
        self.data_path = data_path

        data_list = path_join(self.data_path, "list/test.txt")
        if train_flag:
            data_list = path_join(self.data_path, "list/train.txt")
        self.img_infos, self.annotations = self.parser_datalist(data_list)
        self.transforms = transforms
        self.to_rgb = to_rgb

    def parser_datalist(self, data_list):
        img_infos, annotations = [], []
        with open(data_list) as f:
            lines = f.readlines()
            for line in lines:
                img_dir = line.strip()
                img_infos.append(img_dir)
                anno_dir = img_dir.replace('.jpg', '.lines.txt')
                annotations.append(anno_dir)
        return img_infos, annotations

    def load_labels(self, idx):
        anno_dir = path_join(self.data_path, self.annotations[idx])
        annos = []
        with open(anno_dir, 'r') as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords_str = line.strip().split(' ')
                num_point = len(coords_str) // 2
                if num_point < 2:
                    continue
                points = np.zeros([num_point, 2], dtype=np.float32)
                for i in range(num_point):
                    points[i][0] = float(coords_str[2 * i])
                    points[i][1] = float(coords_str[2 * i + 1])
                annos.append(points)
        return annos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        data = {}
        img_name = self.img_infos[idx]
        img_path = path_join(self.data_path, img_name)

        img = cv2.imread(img_path)

        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            color_space = "rgb"

        ori_shape = img.shape
        gt_lines = self.load_labels(idx)

        data["image_name"] = img_name
        data["img"] = img
        data["gt_lines"] = gt_lines
        data["img_shape"] = ori_shape
        data["color_space"] = color_space
        data["layout"] = "hwc"

        data["ori_gt_lines"] = copy.deepcopy(data["gt_lines"])
        data["ori_img"] = copy.deepcopy(data["img"])

        if self.transforms is not None:
            data = self.transforms(data)
        return data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--save-path", type=str, help="output path", default="./culane/")
args = parser.parse_args()

if os.path.exists(args.save_path):
    print(f"{args.save_path} dir exits!")
    sys.exit(0)

os.mkdir(args.save_path)

culane_data = CuLaneFromImage(data_path=args.image_path, to_rgb=True)

for i, samples in enumerate(culane_data):
    if (i + 1) % 500 == 0:
        print("{}th finished!".format(i + 1))
    image_name = samples["image_name"]
    img = samples["img"][270:590, 0:1640]
    resized_img = cv2.resize(img, (800, 320), interpolation=cv2.INTER_LINEAR)

    save_name = os.path.join(args.save_path,
                             image_name[1:-4].replace('/', '_') + ".png")
    image = Image.fromarray(resized_img)
    image.save(save_name)
