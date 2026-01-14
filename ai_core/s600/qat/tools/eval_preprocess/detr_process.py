# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import shutil

import cv2
import argparse
import numpy as np
import torchvision
from torchvision.datasets import CocoDetection


class PILToNumpy(object):
    def __call__(self, image):
        return np.array(image)


class ResizeMinMax(object):
    def __init__(self, img_scale, keep_ratio=True):
        assert len(img_scale) == 2
        self.min_side = img_scale[0]
        self.max_side = img_scale[1]
        self.keep_ratio = keep_ratio

    def __call__(self, image):
        h, w, c = image.shape
        if self.keep_ratio:
            im_size_min, im_size_max = (h, w) if w > h else (w, h)
            scale = float(self.min_side) / float(im_size_min)
            if np.floor(scale * im_size_max) > self.max_side:
                scale = float(self.max_side) / float(im_size_max)

            new_w, new_h = (int(np.floor(w * scale)), int(np.floor(h * scale)))
        else:
            new_w, new_h = (int(self.max_side), int(self.min_side))
        resized_img = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized_img


transform = torchvision.transforms.Compose([
    PILToNumpy(),
    ResizeMinMax(img_scale=(800, 1333), keep_ratio=False),
])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--label-path", type=str, help="path to label", required=True)
parser.add_argument(
    "--save-path",
    type=str,
    help="output path",
    default="./processed_detr_val_2017")
args = parser.parse_args()

save_path = args.save_path
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

image_path = args.image_path
label_path = args.label_path

mscoc_data = CocoDetection(
    root=image_path,
    annFile=label_path,
    transform=transform,
)

for i, samples in enumerate(mscoc_data):
    if (i + 1) % 500 == 0:
        print("{}th finished!".format(i + 1))
    image, target = samples
    image_id = mscoc_data.ids[i]
    image_name = "%012d" % image_id + ".png"
    image = image[0:800, 0:1332, ::-1]
    cv2.imwrite(os.path.join(save_path, image_name), image)
print("All images finished!")
