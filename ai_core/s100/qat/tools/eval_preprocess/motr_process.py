# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

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
import glob
from PIL import Image


class Mot17FromImage(data.Dataset):
    """Mot17FromImage which gets img data and gt from the data_path.

    Args:
        data_path: The dir of mot17 data.
        sampler_lengths: The length of the sequence data.
        sample_mode: The sampling mode,
            only support 'fixed_interval' or 'random_interval'.
        sample_interval: The sampling interval,
            if sample_mode is 'random_interval',
            randomly select from [1, sample_interval].
        sampler_steps: Sequence length changes according to the epoch.
        transforms: List of transform.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
            self,
            data_path: str,
            transforms: Optional[List] = None,
            to_rgb: bool = True,
    ):
        self.data_path = data_path
        self.transforms = transforms
        self.to_rgb = to_rgb

        self.seqs = sorted(os.listdir(self.data_path))

        self.img_files = []
        for seq in self.seqs:
            seq_path = os.path.join(self.data_path, seq, "img1")
            images = sorted(glob.glob(seq_path + "/*.jpg"))
            self.img_files.extend(images)

        self.num_samples = len(self.img_files)

    def __len__(self):
        return self.num_samples

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]

        img = Image.open(img_path)
        single_data = {}

        img_name = img_path[len(self.data_path) + 1:]

        single_data["img"] = np.array(img)
        single_data["color_space"] = "rgb"
        single_data["layout"] = "hwc"
        single_data["img_name"] = img_name

        return single_data

    def __getitem__(self, idx: int):

        data_seq = self._pre_single_frame(idx)

        if self.transforms is not None:
            data_seq = self.transforms(data_seq)
        return data_seq


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--save-path", type=str, help="output path", default="./mot17/")
args = parser.parse_args()

if os.path.exists(args.save_path):
    print(f"{args.save_path} dir exits!")
    sys.exit(0)

os.mkdir(args.save_path)
if args.image_path[-1] == "/":
    args.image_path = args.image_path[:-1]

culane_data = Mot17FromImage(data_path=args.image_path, )

for i, samples in enumerate(culane_data):
    if (i + 1) % 500 == 0:
        print("{}th finished!".format(i + 1))
    image_name = samples["img_name"]
    img = samples["img"]
    resized_img = cv2.resize(img, (1422, 800), interpolation=cv2.INTER_LINEAR)

    save_name = os.path.join(args.save_path,
                             "M" + image_name[1:-4].replace('/', '_') + ".png")
    image = Image.fromarray(resized_img)
    image.save(save_name)
