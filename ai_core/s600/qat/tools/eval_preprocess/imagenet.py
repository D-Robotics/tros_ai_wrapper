# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import sys
import numpy as np
import torchvision
from PIL import Image

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        # return img


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--save-path", type=str, help="output path", default="./imagenet/")
args = parser.parse_args()

dataset = torchvision.datasets.ImageNet(
    root=args.image_path, split="val", transform=transforms, loader=pil_loader)

if os.path.exists(args.save_path):
    print(f"{args.save_path} dir exist!")
    sys.exit(0)

os.mkdir(args.save_path)

for i, samples in enumerate(dataset):
    image, target = samples
    image_name = dataset.imgs[i][0]
    save_name = os.path.join(args.save_path,
                             os.path.basename(image_name)[:-4] + "png")
    data = np.asarray(image)
    # data = data[:, :, ::-1]
    image = Image.fromarray(data)
    image.save(save_name)
