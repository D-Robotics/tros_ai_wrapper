# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from PIL import Image
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='evaluate coco dataset image path',
        required=True)
    args = parser.parse_args()
    return args


def get_single_image_info(image_path):
    image = Image.open(image_path)

    # get name
    image_name = image_path.split("/")[-1]
    image_name = image_name.split(".")[0] + ".png"

    # get width and height
    image_width, image_height = image.size

    return image_name, image_width, image_height


def dump_all_image_info(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            image_info_list = ""
            try:
                image_info = get_single_image_info(file_path)
                image_info_list += image_info[0] + " " + str(
                    image_info[1]) + " " + str(image_info[2])
            except Exception as e:
                print(f"Error processing image {file_name}: {str(e)}")
            with open('coco_origin_info.txt', 'a') as file:
                file.write(image_info_list + "\n")
                file.flush()


if __name__ == '__main__':
    args = get_args()
    dump_all_image_info(args.image_path)
