# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import glob
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

train_list = glob.glob("train/car_craig1/bb/*.txt")


def check_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop car patches from carfusion dataset," +
        "and convert data to coco format")
    parser.add_argument(
        "-s",
        "--src-data-path",
        type=str,
        required=True,
        help="src data path",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="output path",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="if num_workers > 1, would use multiprocessing accelerating.",
    )
    return parser.parse_args()


def visualize(image: Image.Image, keypoints: list, save_path):
    """
    Visualize keypoints on an image with cv2, and saved.

    Args:
        image (PIL Image): The input image.
        keypoints (np.array): A NumPy array with shape [14, 3], where each row
            contains the (x, y, vis) values for a single keypoint.
            The 'x' and 'y' values are floats in the range of (0, 1),
            representing the relative coordinates on the image.

    Returns:
        Image.Image: The image visualized with keypoints.
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    for x, y, _ in keypoints:
        img_x = int(x * image.width)
        img_y = int(y * image.height)

        radius = 3
        color = (255, 0, 0)
        draw.ellipse(
            (img_x - radius, img_y - radius, img_x + radius, img_y + radius),
            fill=color,
        )

    vis_image.save(save_path)
    return vis_image


def convert_single_file(gt_path) -> tuple:
    """
    Convert a single file of the CarFusion dataset.

    Args:
        bb_path (str): The path to the bounding box file.

    Returns:
        tuple: A tuple containing the converted image (PIL Image) and the
        keypoint list (list).
    """
    data_root = "/".join(gt_path.split("/")[:-2])
    img_id = gt_path.split("/")[-1][:-4]
    img_path = os.path.join(data_root, "images_jpg", f"{img_id}.jpg")
    bb_path = os.path.join(data_root, "bb", f"{img_id}.txt")
    width = 1920
    height = 1080
    if not os.path.exists(gt_path):
        print(f"No gt file : {gt_path}, Skip")
        return None, None
    with open(bb_path, "r") as f:
        lines = f.readlines()
        bb_list = []
        cat_list = []
        for line in lines:
            bb_str = line.strip().split("]")[0].strip("[")
            if len(bb_str.split(",")) == 4:  # [y, x, h, w]
                split_bb = bb_str.split(",")
                bb = [float(s.strip().strip(",")) for s in split_bb]
                bb = [bb[0], bb[1], bb[3], bb[2]]
            else:  # [x. y. w. h]
                split_bb = bb_str.split()
                bb = [float(s.strip().strip(",")) for s in split_bb]

            cat = line.strip().split(",")[1]
            bb_list.append(bb)
            cat_list.append(cat)

    keypoint_list = [np.zeros([14, 3]) for i in range(len(bb_list))]
    with open(gt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.strip().split(",")
            keypoints = [float(num) for num in split_line[:2]]
            keypoint_idx, instance_id = int(split_line[2]), int(split_line[3])
            vis_val = int(split_line[4])
            vis_value = 1 if vis_val == 2 else 2
            keypoint_list[instance_id][keypoint_idx] = np.array(
                [keypoints[0], keypoints[1], vis_value])

    image = Image.open(img_path)
    cropped_images = []
    # print(list(keypoint_dict.keys()))
    for i in range(len(bb_list)):
        bbox = bb_list[i]
        xmin, ymin, w, h = bbox

        xmin = max(xmin - 0.1 * w, 0)
        ymin = max(ymin - 0.2 * h, 0)
        w, h = w * 1.2, h * 1.4
        xmax = min(xmin + w, width)
        ymax = min(ymin + h, height)
        try:
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
        except ValueError as e:
            print(e, (xmin, ymin, xmax, ymax), w, h, gt_path, bb_path,
                  img_path)
        cropped_images.append(cropped_img)

        keypoint_list[i][:, 0] = keypoint_list[i][:, 0] - xmin
        keypoint_list[i][:, 1] = keypoint_list[i][:, 1] - ymin

    return cropped_images, keypoint_list


def dump_keypoints(keypoint_dict, output_file):
    new_dict = {}
    for key, value in keypoint_dict.items():
        if isinstance(value, np.ndarray):
            new_dict[key] = value.tolist()
    with open(output_file, "w") as f:
        json.dump(new_dict, f)


def convert_group_data(gt_list, save_dir):

    keypoint_dict_all = {}
    instance_cnt = 0
    skip_cnt = 0
    for idx, gt_path in enumerate(gt_list):
        img_id = gt_path.split("/")[-1][:-4]
        cropped_images, keypoint_dict = convert_single_file(gt_path)
        if cropped_images is None:
            skip_cnt += 1
            continue
        for i in range(len(cropped_images)):
            save_img_path = os.path.join(save_dir, f"{img_id}_{i}.jpg")
            try:
                cropped_images[i].save(save_img_path)
            except ValueError as e:
                print(e, "skip this image")
                continue
            keypoint_dict_all[save_img_path] = keypoint_dict[i]
            instance_cnt += 1
        if idx % 1000 == 0:
            print(f"Convert {save_dir} data {idx}/{len(gt_list)}, " +
                  f"instance_cnt: {instance_cnt}, skip {skip_cnt}.")
    return keypoint_dict_all


def process_subfolder_all(subfolder, data_root):
    phase = subfolder.split("/")[-2]
    base_folder = subfolder.split("/")[-1]
    gt_list = glob.glob(f"{subfolder}/gt/*.txt")
    out_dir = f"{data_root}/{phase}"

    save_dir = os.path.join(out_dir, base_folder, "crop_imgs")
    check_mkdir(f"{out_dir}/{base_folder}/crop_imgs")

    out_keypoint_dict_path = (
        f"{data_root}/simple_anno/keypoints_{phase}_{base_folder}.json")

    if not os.path.exists(out_keypoint_dict_path):
        print("Crop and compute keypoint data")
        keypoint_dict = convert_group_data(gt_list, save_dir)
        dump_keypoints(keypoint_dict, out_keypoint_dict_path)
    else:
        print("Load pre compute keypoint data")
        with open(out_keypoint_dict_path, "r") as f:
            keypoint_dict = json.load(f)


def fuse_annotation(phase, data_root, out_keypoint_dict_path):
    anno_list = glob.glob(
        f"{data_root}/simple_anno/keypoints_{phase}_car*.json")
    keypoint_dic_list = []
    for i in range(len(anno_list)):
        with open(anno_list[i], "r") as f:
            keypoint_dict = json.load(f)
            keypoint_dic_list.append(keypoint_dict)

    fused_dict = {}
    for d in keypoint_dic_list:
        fused_dict.update(d)

    with open(out_keypoint_dict_path, "w") as f:
        json.dump(fused_dict, f)
    return fused_dict


if __name__ == "__main__":
    args = parse_args()
    data_root = args.out_dir

    sub_folder_list = glob.glob(f"{args.src_data_path}/test/car_*")
    print(sub_folder_list)

    check_mkdir(f"{data_root}/simple_anno")
    if args.num_workers <= 1:
        for subfolder in sub_folder_list:
            process_subfolder_all(subfolder, data_root)
    else:
        with Pool(args.num_workers) as p:
            p.map(
                partial(process_subfolder_all, data_root=data_root),
                sub_folder_list,
            )

    print("Cropped car patches done.")
    print("---------------------------------")
    print("Begin fuse annoatations of subsets")
    for phase in ["train", "test"]:
        out_keypoint_dict_path = (
            f"{data_root}/simple_anno/keypoints_{phase}.json")
        keypoint_dict_simple = fuse_annotation(phase, data_root,
                                               out_keypoint_dict_path)
    print("Done")
