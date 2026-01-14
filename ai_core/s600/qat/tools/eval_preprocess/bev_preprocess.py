# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import copy
import argparse
import os
import shutil
import pickle
import multiprocessing
from typing import Mapping, Optional, Sequence, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision
import torchvision.transforms.functional as F

try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
    from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from shapely import affinity, ops
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, box
except ImportError:
    NuScenesCanBus = None
    Quaternion = None
    quaternion_yaw = None
    NuScenesMap = None
    NuScenesMapExplorer = None
    NuScenes = None
    splits = None

PIL_INTERP_CODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
}

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)
DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}
Attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)


def get_temporal_info(meta):
    meta_list = [{key: value} for key, value in meta.items()]
    temporal_info = []
    for i in range(len(meta_list)):
        current = meta_list[i]
        previous1 = meta_list[i - 1] if i >= 1 else meta_list[0]
        previous2 = meta_list[i - 2] if i >= 2 else meta_list[0]
        group = [previous2, previous1, current]
        temporal_info.append(group)

    return temporal_info


def gen_prev_points(data, scale, coords, offset, save_path):
    temporal_info = get_temporal_info(data)
    for i in range(len(temporal_info)):
        curr_token = list(temporal_info[i][2].keys())[0]
        file_path = os.path.join(save_path, curr_token)

        curr_ego2global = np.array(
            list(temporal_info[i][2].values())[0]["metadata"]["ego2global"])
        prev1_ego2global = np.array(
            list(temporal_info[i][1].values())[0]["metadata"]["ego2global"])
        prev2_ego2global = np.array(
            list(temporal_info[i][0].values())[0]["metadata"]["ego2global"])

        prev1_points = wrap(bev_size, prev1_ego2global, curr_ego2global,
                            coords, offset, scale)
        prev2_points = wrap(bev_size, prev2_ego2global, prev1_ego2global,
                            coords, offset, scale)

        prev_points = np.concatenate((prev1_points, prev2_points), axis=0)

        prev_points.tofile(file_path)


def get_min_max_coords(bev_size: Tuple[float],
                       ) -> Tuple[float, float, float, float]:
    """Get min and max coords.

    Args:
        bev_size: Backbone module.
        neck: Neck module.
    """

    min_x = -bev_size[1] + bev_size[2] / 2
    max_x = bev_size[1] - bev_size[2] / 2
    min_y = -bev_size[0] + bev_size[2] / 2
    max_y = bev_size[0] - bev_size[2] / 2
    return min_x, max_x, min_y, max_y


def get_matrix(prev_e2g, cur_e2g, bev_x, bev_y, bev_size):
    prev_e2g = prev_e2g[np.newaxis, :]
    prev_g2e = np.linalg.inv(prev_e2g)
    cur_e2g = cur_e2g[np.newaxis, :]
    wrap_m = prev_g2e @ cur_e2g
    wrap_r = wrap_m[:, :2, :2].transpose((0, 2, 1))
    wrap_t = wrap_m[:, :2, 3]
    wrap_t = wrap_t + np.array([bev_x, bev_y])

    wrap_r /= bev_size[2]
    wrap_t /= bev_size[2]
    return wrap_r, wrap_t


def gen_coords(bev_size, grid_size):
    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(bev_size)
    W = grid_size[0]
    H = grid_size[1]
    x = (torch.linspace(bev_min_x, bev_max_x, W).reshape(
        (1, W)).repeat(H, 1)).double()
    y = (torch.linspace(bev_min_y, bev_max_y, H).reshape(
        (H, 1)).repeat(1, W)).double()
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords


def gen_offset(grid_size):
    W = grid_size[0]
    H = grid_size[1]
    bev_x = (torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)).double()
    bev_y = (torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)).double()
    bev_offset = torch.stack([bev_x, bev_y], axis=-1) * -1
    bev_offset = bev_offset.unsqueeze(0)
    return bev_offset


def wrap(bev_size, prev_ego2global, curr_ego2global, coords, offset, scale):
    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(bev_size)
    wrap_r, wrap_t = get_matrix(prev_ego2global, curr_ego2global, bev_max_x,
                                bev_max_y, bev_size)
    wrap_r = torch.tensor(wrap_r)
    wrap_t = torch.tensor(wrap_t)
    new_coords = []
    batch = wrap_r.shape[0]

    for i in range(batch):
        new_coord = torch.matmul(coords, wrap_r[i].double()).float()
        new_coord += wrap_t[i]
        new_coord += offset
        new_coords.append(new_coord)

    scale = np.array(scale, dtype=np.float32)
    scales = torch.tensor(scale).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(
        dim=3)
    new_coords = torch.cat(new_coords).permute(0, 3, 1, 2) / scales
    new_coords = torch.floor(new_coords + 0.5)
    new_coords = torch.clamp(new_coords, np.float32(-32768), np.float32(32767))
    array = new_coords.cpu().numpy().astype(np.int16)
    return array


def gen_bevformer_prevpoints(infos, prev_points_path):
    prev_meta = None
    for file_name, meta in infos.items():
        norm_coords = export_bev_transform_points(meta["metadata"], prev_meta)
        prev_meta = meta["metadata"]
        file_path = os.path.join(prev_points_path, file_name)
        prev_points = norm_coords.numpy()
        prev_points.tofile(file_path)


def get_min_max_coords_bevformer(
        real_range,
        grid_resolution,
) -> Tuple[float]:
    """Get min and max coords."""
    min_x = real_range[0] + grid_resolution[0] / 2
    min_y = real_range[1] + grid_resolution[1] / 2
    max_x = real_range[2] - grid_resolution[0] / 2
    max_y = real_range[3] - grid_resolution[1] / 2
    return min_x, max_x, min_y, max_y


def gen_coords_bevformer(bev_size: Tuple[int],
                         pc_range: Tuple[float]) -> torch.Tensor:
    """Generate coords."""
    real_w = pc_range[3] - pc_range[0]
    real_h = pc_range[4] - pc_range[1]

    W = bev_size[0]
    H = bev_size[1]

    grid_resolution = (real_w / W, real_h / H)
    real_range = (pc_range[0], pc_range[1], pc_range[3], pc_range[4])

    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords_bevformer(
        real_range,
        grid_resolution,
    )

    # Generate a tensor for the x-coordinates of the bird's eye view grid
    x = (torch.linspace(bev_min_x, bev_max_x, W).reshape(
        (1, W)).repeat(H, 1)).double()
    y = (torch.linspace(bev_min_y, bev_max_y, H).reshape(
        (H, 1)).repeat(1, W)).double()
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords


def export_bev_transform_points(cur_meta: Dict, pre_meta: Dict):
    """Get normed coords for bevfeat transformer."""
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    bev_size = (50, 50)
    real_w = pc_range[3] - pc_range[0]
    real_h = pc_range[4] - pc_range[1]
    bev_w = 50
    bev_h = 50
    grid_resolution = (
        real_w / bev_w,
        real_h / bev_h,
    )
    bs = 1
    real_range = (
        pc_range[0],
        pc_range[1],
        pc_range[3],
        pc_range[4],
    )

    coords = gen_coords_bevformer(bev_size, pc_range)

    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords_bevformer(
        real_range,
        grid_resolution,
    )

    bs_new_coords = []
    if pre_meta is not None:
        pre_scene = pre_meta["scene"]
        if pre_scene != cur_meta["scene"]:
            new_coords = coords.clone()
        else:
            new_coords = get_refine_coords(coords, cur_meta, pre_meta)
        bs_new_coords.append(new_coords)
    else:
        new_coords = coords.clone().repeat(bs, 1, 1, 1)
        bs_new_coords.append(new_coords)

    bs_new_coords = torch.cat(bs_new_coords, dim=0)
    bs_new_coords[..., 0] = (bs_new_coords[..., 0] - bev_min_x) / (
        (bev_max_x - bev_min_x))
    bs_new_coords[..., 1] = (bs_new_coords[..., 1] - bev_min_y) / (
        (bev_max_y - bev_min_y))

    norm_coords = (bs_new_coords * 2 - 1).to(torch.float32)
    return norm_coords


def get_refine_coords(
        coords,
        img_metas: Dict,
        img_metas_pre: Dict,
):
    bev_w = 50
    bev_h = 50
    """Get refine coords."""
    cur_e2g = img_metas["ego2global"]
    cur_e2g = np.array(cur_e2g)
    prev_e2g = img_metas_pre["ego2global"]
    prev_e2g = np.array(prev_e2g)
    prev_g2e = np.linalg.inv(prev_e2g)
    wrap_m = prev_g2e @ cur_e2g
    wrap_m = wrap_m[None, ...]
    wrap_r_t = wrap_m[:, :2, :3]
    # Extract the translation component
    trans = np.eye(3)[None, :, :]
    trans[:, :2, :3] = wrap_r_t
    output = trans.transpose((0, 2, 1))
    wrap_r1 = torch.tensor(output, dtype=torch.float32)
    ones = torch.ones((1, bev_h, bev_w, 1), dtype=torch.float32)
    coords_3d = torch.cat([coords, ones], dim=-1)
    new_coords_all = []
    batch = wrap_r1.shape[0]
    for i in range(batch):
        new_coord = torch.matmul(coords_3d, wrap_r1[i].double()).float()
        new_coords_all.append(new_coord)
    new_coords_all = torch.cat(new_coords_all)
    return new_coords_all[..., :2]


def preprocess(img_list, image_data_path, file_name, input_size=(512, 960)):
    print("image_data_path: ", image_data_path, ", file_name: ", file_name)
    orig_imgs = []
    for i, img in enumerate(img_list):
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        orig_imgs.append({"name": i, "img": img_bgr})
    input_imgs = []
    for image in orig_imgs:
        # bgr to nv12
        image = image["img"].astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        yuv420p = yuv420p.reshape((height * width * 3 // 2, ))
        y = yuv420p[:height * width]
        uv_planar = yuv420p[height * width:].reshape((2, height * width // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape(
            (height * width // 2, ))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        input_imgs.append(nv12)
    for idx, input_img in enumerate(input_imgs):
        input_img = np.array(input_img)
        new_path = image_data_path + f"_{idx}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        fn = os.path.join(new_path, file_name)
        with open(fn, "w") as sf:
            input_img.tofile(sf)


def process_reference_points(location, model, points_path, file_name, idx):
    city = location.split("-")[0]
    points_file = os.path.join(points_path, model, city) + str(idx) + ".npy"
    np.load(points_file).tofile(file_name)


def process_detr3d_input(reference_points_path, location, name,
                         coords_data_path, masks_data_path, points_path):
    city = location.split("-")[0]
    input_path = os.path.join(reference_points_path, city)

    # masks
    masks_save_path = os.path.join(masks_data_path, name)
    masks = np.load(input_path + "/masks.npy")
    masks.tofile(masks_save_path)

    # reference_points
    reference_path = os.path.join(points_path, name)
    reference_points = np.load(input_path + "/reference_points.npy").transpose(
        0, 1, 3, 2).transpose(0, 2, 1, 3)
    reference_points.tofile(reference_path)

    # coords
    coords0 = np.load(input_path + "/coords0.npy")
    coord_save_path0 = os.path.join(coords_data_path[0], name)
    coords0.tofile(coord_save_path0)

    coords1 = np.load(input_path + "/coords1.npy")
    coord_save_path1 = os.path.join(coords_data_path[1], name)
    coords1.tofile(coord_save_path1)

    coords2 = np.load(input_path + "/coords2.npy")
    coord_save_path2 = os.path.join(coords_data_path[2], name)
    coords2.tofile(coord_save_path2)

    coords3 = np.load(input_path + "/coords3.npy")
    coord_save_path3 = os.path.join(coords_data_path[3], name)
    coords3.tofile(coord_save_path3)


def process_petr_input(reference_points_path, location, name, pos_embed_path,
                       points_path):
    city = location.split("-")[0]
    input_path = os.path.join(reference_points_path, city)

    # pos_embed
    pos_embed_save_path = os.path.join(pos_embed_path, name)
    pos_embed = np.load(input_path + "/pos_embed.npy")
    pos_embed.tofile(pos_embed_save_path)

    # reference_points
    reference_path = os.path.join(points_path, name)
    reference_points = np.load(input_path + "/reference_points.npy").transpose(
        0, 1, 3, 2).transpose(0, 2, 1, 3)
    reference_points.tofile(reference_path)


def bbox_ego2bev(bboxes: torch.Tensor, bev_size: Tuple[float]) -> torch.Tensor:
    """Convert ego coordinate to bev coordinate.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        bev_size: Bev size.
    """

    bev_bboxes = []
    min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
    for bbox in bboxes:
        bbox = copy.deepcopy(bbox)
        bbox[0] += max_x
        bbox[1] += max_y
        bbox[:6] /= bev_size[2]
        bbox[7:9] /= bev_size[2]
        bev_bboxes.append(bbox)
    return bev_bboxes


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(
        patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type="index", angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == "backward":
        coords = np.flip(coords, 0)

    if type == "index":
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(
                mask,
                [coords[i:]],
                False,
                color=get_discrete_degree(
                    coords[i + 1] - coords[i], angle_class=angle_class),
                thickness=thickness,
            )
    return mask, idx


def line_geom_to_mask(
        layer_geom,
        confidence_levels,
        local_box,
        canvas_size,
        thickness,
        idx,
        type="index",
        angle_class=36,
):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == "MultiLineString":
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(
                        new_single_line,
                        map_mask,
                        thickness,
                        idx,
                        type,
                        angle_class,
                    )
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness,
                                               idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C - 1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, max_channel, thickness,
                   angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []

    for vector in vectors:
        if vector["pts_num"] >= 2:
            vector_num_list[vector["type"]].append(
                LineString(vector["pts"][:vector["pts_num"]]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            idx,
        )
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness + 4,
            1,
        )
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="forward",
            angle_class=angle_class,
        )
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="backward",
            angle_class=angle_class,
        )
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = (overlap_filter(forward_masks,
                                    filter_masks).sum(0).astype("int32"))
    backward_masks = (overlap_filter(backward_masks,
                                     filter_masks).sum(0).astype("int32"))

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks


class MultiViewsImgTransformWrapper(object):
    """Wrapper img transform for image inputs.

    Args:
       trnsforms: List of image transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Mapping):
        for i, img in enumerate(data["img"]):
            new_data = {
                "img": img,
                "layout": data["layout"],
                "color_space": data["color_space"],
            }
            for transform in self.transforms:
                new_data = transform(new_data)
            data["img"][i] = new_data["img"]
        return data

    def __repr__(self):
        return "MultiViewsImgTransformWrapper"


class MultiViewsImgPad(object):
    """Crop PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        random: Whether choosing min x randomly.
    """

    def __init__(
            self,
            padding: Sequence[int],
    ):
        self.padding = padding

    def __call__(self, data: Mapping):
        for i, img in enumerate(data["img"]):
            data["img"][i] = F.pad(img, padding=self.padding)
        return data


class MultiViewsImgCrop(object):
    """Crop PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        random: Whether choosing min x randomly.
    """

    def __init__(
            self,
            size: Tuple[int, int],
            random: bool = False,
    ):
        self.size = size if isinstance(size[0], Sequence) else [size]
        self.random = random

    def _crop(self, data: Image.Image, top, left, height, width):
        return F.crop(data, top, left, height, width)

    def __call__(self, data: Mapping):
        if len(self.size) == 1:
            sizes = self.size * len(data["img"])
        elif len(self.size) == len(data["img"]):
            sizes = self.size
        else:
            raise ValueError("Size should equal to img num or 1")

        for i, img in enumerate(data["img"]):
            size = sizes[i]
            top = img.size[1] - size[0]
            if self.random:
                left = int(np.random.uniform(0, max(0, img.size[0] - size[1])))
            else:
                left = (img.size[0] - size[1]) / 2
            data["img"][i] = self._crop(img, top, left, size[0], size[1])
        return data


class MultiViewsImgResize(object):
    """Resize PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        scales: Scale for random choosen.
        interpolation:Desired interpolation. Default is 'nearest'.
    """

    def __init__(
            self,
            size: Optional[Tuple[int, int]] = None,
            scales: Optional[Tuple[float, float]] = None,
            interpolation: str = "bilinear",
    ):
        self.size = size
        self.scales = scales
        assert interpolation in PIL_INTERP_CODES
        self.interpolation = interpolation

    def _resize(self, data: Image.Image, size):
        return F.resize(data, size, PIL_INTERP_CODES[self.interpolation])

    def __call__(self, data: Mapping):
        if self.size:
            if not isinstance(self.size[0], Sequence):
                self.size = [self.size]
            if len(self.size) == 1:
                sizes = self.size * len(data["img"])
            elif len(self.size) == len(data["img"]):
                sizes = self.size
            else:
                raise ValueError("Size should equal to img num or 1")
        elif self.scales:
            sizes = []
            for img in data["img"]:
                W, H = img.size
                resize = np.random.uniform(*self.scales)
                resize_dims = (int(H * resize), int(W * resize))
                sizes.append(resize_dims)
        else:
            raise ValueError("Size or scale should be set")
        for i, img in enumerate(data["img"]):
            data["img"][i] = self._resize(img, sizes[i])
        return data


class VectorizedLocalMap(object):
    def __init__(
            self,
            dataroot,
            patch_size,
            canvas_size,
            line_classes,
            ped_crossing_classes,
            contour_classes,
            sample_dist=1,
            num_samples=250,
            padding=False,
            normalize=False,
            fixed_num=-1,
            class2label=None,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        ]
        if class2label is None:
            class2label = {
                "road_divider": 0,
                "lane_divider": 0,
                "ped_crossing": 1,
                "contours": 2,
                "others": -1,
            }

        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation,
                               ego2global_rotation):
        map_pose = ego2global_translation[:2]
        if Quaternion is None:
            raise ModuleNotFoundError(
                "nuscenes-devkit is needed while using NuScenes dataset.")

        rotation = Quaternion(ego2global_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(patch_box, patch_angle,
                                      self.line_classes, location)
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(patch_box, patch_angle,
                                     self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)["ped_crossing"]

        polygon_geom = self.get_map_geom(patch_box, patch_angle,
                                         self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((
                    line.astype(float),
                    length,
                    self.class2label.get(line_type, -1),
                ))

        for ped_line, length in ped_vector_list:
            vectors.append((
                ped_line.astype(float),
                length,
                self.class2label.get("ped_crossing", -1),
            ))

        for contour, length in poly_bound_list:
            vectors.append((
                contour.astype(float),
                length,
                self.class2label.get("contours", -1),
            ))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    "pts": pts,
                    "pts_num": pts_num,
                    "type": type
                })

        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle,
                                                   location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for li in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(li))
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != "MultiPolygon":
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y,
                     line_list):
            points = [(p0, p1)
                      for p0, p1 in zip(poly_xy[0, idx:idx +
                                                2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(
                    line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                line = affinity.affine_transform(
                    line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []

        records = self.nusc_maps[location].ped_crossing
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(
                record["polygon_token"])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y,
                     line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y,
                     line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array(
                [self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding],
                                                axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array(
                    [self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid


class NuscenesSample(object):
    """Sample object for NuScenes dataset.

    Args:
        sample: Meta data for sample.
    """

    def __init__(self, sample: dict):
        self.sample = sample

    def __del__(self):
        del self.sample

    def get_scene(self):
        return self.sample["scene"]

    def get_timestamp(self):
        return self.sample["timestamp"]

    def get_can_bus(self):
        return self.sample["can_bus"]

    def get_bev_mask(self, ) -> np.array:
        """Get bev seg mask."""
        if "gt_mask" in self.sample:
            semantic_masks = self.sample["gt_mask"].astype(np.int64)
            num_cls = semantic_masks.shape[0]
            indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
            semantic_indices = np.sum(semantic_masks * indices, axis=0)
            return semantic_masks, semantic_indices
        raise ValueError("No gt_mask")

    def _get_homography_by_cam(self, cam):
        sensor2ego_translation = cam["sensor2ego_translation"]
        sensor2ego_rotation = cam["sensor2ego_rotation"]
        rotation = Quaternion(sensor2ego_rotation).rotation_matrix
        ego2sensor_r = np.linalg.inv(rotation)
        ego2sensor_t = sensor2ego_translation @ ego2sensor_r.T
        ego2sensor = np.eye(4)
        ego2sensor[:3, :3] = ego2sensor_r.T
        ego2sensor[3, :3] = -np.array(ego2sensor_t)

        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)

        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.
                shape[1]] = camera_intrinsic
        ego2img = viewpad @ ego2sensor.T
        return ego2img

    def get_cam_by_name(self, name: str) -> Dict:
        """Get cam info by cam name."""
        cams = self.sample["cam"]
        cam = None
        for c in cams:
            if name == c["name"]:
                cam = copy.deepcopy(c)
        if cam is None:
            raise ValueError(f"Cannot find cam {name}")
        cam["img"] = self._load_img(cam["img"])

        # cam["cam2ego"] = self._get_cam2ego(cam)
        # cam["ego2img"] = self._get_homography_by_cam(cam)
        return cam

    def get_ego2gloabl(self):
        e2g_t = np.array(self.sample["ego2global_translation"])
        e2g_r = np.array(self.sample["ego2global_rotation"])
        e2g_m = np.zeros((4, 4), dtype=np.float32)
        e2g_m[:3, :3] = Quaternion(e2g_r).rotation_matrix
        e2g_m[:3, 3] = np.array(e2g_t)
        e2g_m[3, 3] = 1.0

        return e2g_m

    def _get_cam2ego(self, cam):
        sensor2ego_translation = cam["sensor2ego_translation"]
        sensor2ego_rotation = cam["sensor2ego_rotation"]

        cam2ego = np.eye(4)
        rotation = Quaternion(sensor2ego_rotation).rotation_matrix
        cam2ego[:3, :3] = rotation
        cam2ego[:3, 3] = sensor2ego_translation
        return cam2ego

    def get_location(self):
        return self.sample["location"]

    def get_token(self):
        return self.sample["sample_token"]

    def get_ego2global_rotation(self):
        return self.sample["ego2global_rotation"]

    def get_ego2global_translation(self):
        return self.sample["ego2global_translation"]

    def get_meta(self):
        meta = {}

        exclude = ["cam"]
        for k, v in self.sample.items():
            if k not in exclude:
                meta[k] = v
        return meta

    def _load_img(self, img):
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return Image.fromarray(img)

    def get_center2d(self, bbox, cam):
        center3d = bbox[:3]
        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)
        center2d = camera_intrinsic @ center3d
        center2d[:2] = center2d[:2] / center2d[2]
        return center2d[:2], center2d[2]

    def get_corner2d(self, corners3d, cam, im_size):
        in_front = np.argwhere(corners3d[2, :] > 0).flatten()
        corners3d = corners3d[:, in_front]
        camera_intrinsic = cam["camera_intrinsic"]
        camera_intrinsic = np.array(camera_intrinsic)
        corner2d = camera_intrinsic @ corners3d
        corner2d = np.transpose(corner2d, (1, 0))
        corner2d[..., 0] = corner2d[..., 0] / corner2d[..., 2]
        corner2d[..., 1] = corner2d[..., 1] / corner2d[..., 2]

        if len(corner2d) == 0:
            min_x = 0
            min_y = 0
            max_x = 0
            max_y = 0
        else:
            min_x = min(corner2d[..., 0])
            min_y = min(corner2d[..., 1])
            max_x = max(corner2d[..., 0])
            max_y = max(corner2d[..., 1])

            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, im_size[0])
            max_y = min(max_y, im_size[1])

        return np.array([min_x, min_y, max_x, max_y])

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def get_lidar_points(
            self,
            num_sweeps,
            load_dim,
            use_dim,
            time_dim=4,
            pad_empty_sweeps=True,
            remove_close=True,
            test_mode=False,
    ):
        points = np.array(self.sample["lidar_points"], dtype=np.float32)
        ts = self.sample["timestamp"] / 1e6

        points = points.reshape(-1, load_dim)
        points[:, time_dim] = 0

        sweeps = self.sample["sweeps"]

        points_list = [points]

        if pad_empty_sweeps and len(sweeps) == 0:
            for _ in range(num_sweeps):
                if remove_close:
                    points_list.append(self._remove_close(points))
                else:
                    points_list.append(points)
        else:
            if len(sweeps) <= num_sweeps:
                choices = np.arange(len(sweeps))
            elif test_mode:
                choices = np.arange(num_sweeps)
            else:
                choices = np.random.choice(
                    len(sweeps), num_sweeps, replace=False)
            for idx in choices:
                sweep = sweeps[idx]
                points_sweep = np.array(
                    sweep["lidar_points"], dtype=np.float32)
                points_sweep = points_sweep.reshape(-1, load_dim)
                if remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (points_sweep[:, :3] @ np.array(
                    sweep["sensor2lidar_rotation"]).T)
                points_sweep[:, :3] += np.array(
                    sweep["sensor2lidar_translation"])
                points_sweep[:, time_dim] = ts - sweep_ts
                points_list.append(points_sweep)
        points = np.concatenate(points_list)
        points = points[:, use_dim]

        return points

    def get_lidar_ann_info(self,
                           use_valid_flag=False,
                           with_velocity=True,
                           classes=None):
        info = self.sample
        # filter out bbox containing no points
        if use_valid_flag:
            mask = np.array(info["valid_flag"], dtype=bool)
        else:
            mask = np.array(info["num_lidar_pts"]) > 0
        gt_bboxes_3d = np.array(info["gt_boxes"]).reshape(-1, 7)[mask]

        gt_names_3d = []
        for obj in info["gt_names"]:
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            gt_names_3d.append(obj)
        gt_names_3d = np.array(gt_names_3d)[mask]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in classes:
                gt_labels_3d.append(classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if with_velocity:
            gt_velocity = np.array(info["gt_velocity"]).reshape(-1, 2)[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        locs = gt_bboxes_3d[:, :3]
        dims = gt_bboxes_3d[:, 3:6]
        rots = gt_bboxes_3d[:, 6:7]
        vel = gt_bboxes_3d[:, 7:9]
        gt_bboxes_3d = np.concatenate(
            [locs, dims[:, [1, 0, 2]], vel, -rots - np.pi / 2], axis=-1)

        anns_results = dict(  # noqa C408
            boxes=gt_bboxes_3d,
            labels=gt_labels_3d,
            names=gt_names_3d,
        )
        return anns_results

    def num_cams(self):
        return len(self.sample["cam"])

    def _get_ego_occ(self):
        voxel_semantics = np.frombuffer(
            self.sample["voxel_semantics"], dtype=np.uint8).reshape(
                200, 200, 16)

        mask_lidar = np.frombuffer(
            self.sample["mask_lidar"], dtype=np.uint8).reshape(200, 200, 16)

        mask_camera = np.frombuffer(
            self.sample["mask_camera"], dtype=np.uint8).reshape(200, 200, 16)

        gt_occ = {
            "voxel_semantics": voxel_semantics,
            "mask_lidar": mask_lidar,
            "mask_camera": mask_camera,
        }
        return gt_occ


class NuscenesBevSampler(object):
    """Bev Dataset object for packed NuScenes.

    Args:
        with_bev_bboxes: Whether include bev bboxes.
        with_bev_mask: Whether include bev bboxes.
        with_ego_occ: Whether include ego occ. Default: False.
        map_path: Path to Nuscenes Map, needed if include bev mask.
        line_classes: Classes of line. ex. road divider, lane divider.
        ped_crossing_classes: Classes of ped corssing. ex. ped_crossing
        contour_classes: Classes of contour. ex. road segment, lane.
        bev_size: Size for bev using meter. ex. (51.2, 51.2, 0.2)
        bev_range: range for bev, alternative of bev_size.
            ex.(-61.2, -61.2, -2, 61.2, 61.2, 10)
        map_size: size for seg map.
    """

    def __init__(
            self,
            with_bev_bboxes: bool = True,
            with_ego_bboxes: bool = False,
            with_ego_occ: bool = False,
            with_bev_mask: bool = True,
            map_path: Optional[str] = None,
            line_classes=None,
            ped_crossing_classes=None,
            contour_classes=None,
            bev_size: Optional[Tuple] = None,
            bev_range: Optional[Tuple] = None,
            map_size: Optional[Tuple] = None,
            **kwargs,
    ):
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.map_path = map_path
        self.map_size = map_size

        self.with_bev_mask = with_bev_mask
        self.with_bev_bboxes = with_bev_bboxes
        self.with_ego_bboxes = with_ego_bboxes
        self.with_ego_occ = with_ego_occ

        if self.with_bev_mask is True:
            self.patch_size = (
                self.map_size[0] * 2,
                self.map_size[1] * 2,
            )
            self.canvas_size = (
                int(self.map_size[0] * 2 / self.map_size[2]),
                int(self.map_size[1] * 2 / self.map_size[2]),
            )

            if line_classes is None:
                line_classes = ["road_divider", "lane_divider"]
            if ped_crossing_classes is None:
                ped_crossing_classes = ["ped_crossing"]
            if contour_classes is None:
                contour_classes = ["road_segment", "lane"]
            self.vector_map = VectorizedLocalMap(
                map_path,
                self.patch_size,
                self.canvas_size,
                line_classes=line_classes,
                ped_crossing_classes=ped_crossing_classes,
                contour_classes=contour_classes,
            )

    def _get_map_info(self, sample):
        vectors = self.vector_map.gen_vectorized_samples(
            sample["location"],
            sample["ego2global_translation"],
            sample["ego2global_rotation"],
        )

        for vector in vectors:
            pts = vector["pts"]
            vector["pts"] = np.concatenate((pts, np.zeros((pts.shape[0], 1))),
                                           axis=1)

        for vector in vectors:
            vector["pts"] = vector["pts"][:, :2]

        (
            semantic_masks,
            instance_masks,
            forward_masks,
            backward_masks,
        ) = preprocess_map(vectors, self.patch_size, self.canvas_size, 3, 5,
                           36)
        sample["gt_mask"] = semantic_masks

    def __len__(self):
        return len(self.samples)

    def __call__(self, sample):
        data = {}

        if self.map_path and self.with_bev_mask:
            self._get_map_info(sample)
        sample = NuscenesSample(sample)

        if self.with_bev_mask:
            (
                data["bev_seg_mask"],
                data["bev_seg_indices"],
            ) = sample.get_bev_mask()

        cam_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        imgs = []
        img_names = []
        for name in cam_names:
            cam = sample.get_cam_by_name(name)
            img_names.append(cam["img_path"])
            imgs.append(cam["img"])
        data["sample_idx"] = sample.get_token()
        data["img_name"] = img_names
        data["img"] = imgs
        data["location"] = sample.get_location()

        data["ego2global"] = sample.get_ego2gloabl()

        data["detection"] = {}
        data["detection"][
            "ego2global_rotation"] = sample.get_ego2global_rotation()
        data["detection"][
            "ego2global_translation"] = sample.get_ego2global_translation()

        data["metadata"] = {}
        data["metadata"]["detection"] = data["detection"]
        data["metadata"]["segment"] = data["bev_seg_indices"]
        data["metadata"]["ego2global"] = data["ego2global"]
        data["metadata"]["scene"] = sample.get_scene()

        if self.with_ego_occ:
            data["gt_occ_info"] = sample._get_ego_occ()
            data["metadata"]["gt_occ_info"] = data["gt_occ_info"]

        return data


class NuscenesParser(object):
    """Parser object for packed NuScenes.

    Args:
        version: Version for nuscenes.
        src_data_dir: Path for data.
        split_name: Split_name for dataset.(ex. "train", "val")
        max_sweeps: Max number of sweeps for packed lidar points.
        with_ego_occ: Whether include ego occ. Default: False.
    """

    def __init__(
            self,
            version: str,
            src_data_dir: str,
            split_name: str = "val",
            max_sweeps: int = 10,
            with_ego_occ: bool = False,
    ):
        if NuScenes is None:
            raise ModuleNotFoundError(
                "nuscenes-devkit is needed while using NuScenes dataset.")

        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True)

        self.max_sweeps = max_sweeps
        # self.nusc_can_bus = NuScenesCanBus(dataroot=src_data_dir)
        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == "v1.0-test":
            train_scenes = splits.test
            val_scenes = []
        elif version == "v1.0-mini":
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError("unknown")

        self.test = "test" in version
        self.root_path = src_data_dir
        self.with_ego_occ = with_ego_occ
        if split_name == "train":
            available_scenes = train_scenes
        else:
            available_scenes = val_scenes

        self.scenes = [
            scene["token"] for scene in self.nusc.scene
            if scene["name"] in available_scenes
        ]

        self.samples = []
        
        # scene.json:24:"first_sample_token": "8687ba92abd3406aa797115b874ebeba",
        # scene.json:33:"first_sample_token": "5991fad3280c4f84b331536c32001a04",
        # scene.json:42:"first_sample_token": "cd9964f8c3d34383b16e9c2997de1ed0",
        # scene.json:51:"first_sample_token": "c1676a2feac74eee8aa38ca3901787d6",
        # scene.json:60:"first_sample_token": "b5989651183643369174912bc5641d3b",
        # scene.json:69:"first_sample_token": "5998b71b64c146769bde1d5430741381",
        # scene.json:78:"first_sample_token": "e6b0b282aa174a978272dc2d0a89d560",
        # scene.json:87:"first_sample_token": "a480496a5988410fbe3d8ed6c84da996",

        # print("self.scenes len: ", self.scenes.len())

        # TODO
        self.scenes = [
            "cc8c0bf57f984915a77078b10eb33198",
            "fcbccedd61424f1b85dcbf8f897f9754",
            "6f83169d067343658251f72e1dd17dbc",
            "bebf5f5b2a674631ab5c88fd1aa9e87a",
            "2fc3753772e241f2ab2cd16a784cc680",
            "c5224b9b454b4ded9b5d2d2634bbda8a",
            "325cef682f064c55a255f2625c533b75",
            "d25718445d89453381c659b9c8734939",
            "de7d80a1f5fb4c3e82ce8a4f213b450a",
            "e233467e827140efa4b42d2b4c435855"
        ]

        for scene_token in self.scenes:
            scene = self.nusc.get("scene", scene_token)

            print("scene_token: ", scene_token)
            print("scene: ", scene)

            next_sample_token = scene["first_sample_token"]
            print("first_sample_token: ", next_sample_token)
            while next_sample_token != "":
                self.samples.append(next_sample_token)
                print("scene_token: ", next_sample_token)
                sample = self.nusc.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                print("next: ", next_sample_token)
            # while next_sample_token != "end":
            #     print("scene_token: ", next_sample_token)
            #     if next_sample_token != "":
            #         self.samples.append(next_sample_token)
            #     sample = self.nusc.get("sample", next_sample_token)
            #     next_sample_token = sample["next"]
            #     print("next: ", next_sample_token)

    def _gen_info(self, sample_token):
        sample = self.nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene = self.nusc.get("scene", scene_token)
        loc = self.nusc.get("log", scene["log_token"])["location"]
        sample_info = {"scene": scene_token}
        sample_info["location"] = loc
        sample_info["sample_token"] = sample_token
        sample_info["prev"] = sample["prev"]
        sample_info["next"] = sample["next"]
        sample_info["scene_token"] = scene_token
        sample_info["timestamp"] = sample["timestamp"]
        self._gen_lidar_info(sample, sample_info)
        self._gen_cam_info(sample, sample_info)
        if self.with_ego_occ:
            self._gen_occ_gt(scene, sample_info)
        return sample_info

    def _gen_anns(self, sample, info):
        anns = sample["anns"]
        gt_bboxes = []
        for ann_token in anns:
            ann = self.nusc.get("sample_annotation", ann_token)
            num_lidar_pts = ann["num_lidar_pts"]
            num_radar_pts = ann["num_radar_pts"]

            center = np.array(ann["translation"])
            wlh = np.array(ann["size"])
            rot = Quaternion(ann["rotation"]).elements.tolist()
            rot = np.array(rot)

            attr_token = ann["attribute_tokens"]
            if len(attr_token) == 0:
                attr_name = "None"
            else:
                attr_name = self.nusc.get("attribute", attr_token[0])["name"]
            attr_id = Attributes.index(attr_name)

            global_velo3d = self.nusc.box_velocity(ann_token)[:2].tolist()
            gt_bbox = {
                "bbox": np.concatenate([center, wlh], axis=0).tolist(),
                "rot": rot.tolist(),
                "attr_name": attr_name,
                "attr_id": attr_id,
                "cat": ann["category_name"],
                "token": ann_token,
                "velocity": global_velo3d,
                "num_lidar_pts": num_lidar_pts,
                "num_radar_pts": num_radar_pts,
            }
            gt_bboxes.append(gt_bbox)
        info["gt_bboxes"] = gt_bboxes

    def _gen_occ_gt(self, scene, info):
        occ_path = 'gts/%s/%s' % (scene['name'], info['sample_token'])
        occ_gt_path = os.path.join(self.root_path, occ_path, "labels.npz")
        occ_labels = np.load(occ_gt_path)

        info["voxel_semantics"] = occ_labels["semantics"].tobytes()
        info["mask_lidar"] = occ_labels["mask_lidar"].tobytes()
        info["mask_camera"] = occ_labels["mask_camera"].tobytes()

    def _load_img(self, img_path):
        img_path = os.path.join(self.root_path, img_path)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def _gen_cam_info(self, sample, info):
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        cams = []
        for name, token in sample["data"].items():
            if name in camera_types:
                sd_rec = self.nusc.get("sample_data", token)
                pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
                cam = {
                    "img_path": sd_rec["filename"],
                    "img": self._load_img(sd_rec["filename"]),
                    "name": name,
                    "ego2global_translation": pose_rec["translation"],
                    "ego2global_rotation": pose_rec["rotation"],
                }
                cams.append(cam)
        info["cam"] = cams

    def _obtain_sensor2top(
            self,
            sensor_token,
            l2e_t,
            l2e_r_mat,
            e2g_t,
            e2g_r_mat,
            sensor_type="lidar",
    ):
        sd_rec = self.nusc.get("sample_data", sensor_token)
        cs_record = self.nusc.get("calibrated_sensor",
                                  sd_rec["calibrated_sensor_token"])
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path = sd_rec["filename"]
        points = self._read_points(lidar_path)
        sweep = {
            "lidar_points": points.tolist(),
            "type": sensor_type,
            "sample_data_token": sd_rec["token"],
            "sensor2ego_translation": cs_record["translation"],
            "sensor2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sd_rec["timestamp"],
        }
        l2e_r_s = sweep["sensor2ego_rotation"]
        l2e_t_s = sweep["sensor2ego_translation"]
        e2g_r_s = sweep["ego2global_rotation"]
        e2g_t_s = sweep["ego2global_translation"]

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= (e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
              + l2e_t @ np.linalg.inv(l2e_r_mat).T)
        sweep["sensor2lidar_rotation"] = R.T.tolist()  # points @ R.T + T
        sweep["sensor2lidar_translation"] = T.tolist()
        return sweep

    def _read_points(self, lidar_path):
        lidar_file = os.path.join(self.root_path, lidar_path)
        points = np.fromfile(lidar_file, dtype=np.float32)
        return points

    def _gen_lidar_info(self, sample, info):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)
        lidar_path = sd_rec["filename"]
        points = self._read_points(lidar_path)
        info["lidar_points"] = points.tolist()
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])

    @property
    def num_available_scenes(self):
        return len(self.scenes)


class NuscenesFromImage(Dataset):
    """Read NuScenes from image.

    Args:
        version: Version for nuscenes.
        src_data_dir: Path for data.
        split_name: Split_name for dataset.(ex. "train", "val")
    """

    def __init__(
            self,
            version,
            src_data_dir,
            split_name="train",
            transforms=None,
            with_bev_bboxes: bool = True,
            with_ego_bboxes: bool = False,
            with_ego_occ: bool = False,
            with_bev_mask: bool = True,
            map_path: Optional[str] = None,
            line_classes=None,
            ped_crossing_classes=None,
            contour_classes=None,
            bev_size: Optional[Tuple] = None,
            bev_range: Optional[Tuple] = None,
            map_size: Optional[Tuple] = None,
    ):
        self.dataset = NuscenesParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            with_ego_occ=with_ego_occ
        )
        self.sampler = NuscenesBevSampler(
            with_bev_bboxes=with_bev_bboxes,
            with_ego_bboxes=with_ego_bboxes,
            with_bev_mask=with_bev_mask,
            map_path=map_path,
            line_classes=line_classes,
            ped_crossing_classes=ped_crossing_classes,
            contour_classes=contour_classes,
            bev_size=bev_size,
            bev_range=bev_range,
            map_size=map_size,
            with_ego_occ=with_ego_occ)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.sampler(self.dataset[index])
        if self.transforms is not None:
            data = self.transforms(data)
        return data


supported_models = [
    'bev_gkt_mixvargenet_multitask_nuscenes',
    'bev_ipm_4d_efficientnetb0_multitask_nuscenes',
    'bev_ipm_efficientnetb0_multitask_nuscenes',
    'bev_lss_efficientnetb0_multitask_nuscenes',
    'detr3d_efficientnetb3_nuscenes', 'petr_efficientnetb3_nuscenes',
    'bevformer_tiny_resnet50_detection_nuscenes',
    "flashocc_henet_lss_occ3d_nuscenes"
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=supported_models,
    help="model name",
    required=True)
parser.add_argument(
    "--data-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--meta-path", type=str, help="path to dataset meta", required=True)
parser.add_argument(
    "--reference-path",
    type=str,
    help="path to reference points",
    required=True)
parser.add_argument(
    "--save-path", type=str, help="output path", default="./nuscenes_bev")
parser.add_argument(
    "--with-occ-gt",
    action='store_true',
    help="Include 'occ_gt' if this flag is set."
)

args = parser.parse_args()

save_path = args.save_path
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

image_data_path = os.path.join(save_path, "images")

resize_size = (540, 960)
input_size = (512, 960)
bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)
map_size = (15, 30, 0.15)

reference_points_num = 1
points_data_path = []
coords_data_path = []
masks_data_path = ""
points_path = ""
pos_embed_path = ""

# bev_gkt_mixvargenet_multitask_nuscenes
if args.model == supported_models[0]:
    resize_size = (540, 960)
    input_size = (512, 960)
    reference_points_num = 9
# bev_ipm_4d_efficientnetb0_multitask_nuscenes,
# bev_ipm_efficientnetb0_multitask_nuscenes
elif args.model == supported_models[1] or args.model == supported_models[2]:
    resize_size = (540, 960)
    input_size = (512, 960)
    reference_points_num = 1
# bev_lss_efficientnetb0_multitask_nuscenes
elif args.model == supported_models[3]:
    resize_size = (396, 704)
    input_size = (256, 704)
    reference_points_num = 2
# detr3d_efficientnetb3_nuscenes, petr_efficientnetb3_nuscenes
elif args.model == supported_models[4] or args.model == supported_models[5]:
    resize_size = (792, 1408)
    input_size = (512, 1408)
    reference_points_num = 1
elif args.model == supported_models[6]:
    resize_size = (450, 800)
    input_size = (480, 800)
    padding = (0, 0, 0, 30)
    reference_points_num = 4
# flashocc_henet_lss_occ3d_nuscenes
elif args.model == supported_models[7]:
    resize_size = (540, 960)
    input_size = (512, 960)
    reference_points_num = 2

if args.reference_path:
    if args.model == supported_models[4]:
        for idx in range(4):
            coords_path = os.path.join(save_path, "coords" + str(idx))
            os.makedirs(coords_path)
            coords_data_path.append(coords_path)
        masks_data_path = os.path.join(save_path, "masks")
        os.makedirs(masks_data_path)
        points_path = os.path.join(save_path, "reference_points")
        os.makedirs(points_path)
    elif args.model == supported_models[5]:
        pos_embed_path = os.path.join(save_path, "pos_embed")
        os.makedirs(pos_embed_path)
        points_path = os.path.join(save_path, "reference_points")
        os.makedirs(points_path)
    else:
        for idx in range(reference_points_num):
            points_path = "reference_points" + str(idx)
            reference_points_path = os.path.join(save_path, points_path)
            os.makedirs(reference_points_path)
            points_data_path.append(reference_points_path)

transforms = torchvision.transforms.Compose([
    MultiViewsImgResize(size=resize_size),
    MultiViewsImgCrop(size=input_size),
])

if args.model == supported_models[6]:
    transforms = torchvision.transforms.Compose([
        MultiViewsImgResize(size=resize_size),
        MultiViewsImgPad(padding=padding),
    ])

nuscenes_dataset = NuscenesFromImage(
    # TODO ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    version="v1.0-trainval",
    src_data_dir=args.data_path,
    split_name="val",
    bev_size=bev_size,
    map_size=map_size,
    map_path=args.meta_path,
    transforms=transforms,
    with_ego_occ=args.with_occ_gt)


def target(data):
    file_name = data["sample_idx"] + ".bin"
    preprocess(data["img"], image_data_path, file_name, input_size)
    if args.reference_path:
        if args.model == supported_models[4]:
            rp_path = os.path.join(args.reference_path, args.model)
            process_detr3d_input(rp_path, data["location"], file_name,
                                 coords_data_path, masks_data_path,
                                 points_path)
        elif args.model == supported_models[5]:
            rp_path = os.path.join(args.reference_path, args.model)
            process_petr_input(rp_path, data["location"], file_name,
                               pos_embed_path, points_path)
        else:
            for idx, path in enumerate(points_data_path):
                reference_point_name = os.path.join(path, file_name)
                process_reference_points(data["location"], args.model,
                                         args.reference_path,
                                         reference_point_name, idx)

    return file_name, {"metadata": data["metadata"]}


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)

    print(len(nuscenes_dataset))


    infos = {}
    features = []
    for i in tqdm(range(len(nuscenes_dataset))):
        data = nuscenes_dataset[i]
        features.append(pool.apply_async(target, (data, )))
    infos = {e[0]: e[1] for e in [e.get() for e in features]}
    pool.close()
    pool.join()

    # assert len(list(infos.keys())) == len(nuscenes_dataset)

    gt_anno_file = os.path.join(args.save_path, "val_gt_infos.pkl")

    with open(gt_anno_file, 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert os.path.exists(gt_anno_file)

    if args.model == supported_models[1]:
        prev_points_path = os.path.join(args.save_path, "prev_points")
        os.makedirs(prev_points_path)

        coords = gen_coords(bev_size, grid_size)
        offset = gen_offset(grid_size)

        scale = [1 / ele if ele != 0 else 0 for ele in list(grid_size)]
        gen_prev_points(infos, scale, coords, offset, prev_points_path)

    elif args.model == supported_models[6]:
        prev_points_path = os.path.join(args.save_path, "prev_points")
        os.makedirs(prev_points_path)
        gen_bevformer_prevpoints(infos, prev_points_path)

    print('======= Finish =======')
