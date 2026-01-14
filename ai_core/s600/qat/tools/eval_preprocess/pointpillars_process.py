# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import shutil
from typing import Dict, List

import numpy as np
import numba
from skimage import io
import tqdm
import pickle


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    return mat


@numba.njit
def _points_in_convex_polygon_3d_jit(
        points: np.ndarray,
        polygon_surfaces: np.ndarray,
        normal_vec: np.ndarray,
        d: np.ndarray,
        num_surfaces: np.ndarray = None,
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3] array. All surfaces' normal vector
            must direct to internal.
            max_num_points_of_surface must at least 3.
        normal_vec: normal vectors.
        d: distance matrix.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain.
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (points[i, 0] * normal_vec[j, k, 0] +
                        points[i, 1] * normal_vec[j, k, 1] +
                        points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.njit
def surface_equ_3d_jitv2(surfaces: np.ndarray):
    """Calculate surface normal vectors and distances.

    Args:
        surfaces (np.ndarray): [M, S, 3, 3] tensor of polygon surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has.
            3 = points that define a suface.
            3 = xyz coords of a point.

    Returns:
        normal_vec (np.ndarray): [M, S, 3] tensor of normal vectors.
        d (np.ndarray): [M, S] distance matrix. Note that this distance might
            not be normalized.
    """
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3),
                          dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            # Cross product of 2 surface vectors to get normal vector
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (-surfaces[i, j, 0, 0] * normal_vec[i, j, 0] -
                       surfaces[i, j, 0, 1] * normal_vec[i, j, 1] -
                       surfaces[i, j, 0, 2] * normal_vec[i, j, 2])
    return normal_vec, d


def projection_matrix_to_CRT_kitti(proj):
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """Corner_to_surfaces_3d_jit.

    Convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0,
        1,
        2,
        3,
        7,
        6,
        5,
        4,
        0,
        3,
        7,
        4,
        1,
        5,
        6,
        2,
        0,
        4,
        5,
        1,
        3,
        2,
        6,
        7,
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def points_in_convex_polygon_3d_jit(
        points: np.ndarray,
        polygon_surfaces: np.ndarray,
        num_surfaces: np.ndarray = None,
):
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain.
    Returns:
        [num_points, num_polygon] bool array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


class Kitti3dData(object):
    """Kitti3D dataset processor.

    Args:
        dir_root (str): Root directory path of Kitti3D dataset.
          And the directory structure of `dir_root` should be like this:
            ├──── `dir_root`
            │   ├── ImageSets
            │   │   ├── val.txt
            │   │   ├── ...
            │   ├── training
            │   │   ├── calib
            │   │   ├── image_2
            │   │   ├── label_2
            │   │   ├── velodyne
            │   ├── testing
            │   │   ├── ...
        split (str): Dataset split, should be "training" during inference.
    """

    def __init__(self, dir_root: str, split: str = "training"):
        assert split == "training", \
            f"{split} should be 'training', but get {split}"

        self.dir_root = dir_root
        self.val_split = os.path.join(self.dir_root, "ImageSets", "val.txt")
        self.dir_calib = os.path.join(self.dir_root, split, "calib")
        self.dir_imgs = os.path.join(self.dir_root, split, "image_2")
        self.dir_labels = os.path.join(self.dir_root, split, "label_2")
        self.dir_velodyne = os.path.join(self.dir_root, split, "velodyne")

    @property
    def val_data_ids(self) -> List[int]:
        """Get all index of validation dataset.

        Returns:
            List[int]: All index of validation dataset.
        """
        assert os.path.exists(
            self.val_split), f"{self.val_split} does not exist."
        with open(self.val_split, "r") as f:
            lines = f.readlines()
        return [int(line) for line in lines]

    def get_calib(self, index: int, extend_matrix: bool = True) -> Dict:
        """Get the calibration information of one sample.

        Args:
            index (int): Int value in sample name. For example,
                the `index` value of sample '000026.bin' will be `int(26)`.
            extend_matrix (bool): 
                Whether to pad calibration matrix from shape (3, 4) to (4,4).

        Returns:
            Dict: Calibration info.
        """
        calib_path = os.path.join(self.dir_calib, "{:06d}.txt".format(index))
        assert os.path.exists(calib_path), f"{calib_path} does not exists."

        calib_info = {"calib_idx": index}
        with open(calib_path, "r") as f:
            lines = f.readlines()
        P0 = np.array([float(info)
                       for info in lines[0].split(" ")[1:13]]).reshape([3, 4])
        P1 = np.array([float(info)
                       for info in lines[1].split(" ")[1:13]]).reshape([3, 4])
        P2 = np.array([float(info)
                       for info in lines[2].split(" ")[1:13]]).reshape([3, 4])
        P3 = np.array([float(info)
                       for info in lines[3].split(" ")[1:13]]).reshape([3, 4])

        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)

        R0_rect = np.array(
            [float(info) for info in lines[4].split(" ")[1:10]]).reshape(
                [3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.0
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        Tr_velo_to_cam = np.array([
            float(info) for info in lines[5].split(" ")[1:13]
        ]).reshape([3, 4])
        Tr_imu_to_velo = np.array([
            float(info) for info in lines[6].split(" ")[1:13]
        ]).reshape([3, 4])
        if extend_matrix:
            Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
        calib_info["P0"] = P0
        calib_info["P1"] = P1
        calib_info["P2"] = P2
        calib_info["P3"] = P3
        calib_info["R0_rect"] = rect_4x4
        calib_info["Tr_velo_to_cam"] = Tr_velo_to_cam
        calib_info["Tr_imu_to_velo"] = Tr_imu_to_velo
        return calib_info

    def get_img(self, index: int) -> Dict:
        """Get the image information of one sample.

        Args:
            index (int): Int value in sample name.

        Returns:
            Dict: Image info.
        """
        img_path = os.path.join(self.dir_imgs, "{:06d}.png".format(index))
        assert os.path.exists(img_path), f"{img_path} does not exists."
        img_info = {"image_idx": index}
        image_shape = np.array(io.imread(img_path).shape, dtype=np.int32)

        img_info["image_shape"] = image_shape

        return img_info

    def get_velodyne(self, index: int) -> np.ndarray:
        """Get the points cloud data of one sample.

        Args:
            index (int): Int value in sample name.

        Returns:
            np.ndarray: Points cloud data.
        """
        velodyne_path = os.path.join(self.dir_velodyne,
                                     "{:06d}.bin".format(index))
        assert os.path.exists(
            velodyne_path), f"{velodyne_path} does not exists."
        points = np.fromfile(
            velodyne_path, dtype=np.float32, count=-1).reshape((-1, 4))
        return points

    def get_label_annotation(self, index):
        label_path = os.path.join(self.dir_labels, "{:06d}.txt".format(index))
        assert os.path.exists(label_path), f"{label_path} does not exists."
        calib_info = self.get_calib(index)
        img_info = self.get_img(index)
        annotations = {}
        annotations.update({
            "image_idx": index,
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "calib": {},
            "image_shape": img_info['image_shape'],
        })
        with open(label_path, "r") as f:
            lines = f.readlines()
        content = [line.strip().split(" ") for line in lines]
        num_objects = len([x[0] for x in content if x[0] != "DontCare"])
        annotations["name"] = np.array([x[0] for x in content])
        num_gt = len(annotations["name"])
        annotations["truncated"] = np.array([float(x[1]) for x in content])
        annotations["occluded"] = np.array([int(x[2]) for x in content])
        annotations["alpha"] = np.array([float(x[3]) for x in content])
        annotations["bbox"] = np.array([[float(info) for info in x[4:8]]
                                        for x in content]).reshape(-1, 4)
        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations["dimensions"] = np.array(
            [[float(info) for info in x[8:11]] for x in content]).reshape(
                -1, 3)[:, [2, 0, 1]]
        annotations["location"] = np.array([[float(info) for info in x[11:14]]
                                            for x in content]).reshape(-1, 3)
        annotations["rotation_y"] = np.array(
            [float(x[14]) for x in content]).reshape(-1)
        if len(content) != 0 and len(content[0]) == 16:  # have score
            annotations["score"] = np.array([float(x[15]) for x in content])
        else:
            annotations["score"] = np.zeros((annotations["bbox"].shape[0], ))
        idx = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations["index"] = np.array(idx, dtype=np.int32)
        annotations["group_ids"] = np.arange(num_gt, dtype=np.int32)
        annotations["calib"] = calib_info
        return annotations

    def generate_reduced_velodyne(self, index: int) -> np.ndarray:
        """Generate reduced points cloud data of one sample.

        Args:
            index (int): Int value in sample name.

        Returns:
            np.ndarray: Reduced points cloud data
        """
        image_info = self.get_img(index)
        calib_info = self.get_calib(index)
        points = self.get_velodyne(index)

        rect = calib_info["R0_rect"]
        P2 = calib_info["P2"]
        Trv2c = calib_info["Tr_velo_to_cam"]

        points_reduced = remove_outside_points(points, rect, Trv2c, P2,
                                               image_info["image_shape"])
        return points_reduced


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--save-path", type=str, help="output path", default="./kitti3d")
parser.add_argument("--height", type=int, help="input height", default=1)
parser.add_argument("--width", type=int, help="input width", default=150000)
args = parser.parse_args()

save_path = args.save_path
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

velodyne_save_path = os.path.join(args.save_path, "reduced_velodyne")
if not os.path.exists(velodyne_save_path):
    os.makedirs(velodyne_save_path)

kitti3d_data = Kitti3dData(args.data_path)

# Note: The number of samples in the Kitti3d validation dataset is 3769
# assert len(kitti3d_data.val_data_ids) == 3769

keystr = '_' + str(args.height) + '_' + str(args.width)
padding_np = (np.ones(args.width * 4) * -100).astype(np.float32).reshape(
    1, 1, -1, 4)

gt_annos = {}
for idx in tqdm.tqdm(kitti3d_data.val_data_ids):
    reduced_velodyne = kitti3d_data.generate_reduced_velodyne(idx).reshape(
        1, 1, -1, 4)
    padding_bin = np.concatenate((reduced_velodyne, padding_np), axis=2)
    target_bin = padding_bin[:, :, :args.width, :]
    save_file = os.path.join(velodyne_save_path, "{:06d}{}{}.bin".format(
        idx, keystr, keystr))
    with open(save_file, "w") as f:
        target_bin.tofile(f)

    anno = kitti3d_data.get_label_annotation(idx)
    gt_annos[idx] = anno

assert len(os.listdir(velodyne_save_path)) == len(kitti3d_data.val_data_ids)
assert len(list(gt_annos.keys())) == len(kitti3d_data.val_data_ids)

gt_anno_file = os.path.join(args.save_path, "val_gt_infos.pkl")

with open(gt_anno_file, 'wb') as handle:
    pickle.dump(gt_annos, handle, protocol=pickle.HIGHEST_PROTOCOL)

assert os.path.exists(gt_anno_file)
print('======= Finish =======')
