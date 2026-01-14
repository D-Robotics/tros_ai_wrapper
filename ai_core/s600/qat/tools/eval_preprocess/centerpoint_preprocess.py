# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import pickle
import shutil

from nuscenes.nuscenes import NuScenes

import numpy as np
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils import splits
from torch.utils.data import Dataset
from tqdm import tqdm

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


class NuscenesParser(object):
    """Parser object for packed NuScenes.

    Args:
        version: Version for nuscenes.
        src_data_dir: Path for data.
        split_name: Split_name for dataset.(ex. "train", "val")
        max_sweeps: Max number of sweeps for packed lidar points.
    """

    def __init__(
            self,
            version: str,
            src_data_dir: str,
            split_name: str = "val",
            max_sweeps: int = 10,
    ):
        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True)
        self.max_sweeps = max_sweeps
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
        if split_name == "train":
            available_scenes = train_scenes
        else:
            available_scenes = val_scenes

        self.scenes = [
            scene["token"] for scene in self.nusc.scene
            if scene["name"] in available_scenes
        ]

        self.samples = []
        for scene_token in self.scenes:
            scene = self.nusc.get("scene", scene_token)
            next_sample_token = scene["first_sample_token"]
            while next_sample_token != "":
                self.samples.append(next_sample_token)
                sample = self.nusc.get("sample", next_sample_token)
                next_sample_token = sample["next"]

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
        return sample_info

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
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)

        cs_rec = self.nusc.get("calibrated_sensor",
                               sd_rec["calibrated_sensor_token"])
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        info["lidar2ego_translation"] = cs_rec["translation"]
        info["lidar2ego_rotation"] = cs_rec["rotation"]
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]
        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain sweeps for a single key-frame
        sweeps = []
        while len(sweeps) < self.max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = self._obtain_sensor2top(sd_rec["prev"], l2e_t,
                                                l2e_r_mat, e2g_t, e2g_r_mat,
                                                "lidar")
                sweeps.append(sweep)
                sd_rec = self.nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        # obtain annotation
        if not self.test:
            annotations = [
                self.nusc.get("sample_annotation", token)
                for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array([
                self.nusc.box_velocity(token)[:2] for token in sample["anns"]
            ])
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                 for anno in annotations],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = (velo @ np.linalg.inv(e2g_r_mat).T
                        @ np.linalg.inv(l2e_r_mat).T)
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes.tolist()
            info["gt_names"] = names.tolist()
            info["gt_velocity"] = velocity.reshape(-1, 2).tolist()
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]).tolist()
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]).tolist()
            info["valid_flag"] = valid_flag.tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])

    @property
    def num_available_scenes(self):
        return len(self.scenes)


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

    def get_token(self):
        return self.sample["sample_token"]

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
            coord_type="LIDAR",
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


class NuscenesLidarFromPoints(Dataset):
    def __init__(
            self,
            src_data_dir,
            version="v1.0-trainval",
            split_name="val",
            transforms=None,
            num_sweeps=9,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],  # noqa B006
            time_dim=4,
            pad_empty_sweeps=True,
            remove_close=True,
            use_valid_flag=False,
            with_velocity=True,
            classes=None,
            coord_type="LIDAR",
            test_mode=True,
    ):
        self.src_data_dir = src_data_dir
        self.transforms = transforms
        self.dataset = NuscenesParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
        )
        sample = self.dataset[0]
        sample = NuscenesSample(sample)

        self.num_sweeps = num_sweeps
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.time_dim = time_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity

        self.coord_type = coord_type
        self.test_mode = test_mode

        if not self.test_mode:
            self._set_group_flag()

        if use_dim is None:
            self.use_dim = [0, 1, 2, 4]
        assert (max(use_dim) < load_dim
                ), f"Expect all used dimensions < {load_dim}, got {use_dim}"

        assert (
            time_dim < load_dim
        ), f"Expect the timestamp dimension < {load_dim}, got {time_dim}"

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, item):
        sample = self.dataset[item]

        data = {}
        data["metas"] = {}
        data["metas"]["lidar2ego_translation"] = sample[
            "lidar2ego_translation"]
        data["metas"]["lidar2ego_rotation"] = sample["lidar2ego_rotation"]
        data["metas"]["ego2global_translation"] = sample[
            "ego2global_translation"]
        data["metas"]["ego2global_rotation"] = sample["ego2global_rotation"]

        sample = NuscenesSample(sample)

        data["sample_idx"] = sample.get_token()
        data["timestamp"] = sample.get_timestamp()
        data["points"] = sample.get_lidar_points(
            self.num_sweeps,
            self.load_dim,
            self.use_dim,
            self.time_dim,
            self.pad_empty_sweeps,
            self.remove_close,
            self.coord_type,
            self.test_mode,
        )

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __getitem__(self, idx):
        return self.prepare_data(idx)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-path", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--save-path",
    type=str,
    help="output path",
    default="./nuscenes_lidar_val")

args = parser.parse_args()

save_path = args.save_path
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

nuscenes_dataset = NuscenesLidarFromPoints(
    args.data_path, version="v1.0-trainval")

infos = {}
for i in tqdm(range(len(nuscenes_dataset))):
    data = nuscenes_dataset.prepare_data(i)
    points = np.array(data["points"], dtype=np.float32)
    file_name = data["sample_idx"] + ".bin"

    path = os.path.join(save_path, file_name)
    with open(path, "w") as sf:
        points.tofile(sf)

    bin_points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    diff = (points - bin_points).sum()
    if diff > 1e-5:
        print("*" * 20, file_name)
    if points.shape[0] > 300000:
        print("*" * 20, file_name)
        print(points.shape)

    infos[file_name] = {"metas": data["metas"]}

assert len(os.listdir(save_path)) == len(nuscenes_dataset)
assert len(list(infos.keys())) == len(nuscenes_dataset)

gt_anno_file = os.path.join(args.save_path, "val_gt_infos.pkl")

with open(gt_anno_file, "wb") as handle:
    pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)

assert os.path.exists(gt_anno_file)
print("======= Finish =======")
