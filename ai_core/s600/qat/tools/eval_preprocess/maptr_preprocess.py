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
import json
import multiprocessing
from typing import Mapping, Optional, Sequence, Dict, Tuple, List

import cv2
import numpy as np
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
    import geopandas as gpd
except ImportError:
    NuScenesCanBus = None
    Quaternion = None
    quaternion_yaw = None
    NuScenesMap = None
    NuScenesMapExplorer = None
    NuScenes = None
    splits = None
    gpd = None

from map_utils.geo_opensfm import TopocentricConverter

PIL_INTERP_CODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
}

supported_models = [
    'maptroe_henet_tinym_bevformer_nuscenes',
]


class VectorizedLocalMap(object):
    """VectorizedLocalMap.

    Args:
        dataroot: Root directory of the dataset.
        patch_size: Size of the patch.
        map_classes: Tuple of map classes. Default is ("divider",
        "ped_crossing", "boundary").
        line_classes: Tuple of line classes. Default is ("road_divider",
            "lane_divider").
        ped_crossing_classes: Tuple of pedestrian crossing classes. Default is
            ("ped_crossing").
        contour_classes: Tuple of contour classes. Default is ("road_segment",
            "lane").
        sample_dist: Distance between samples. Default is 1.
        num_samples: Number of samples. Default is 250.
        padding: Whether to pad the samples. Default is False.
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        thickness: Thickness of the lines. Default is 3.
        canvas_size: Size of the canvas. Default is (200, 200).
        aux_seg: Auxiliary segmentation information. Default is None.
    """

    CLASS2LABEL = {
        "road_divider": 0,
        "lane_divider": 0,
        "ped_crossing": 1,
        "contours": 2,
        "others": -1,
    }

    def __init__(
            self,
            dataroot: str,
            patch_size: int,
            map_classes: tuple = ("divider", "ped_crossing", "boundary"),
            line_classes: tuple = ("road_divider", "lane_divider"),
            ped_crossing_classes: tuple = ("ped_crossing", ),
            contour_classes: tuple = ("road_segment", "lane"),
            sample_dist: int = 1,
            num_samples: int = 250,
            padding: bool = False,
            fixed_ptsnum_per_line: int = -1,
            padding_value: int = -10000,
            thickness: int = 3,
            canvas_size: tuple = (200, 200),
            aux_seg: any = None,
            sd_map_path: Optional[str] = None,
            osm_thickness: int = 5,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        ]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

        # for semantic mask
        self.canvas_size = canvas_size
        self.thickness = thickness
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]
        self.aux_seg = aux_seg

        # sdmap
        self.osm_thickness = osm_thickness
        self.sd_maps = None
        if sd_map_path is not None:
            self.sd_maps = {}
            options = [
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",  # road
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary_link"
                "living_street",  # road link
                "road",  # Special road  'service'
            ]
            map_origin = {
                "boston-seaport": (
                    42.336849169438615,
                    -71.05785369873047,
                    0.0,
                ),
                "singapore-onenorth": (
                    1.2882100868743724,
                    103.78475189208984,
                    0.0,
                ),
                "singapore-hollandvillage": (
                    1.2993652317780957,
                    103.78217697143555,
                    0.0,
                ),
                "singapore-queenstown": (
                    1.2782562240223188,
                    103.76741409301758,
                    0.0,
                ),
            }

            for loc in self.MAPS:
                lat, lon, alt = map_origin[loc]
                sd_map = gpd.read_file(
                    os.path.join(sd_map_path, "{}.shp".format(loc)))
                converter = TopocentricConverter(lat, lon, alt)
                sd_map = sd_map[sd_map["type"].isin(options)]
                sd_map_topo_list = []
                for _, row in sd_map.iterrows():
                    tmp_sd_data = list(row.geometry.coords)
                    tmp_sd_data_topo = [
                        converter.to_topocentric(lonlat[1], lonlat[0], 0.0)[:2]
                        for lonlat in tmp_sd_data
                    ]
                    sd_map_topo_list.append(tmp_sd_data_topo)
                self.sd_maps[loc] = MultiLineString(sd_map_topo_list)

    def gen_vectorized_samples(
            self,
            location,
            homo_translation,
            homo_rotation,
    ):
        """Use coords2global to get gt map layers."""

        map_pose = homo_translation[:2]
        rotation = Quaternion(homo_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == "divider":
                line_geom = self.get_map_geom(patch_box, patch_angle,
                                              self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((
                            LineString(np.array(instance.coords)),
                            self.CLASS2LABEL.get(line_type, -1),
                        ))
            elif vec_class == "ped_crossing":
                ped_geom = self.get_map_geom(patch_box, patch_angle,
                                             self.ped_crossing_classes,
                                             location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((
                        LineString(np.array(instance.coords)),
                        self.CLASS2LABEL.get("ped_crossing", -1),
                    ))
            elif vec_class == "boundary":
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    vectors.append((
                        LineString(np.array(instance.coords)),
                        self.CLASS2LABEL.get("contours", -1),
                    ))
            else:
                raise ValueError(f"WRONG vec_class: {vec_class}")

        gt_labels = []
        gt_instance = []

        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)

        # sdmap
        if self.sd_maps is not None:
            osm_geom = self.get_osm_geom(patch_box, patch_angle, location)
            osm_vector_list = []

            for line in osm_geom:
                if not line.is_empty:
                    if line.geom_type == "MultiLineString":
                        for single_line in line.geoms:
                            pts, pts_num = self.sample_fixed_pts_from_line(
                                single_line, padding=False, fixed_num=50)
                            if pts_num >= 2:
                                osm_vector_list.append(LineString(pts))
                    elif line.geom_type == "LineString":
                        pts, pts_num = self.sample_fixed_pts_from_line(
                            line, padding=False, fixed_num=50)
                        if pts_num >= 2:
                            osm_vector_list.append(LineString(pts))
                    else:
                        raise NotImplementedError

            local_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
            osm_mask = self.line_osms_to_mask(
                osm_vector_list,
                local_box,
                self.canvas_size,
                thickness=self.osm_thickness,
            )
            osm_vectors = osm_vector_list

        anns_results = {
            "gt_vecs_pts_loc": gt_instance,
            "gt_vecs_label": gt_labels,
            "osm_vectors": osm_vectors if self.sd_maps is not None else None,
            "osm_mask": osm_mask if self.sd_maps is not None else None,
        }
        return anns_results

    def line_osms_to_mask(
            self,
            lines: List[LineString],
            local_box: Sequence[float],
            canvas_size: Sequence[int],
            color: int = 1,
            thickness: int = 5,
    ):
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = NuScenesMapExplorer.get_patch_coord(local_box)
        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        osm_mask = np.zeros(canvas_size, np.uint8)
        for line in lines:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(
                    new_line,
                    xfact=scale_width,
                    yfact=scale_height,
                    origin=(0, 0),
                )

                coords = np.array(list(new_line.coords), dtype=np.int32)
                coords = coords.reshape((-1, 2))
                assert len(coords) >= 2

                cv2.polylines(
                    osm_mask,
                    np.int32([coords]),
                    False,
                    color=color,
                    thickness=thickness,
                )
        osm_mask = osm_mask[np.newaxis, :, :]
        osm_mask = osm_mask.astype(np.int32)
        osm_mask[np.where(osm_mask == 0)] = -1
        osm_mask = osm_mask.astype(np.float32)
        return osm_mask

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle,
                                              layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle,
                                              layer_name, location)
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
                    for single_line in line.geoms:
                        line_vectors.append(
                            self.sample_pts_from_line(single_line))
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == "LineString":
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

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

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
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

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
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

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

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
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if (layer_name not in self.map_explorer[location].map_api.
                non_geometric_polygon_layers):
            raise ValueError("{} is not a polygonal layer".format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == "drivable_area":
            for record in records:
                polygons = [
                    self.map_explorer[location].map_api.extract_polygon(
                        polygon_token)
                    for polygon_token in record["polygon_tokens"]
                ]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon,
                            -patch_angle,
                            origin=(patch_x, patch_y),
                            use_radians=False,
                        )
                        new_polygon = affinity.affine_transform(
                            new_polygon,
                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y],
                        )
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(
                    record["polygon_token"])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon,
                            -patch_angle,
                            origin=(patch_x, patch_y),
                            use_radians=False,
                        )
                        new_polygon = affinity.affine_transform(
                            new_polygon,
                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y],
                        )
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if (layer_name not in self.map_explorer[location].map_api.
                non_geometric_line_layers):
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == "traffic_light":
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(
                record["line_token"])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)
        polygon_list = []
        records = self.map_explorer[location].map_api.ped_crossing
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(
                record["polygon_token"])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(
                        new_polygon,
                        -patch_angle,
                        origin=(patch_x, patch_y),
                        use_radians=False,
                    )
                    new_polygon = affinity.affine_transform(
                        new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == "Polygon":
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length/self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
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

        return sampled_points, num_valid

    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        """padding=True,间隔1m均匀采样,根据fixed_num不足进行补0,超过直接舍弃
        padding=False,根据fixed_num变步长进行采样,以满足长度要求"""
        if padding:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, fixed_num)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)

        num_valid = len(sampled_points)

        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        elif num_valid > fixed_num:
            sampled_points = sampled_points[:fixed_num, :]
            num_valid = fixed_num

        num_valid = len(sampled_points)
        return sampled_points, num_valid

    def get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for geom_line in osm_map.geoms:
            if geom_line.is_empty:
                continue
            new_line = geom_line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        return line_list


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

    def num_cams(self):
        return len(self.sample["cam"])


class NuscenesBevSampler(object):
    def __call__(self, sample):
        data = {}
        sample = NuscenesSample(sample)

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
        data["sample_token"] = sample.get_token()
        data["img_name"] = img_names
        data["img"] = imgs
        data["location"] = sample.get_location()

        data["ego2global"] = sample.get_ego2gloabl()
        data["meta"] = sample.get_meta()

        return data


class NuscenesParser(object):
    """Parser NuScenes.

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
        if NuScenes is None:
            raise ModuleNotFoundError(
                "nuscenes-devkit is needed while using NuScenes dataset.")

        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True)

        self.max_sweeps = max_sweeps
        self.nusc_can_bus = NuScenesCanBus(dataroot=src_data_dir)
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
        self._gen_cam_info(sample, sample_info)
        return sample_info

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

    def _gen_lidar_info(self, sample, info):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)

        cs_rec = self.nusc.get("calibrated_sensor",
                               sd_rec["calibrated_sensor_token"])
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        info["lidar2ego_translation"] = cs_rec["translation"]
        info["lidar2ego_rotation"] = cs_rec["rotation"]
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])

    @property
    def num_available_scenes(self):
        return len(self.scenes)


class NuscenesFromImageMap(Dataset):
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
            map_path: Optional[str] = None,
            map_classes: Optional[Tuple[str]] = None,
            pc_range: List[int] = None,
            bev_size: Tuple[int, int] = (200, 200),
            fixed_ptsnum_per_line: int = -1,
            padding_value: int = -10000,
            use_lidar_gt: bool = True,
            sd_map_path: Optional[str] = None,
    ):
        self.dataset = NuscenesParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
        )
        self.sampler = NuscenesBevSampler()
        self.transforms = transforms

        self.use_lidar_gt = use_lidar_gt
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=self.patch_size,
            map_classes=map_classes,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            canvas_size=bev_size,
            aux_seg=None,
            sd_map_path=sd_map_path,
        )

    def vectormap_pipeline(self, example):
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            example["meta"]["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = example["meta"]["lidar2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            example["meta"]["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = example["meta"]["ego2global_translation"]

        lidar2global = ego2global @ lidar2ego
        example["lidar2global"] = lidar2global

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)

        if not self.use_lidar_gt:
            translation = example["meta"]["ego2global_translation"]
            rotation = example["meta"]["ego2global_rotation"]

        location = example["meta"]["location"]

        anns_results = self.vector_map.gen_vectorized_samples(
            location,
            translation,
            rotation,
        )

        gt_vecs_label = anns_results["gt_vecs_label"]
        gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]

        example["gt_labels_map"] = gt_vecs_label
        example["gt_instances"] = gt_vecs_pts_loc

        if anns_results["osm_vectors"] is not None:
            example["osm_vectors"] = anns_results["osm_vectors"]
        if anns_results["osm_mask"] is not None:
            example["osm_mask"] = anns_results["osm_mask"]
        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.sampler(self.dataset[index])
        if self.transforms is not None:
            data = self.transforms(data)
        data = self.vectormap_pipeline(data)
        return data


def preprocess(img_list, image_data_path, file_name):
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


def process_reference_points(location, model, points_path, points_save_dir,
                             file_name):
    city = location.split("-")[0]
    input_path = os.path.join(points_path, model, city)
    for idx in range(4):
        reference_points = np.load(input_path + "/reference_points" +
                                   str(idx) + ".npy")
        new_path = points_save_dir + str(idx)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        fn = os.path.join(new_path, file_name)
        reference_points.tofile(fn)


def process_sdmap_masks(sdmap_masks, file_name):
    print(sdmap_masks.shape)
    sdmap_masks = sdmap_masks[np.newaxis, :, :, :]
    with open(file_name, "w") as sf:
        sdmap_masks.tofile(sf)


def target(data, image_data_path, sdmap_masks_data_path, points_save_dir,
           reference_path, model):
    file_name = data["sample_token"] + ".bin"
    preprocess(data["img"], image_data_path, file_name)

    sdmap_masks_save_name = os.path.join(sdmap_masks_data_path, file_name)
    process_sdmap_masks(data["osm_mask"], sdmap_masks_save_name)

    process_reference_points(data["location"], model, reference_path,
                             points_save_dir, file_name)


def parse_args():
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
        "--sdmap-path", type=str, help="path to sdmap", required=True)
    parser.add_argument(
        "--reference-path",
        type=str,
        help="path to reference points",
        required=True)
    parser.add_argument(
        "--save-path", type=str, help="output path", default="./nuscenes_bev")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    save_path = args.save_path
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    resize_size = (450, 800)
    input_size = (480, 800)
    padding = (0, 0, 0, 30)
    map_classes = ["divider", "ped_crossing", "boundary"]
    use_lidar_gt = False
    if use_lidar_gt:
        point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
        bev_h_ = 100
        bev_w_ = 50
        post_center_range = [-20, -35, -20, -35, 20, 35, 20, 35]
    else:
        point_cloud_range = [-30.0, -15.0, -10.0, 30.0, 15.0, 10.0]
        bev_h_ = 50
        bev_w_ = 100
        post_center_range = [-35, -20, -35, -20, 35, 20, 35, 20]

    transforms = torchvision.transforms.Compose([
        MultiViewsImgResize(size=resize_size),
        MultiViewsImgPad(padding=padding),
    ])

    nuscenes_dataset = NuscenesFromImageMap(
        version="v1.0-trainval",
        src_data_dir=args.data_path,
        split_name="val",
        map_path=args.meta_path,
        transforms=transforms,
        sd_map_path=args.sdmap_path,
        map_classes=map_classes,
        pc_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=20,
        padding_value=-10000,
        use_lidar_gt=use_lidar_gt,
    )

    image_data_path = os.path.join(save_path, "images")
    sdmap_masks_data_path = os.path.join(save_path, "sdmap_masks")
    os.makedirs(sdmap_masks_data_path)
    reference_points_path = os.path.join(save_path, "reference_points")

    pool = multiprocessing.Pool(processes=4)
    print(len(nuscenes_dataset))
    for i in tqdm(range(len(nuscenes_dataset))):
        data = nuscenes_dataset[i]
        pool.apply_async(target, (
            data,
            image_data_path,
            sdmap_masks_data_path,
            reference_points_path,
            args.reference_path,
            args.model,
        ))
    pool.close()
    pool.join()

    gt_annos = []
    for i in tqdm(range(len(nuscenes_dataset))):
        data = nuscenes_dataset[i]
        gt_anno = {}

        gt_vecs = data["gt_instances"]
        gt_labels = data["gt_labels_map"]
        gt_anno["sample_token"] = data["sample_token"]

        gt_vec_list = []
        for _, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
            name = map_classes[gt_label]
            anno = {
                "pts": list(gt_vec.coords),
                "pts_num": len(list(gt_vec.coords)),
                "cls_name": name,
                "type": gt_label,
            }
            gt_vec_list.append(anno)
        gt_anno["vectors"] = gt_vec_list
        gt_annos.append(gt_anno)

    gt_annos.sort(key=lambda x: x["sample_token"])
    map_ann_file = os.path.join(args.save_path, "nuscenes_map_anns_val.json")

    nusc_submissions = {'GTs': gt_annos}
    print('\n GT anns writes to', map_ann_file)
    with open(map_ann_file, "w") as f:
        json.dump(nusc_submissions, f, indent=2)

    assert os.path.exists(map_ann_file)

    print('======= Finish =======')
