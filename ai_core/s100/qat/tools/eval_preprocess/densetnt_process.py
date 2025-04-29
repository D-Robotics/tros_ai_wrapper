# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from tqdm import tqdm
import argparse
import json
import sys
import logging
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import shutil
import msgpack
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from argoverse_util.data_packer import Packer

import os
from pathlib import Path
from typing import Union

from argoverse_util.data_packer import get_packtype_from_path

logger = logging.getLogger(__name__)

MIAMI_ID = 10316
PITTSBURGH_ID = 10314

_PathLike = Union[str, "os.PathLike[str]"]


def read_json_file(fpath: Union[str, "os.PathLike[str]"]) -> Any:
    """Load dictionary from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        Deserialized Python dictionary.
    """
    with open(fpath, "rb") as f:
        return json.load(f)


def assert_np_array_shape(array: np.ndarray,
                          target_shape: Sequence[Optional[int]]) -> None:
    """Check for shape correctness.

    Args:
        array: array to check dimensions of.
        target_shape: desired shape. use None for unknown dimension sizes.

    Raises:
        ValueError: if array's shape does not match target_shape for any of the specified dimensions. # noqa
    """
    for index_dim, (array_shape_dim, target_shape_dim) in enumerate(
            zip(array.shape, target_shape)):
        if target_shape_dim and array_shape_dim != target_shape_dim:
            raise ValueError(
                f"array.shape[{index_dim}]: {array_shape_dim} != {target_shape_dim}."  # noqa
            )


class SE2:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize.

        Args:
            rotation: np.ndarray of shape (2,2).
            translation: np.ndarray of shape (2,1).

        Raises:
            ValueError: if rotation or translation do not have the required shapes. # noqa
        """
        assert_np_array_shape(rotation, (2, 2))
        assert_np_array_shape(translation, (2, ))
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(2) transformation to point_cloud.

        Args:
            point_cloud: np.ndarray of shape (N, 2).

        Returns:
            transformed_point_cloud: np.ndarray of shape (N, 2).

        Raises:
            ValueError: if point_cloud does not have the required shape.
        """
        assert_np_array_shape(point_cloud, (None, 2))
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        """Return the inverse of the current SE2 transformation.

        For example, if the current object represents target_SE2_src, we will return instead src_SE2_target. # noqa

        Returns:
            inverse of this SE2 transformation.
        """
        return SE2(
            rotation=self.rotation.T,
            translation=self.rotation.T.dot(-self.translation),
        )

    def inverse_transform_point_cloud(self,
                                      point_cloud: np.ndarray) -> np.ndarray:
        """Transform the point_cloud by the inverse of this SE2.

        Args:
            point_cloud: Numpy array of shape (N,2).

        Returns:
            point_cloud transformed by the inverse of this SE2.
        """
        return self.inverse().transform_point_cloud(point_cloud)

    def right_multiply_with_se2(self, right_se2: "SE2") -> "SE2":
        """Multiply this SE2 from right by right_se2 and \
return the composed transformation.

        Args:
            right_se2: SE2 object to multiply this object by from right.

        Returns:
            The composed transformation.
        """
        chained_transform_matrix = self.transform_matrix.dot(
            right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2


class Node:
    """Node.

    Args:
        id: representing unique node ID
        x: x-coordinate in city reference system
        y: y-coordinate in city reference system

    Returns:
        None
    """

    def __init__(self,
                 id: int,
                 x: float,
                 y: float,
                 height: Optional[float] = None):
        self.id = id
        self.x = x
        self.y = y
        self.height = height


class LaneSegment:
    def __init__(
            self,
            id: int,
            has_traffic_control: bool,
            turn_direction: str,
            is_intersection: bool,
            l_neighbor_id: Optional[int],
            r_neighbor_id: Optional[int],
            predecessors: List[int],
            successors: Optional[List[int]],
            centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way".
            has_traffic_control:
                turn_direction: 'RIGHT', 'LEFT', or 'NONE'.
            is_intersection: Whether or not this lane segment is an intersection.  # noqa
            l_neighbor_id: Unique ID for left neighbor.
            r_neighbor_id: Unique ID for right neighbor.
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline


def find_all_polygon_bboxes_overlapping_query_bbox(
        polygon_bboxes: np.ndarray, query_bbox: np.ndarray) -> np.ndarray:
    """Find all the overlapping polygon bounding boxes.

    Each bounding box has the following structure:
        bbox = np.array([x_min,y_min,x_max,y_max])

    In 3D space, if the coordinates are equal (polygon bboxes touch),
    then these are considered overlapping. We have a guarantee that
    the cropped image will have any sort of overlap with the zero'th
    object bounding box. inside of the image e.g. along the x-dimension,
    either the left or right side of the bounding box lies between
    the edges of the query bounding box, or the bounding box completely
    engulfs the query bounding box.

    Args:
        polygon_bboxes: An array of shape (K,), each array element is
                        a NumPy array of shape (4,) representing
                        the bounding box for a polygon or point cloud.
        query_bbox: An array of shape (4,) representing a 2d axis-aligned
                    bounding box, with order [min_x,min_y,max_x,max_y].

    Returns:
        An integer array of shape (K,) representing indices where overlap occurs. # noqa
    """
    query_min_x = query_bbox[0]
    query_min_y = query_bbox[1]

    query_max_x = query_bbox[2]
    query_max_y = query_bbox[3]

    bboxes_x1 = polygon_bboxes[:, 0]
    bboxes_x2 = polygon_bboxes[:, 2]

    bboxes_y1 = polygon_bboxes[:, 1]
    bboxes_y2 = polygon_bboxes[:, 3]

    # check if falls within range
    overlaps_left = (query_min_x <= bboxes_x2) & (bboxes_x2 <= query_max_x)
    overlaps_right = (query_min_x <= bboxes_x1) & (bboxes_x1 <= query_max_x)

    x_check1 = bboxes_x1 <= query_min_x
    x_check2 = query_min_x <= query_max_x
    x_check3 = query_max_x <= bboxes_x2
    x_subsumed = x_check1 & x_check2 & x_check3

    x_in_range = overlaps_left | overlaps_right | x_subsumed

    overlaps_below = (query_min_y <= bboxes_y2) & (bboxes_y2 <= query_max_y)
    overlaps_above = (query_min_y <= bboxes_y1) & (bboxes_y1 <= query_max_y)

    y_check1 = bboxes_y1 <= query_min_y
    y_check2 = query_min_y <= query_max_y
    y_check3 = query_max_y <= bboxes_y2
    y_subsumed = y_check1 & y_check2 & y_check3
    y_in_range = overlaps_below | overlaps_above | y_subsumed
    overlap_indxs = np.where(x_in_range & y_in_range)[0]
    return overlap_indxs


def extract_node_from_ET_element(child: ET.Element) -> Node:
    """
    Extract node from ET element.

    Given a line of XML, build a node object.
    The "node_fields" dictionary will hold "id", "x", "y".
    The XML will resemble:
        <node id="0" x="3168.066310258233" y="1674.663991981186" />

    Args:
        child: xml.etree.ElementTree element

    Returns:
        Node object
    """
    node_fields = child.attrib
    node_id = int(node_fields["id"])
    if "height" in node_fields.keys():
        return Node(
            id=node_id,
            x=float(node_fields["x"]),
            y=float(node_fields["y"]),
            height=float(node_fields["height"]),
        )
    return Node(
        id=node_id, x=float(node_fields["x"]), y=float(node_fields["y"]))


def get_lane_identifier(child: ET.Element) -> int:
    """
    Fetch lane ID from XML ET.Element.

    Args:
       child: ET.Element with information about Way

    Returns:
       unique lane ID
    """
    return int(child.attrib["lane_id"])


def append_additional_key_value_pair(lane_obj: MutableMapping[str, Any],
                                     way_field: List[Tuple[str, str]]) -> None:
    """
    Append additional key value pair.

    Key name was either 'predecessor' or 'successor',
    for which we can have multiple. Thus we append them to a list.
    They should be integers, as lane IDs.

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = int(way_field[1][1])
    lane_obj.setdefault(k, []).append(v)


def append_unique_key_value_pair(lane_obj: MutableMapping[str, Any],
                                 way_field: List[Tuple[str, str]]) -> None:
    """
    Append unique key value pair.

    For the following types of Way "tags", the key,
    value pair is defined only once within the object:
        - has_traffic_control
        - turn_direction
        - is_intersection
        - l_neighbor_id
        - r_neighbor_id

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = way_field[1][1]
    lane_obj[k] = v


def extract_node_waypt(way_field: List[Tuple[str, str]]) -> int:
    """
    Extract_node_waypt.

    Given a list with a reference node such as [('ref', '0')],
    extract out the lane ID.

    Args:
       way_field: key and node id pair to extract

    Returns:
       node_id: unique ID for a node waypoint
    """
    key = way_field[0][0]
    node_id = way_field[0][1]
    assert key == "ref"
    return int(node_id)


def convert_node_id_list_to_xy(
        node_id_list: List[int],
        all_graph_nodes: Mapping[int, Node]) -> np.ndarray:
    """
    Convert node id list to centerline xy coordinate.

    Args:
       node_id_list: list of node_id's
       all_graph_nodes: dictionary mapping node_ids to Node

    Returns:
       centerline
    """
    num_nodes = len(node_id_list)

    if all_graph_nodes[node_id_list[0]].height is not None:
        centerline = np.zeros((num_nodes, 3))
    else:
        centerline = np.zeros((num_nodes, 2))
    for i, node_id in enumerate(node_id_list):
        if all_graph_nodes[node_id].height is not None:
            centerline[i] = np.array([
                all_graph_nodes[node_id].x,
                all_graph_nodes[node_id].y,
                all_graph_nodes[node_id].height,
            ])
        else:
            centerline[i] = np.array(
                [all_graph_nodes[node_id].x, all_graph_nodes[node_id].y])

    return centerline


def str_to_bool(s: str) -> bool:
    """
    Convert str to bool.

    Args:
       string representation of boolean, either 'True' or 'False'.

    Returns:
       boolean
    """
    if s == "True":
        return True
    assert s == "False"
    return False


def convert_dictionary_to_lane_segment_obj(
        lane_id: int, lane_dictionary: Mapping[str, Any]) -> LaneSegment:
    """
    Not all lanes have predecessors and successors.

    Args:
       lane_id: representing unique lane ID
       lane_dictionary: dictionary with LaneSegment attributes,
                        not yet in object instance form

    Returns:
       ls: LaneSegment object
    """
    predecessors = lane_dictionary.get("predecessor", None)
    successors = lane_dictionary.get("successor", None)
    has_traffic_control = str_to_bool(lane_dictionary["has_traffic_control"])
    is_intersection = str_to_bool(lane_dictionary["is_intersection"])
    lnid = lane_dictionary["l_neighbor_id"]
    rnid = lane_dictionary["r_neighbor_id"]
    l_neighbor_id = None if lnid == "None" else int(lnid)
    r_neighbor_id = None if rnid == "None" else int(rnid)
    ls = LaneSegment(
        lane_id,
        has_traffic_control,
        lane_dictionary["turn_direction"],
        is_intersection,
        l_neighbor_id,
        r_neighbor_id,
        predecessors,
        successors,
        lane_dictionary["centerline"],
    )
    return ls


def extract_lane_segment_from_ET_element(
        child: ET.Element,
        all_graph_nodes: Mapping[int, Node]) -> Tuple[LaneSegment, int]:
    """
    Eextract lane_segment from ET element.

    We build a lane segment from an XML element.
    A lane segment is equivalent to a "Way" in our XML file.
    Each Lane Segment has a polyline representing its centerline.

    The relevant XML data might resemble::

        <way lane_id="9604854">
            <tag k="has_traffic_control" v="False" />
            <tag k="turn_direction" v="NONE" />
            <tag k="is_intersection" v="False" />
            <tag k="l_neighbor_id" v="None" />
            <tag k="r_neighbor_id" v="None" />
            <nd ref="0" />
            ...
            <nd ref="9" />
            <tag k="predecessor" v="9608794" />
            ...
            <tag k="predecessor" v="9609147" />
        </way>

    Args:
        child: xml.etree.ElementTree element
        all_graph_nodes

    Returns:
        lane_segment: LaneSegment object
        lane_id
    """
    lane_obj: Dict[str, Any] = {}
    lane_id = get_lane_identifier(child)
    node_id_list: List[int] = []
    for element in child:
        way_field = cast(List[Tuple[str, str]], list(element.items()))
        field_name = way_field[0][0]
        if field_name == "k":
            key = way_field[0][1]
            if key in {"predecessor", "successor"}:
                append_additional_key_value_pair(lane_obj, way_field)
            else:
                append_unique_key_value_pair(lane_obj, way_field)
        else:
            node_id_list.append(extract_node_waypt(way_field))

    lane_obj["centerline"] = convert_node_id_list_to_xy(
        node_id_list, all_graph_nodes)
    lane_segment = convert_dictionary_to_lane_segment_obj(lane_id, lane_obj)
    return lane_segment, lane_id


def load_lane_segments_from_xml(map_fpath: _PathLike,
                                ) -> Mapping[int, LaneSegment]:
    """
    Load lane segment object from xml file.

    Args:
       map_fpath: path to xml file

    Returns:
       lane_objs: List of LaneSegment objects
    """
    tree = ET.parse(os.fspath(map_fpath))
    root = tree.getroot()

    logger.info(f"Loaded root: {root.tag}")

    all_graph_nodes = {}
    lane_objs = {}
    # all children are either Nodes or Ways
    for child in root:
        if child.tag == "node":
            node_obj = extract_node_from_ET_element(child)
            all_graph_nodes[node_obj.id] = node_obj
        elif child.tag == "way":
            lane_obj, lane_id = extract_lane_segment_from_ET_element(
                child, all_graph_nodes)
            lane_objs[lane_id] = lane_obj
        else:
            logger.error("Unknown XML item encountered.")
            raise ValueError("Unknown XML item encountered.")
    return lane_objs


class ArgoverseMap(object):
    """
    Argoverse map parser.

    Args:
        root: The path of the parent directory of map data.
    """

    def __init__(self, root):
        self.city_name_to_city_id_dict = {
            "PIT": PITTSBURGH_ID,
            "MIA": MIAMI_ID,
        }
        self.root = root

        self.city_lane_centerlines_dict = self.build_centerline_index()
        self.city_rasterized_ground_height_dict = (
            self.build_city_ground_height_index())
        (
            self.city_halluc_bbox_table,
            self.city_halluc_tableidx_to_laneid_map,
        ) = self.build_hallucinated_lane_bbox_index()

    @property
    def map_files_root(self) -> Path:
        if self.root is None:
            raise ValueError("Map root directory cannot be None!")
        return Path(self.root).resolve()

    def build_city_ground_height_index(
            self, ) -> Mapping[str, Mapping[str, np.ndarray]]:
        """
        Build index of rasterized ground height.

        Returns:
            city_ground_height_index: a dictionary of dictionaries.
                                      Key is city_name, and values
                                      are dictionaries that store
                                      the "ground_height_matrix" and also the
            city_to_pkl_image_se2: SE(2) that produces takes point
                                   in pkl image to city coordinates,
                                   e.g. p_city = city_Transformation_pklimage
                                                 * p_pklimage
        """
        city_rasterized_ground_height_dict: Dict[str, Dict[str, np.
                                                           ndarray]] = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            city_rasterized_ground_height_dict[city_name] = {}
            npy_fpath = (
                self.map_files_root /
                f"{city_name}_{city_id}_ground_height_mat_2019_05_28.npy")

            # load the file with rasterized values
            city_rasterized_ground_height_dict[city_name][
                "ground_height"] = np.load(npy_fpath)

            se2_npy_fpath = (
                self.map_files_root /
                f"{city_name}_{city_id}_npyimage_to_city_se2_2019_05_28.npy")
            city_rasterized_ground_height_dict[city_name][
                "npyimage_to_city_se2"] = np.load(se2_npy_fpath)

        return city_rasterized_ground_height_dict

    def build_centerline_index(self,
                               ) -> Mapping[str, Mapping[int, LaneSegment]]:
        """
        Build dictionary of centerline for each city, with lane_id as key.

        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are
                                         dictionaries
                                         (k=lane_id, v=lane info)
        """
        city_lane_centerlines_dict = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            xml_fpath = (
                self.map_files_root /
                f"pruned_argoverse_{city_name}_{city_id}_vector_map.xml")
            city_lane_centerlines_dict[
                city_name] = load_lane_segments_from_xml(xml_fpath)
        return city_lane_centerlines_dict

    def append_height_to_2d_city_pt_cloud(self, pt_cloud_xy: np.ndarray,
                                          city_name: str) -> np.ndarray:
        """Accept 2d point cloud in xy plane and return 3d point cloud (xyz).

        Args:
            pt_cloud_xy: Numpy array of shape (N,2)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            pt_cloud_xyz: Numpy array of shape (N,3)
        """
        pts_z = self.get_ground_height_at_xy(pt_cloud_xy, city_name)
        return np.hstack([pt_cloud_xy, pts_z[:, np.newaxis]])

    def get_rasterized_ground_height(
            self, city_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get rasterized ground height.

        Get ground height matrix along with se2
        that convert to city coordinate.
        Args:
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            ground_height_matrix
            city_to_pkl_image_se2: SE(2) that produces takes point in pkl image
                                   to city coordinates, e.g. p_city =
                                   city_Transformation_pklimage * p_pklimage
        """
        ground_height_mat = self.city_rasterized_ground_height_dict[city_name][
            "ground_height"]
        return (
            ground_height_mat,
            self.city_rasterized_ground_height_dict[city_name]
            ["npyimage_to_city_se2"],
        )

    def get_ground_height_at_xy(self, point_cloud: np.ndarray,
                                city_name: str) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.

        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3)
            city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

        Returns:
            ground_height_values: Numpy array of shape (k,)
        """
        (
            ground_height_mat,
            npyimage_to_city_se2_mat,
        ) = self.get_rasterized_ground_height(city_name)
        city_coords = np.round(point_cloud[:, :2]).astype(np.int64)

        se2_rotation = npyimage_to_city_se2_mat[:2, :2]
        se2_trans = npyimage_to_city_se2_mat[:2, 2]

        npyimage_to_city_se2 = SE2(
            rotation=se2_rotation, translation=se2_trans)
        npyimage_coords = npyimage_to_city_se2.transform_point_cloud(
            city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        ground_height_values = np.full((npyimage_coords.shape[0]), np.nan)
        ind_valid_pts = (
            npyimage_coords[:, 1] < ground_height_mat.shape[0]) * (
                npyimage_coords[:, 0] < ground_height_mat.shape[1])

        ground_height_values[ind_valid_pts] = ground_height_mat[
            npyimage_coords[ind_valid_pts, 1],
            npyimage_coords[ind_valid_pts, 0], ]

        return ground_height_values

    def get_lane_segment_centerline(self, lane_segment_id: int,
                                    city_name: str) -> np.ndarray:
        """
        We return a 3D centerline for any particular lane segment.

        Args:
            lane_segment_id: unique identifier for a lane segment within a city
            city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

        Returns:
            lane_centerline: Numpy array of shape (N,3)
        """
        lane_centerline = self.city_lane_centerlines_dict[city_name][
            lane_segment_id].centerline
        if len(lane_centerline[0]) == 2:
            lane_centerline = self.append_height_to_2d_city_pt_cloud(
                lane_centerline, city_name)

        return lane_centerline

    def build_hallucinated_lane_bbox_index(
            self, ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Build_hallucinated_lane_bbox_index.

        Populate the pre-computed hallucinated extent of each lane polygon,
        to allow for fast queries.

        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = {}
        city_halluc_tableidx_to_laneid_map = {}

        for city_name, city_id in self.city_name_to_city_id_dict.items():
            json_fpath = (self.map_files_root /
                          f"{city_name}_{city_id}_tableidx_to_laneid_map.json")
            city_halluc_tableidx_to_laneid_map[city_name] = read_json_file(
                json_fpath)

            npy_fpath = (self.map_files_root /
                         f"{city_name}_{city_id}_halluc_bbox_table.npy")
            city_halluc_bbox_table[city_name] = np.load(npy_fpath)

        return city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map

    def get_lane_ids_in_xy_bbox(
            self,
            query_x: float,
            query_y: float,
            city_name: str,
            query_search_range_manhattan: float = 5.0,
    ) -> List[int]:
        query_min_x = query_x - query_search_range_manhattan
        query_max_x = query_x + query_search_range_manhattan
        query_min_y = query_y - query_search_range_manhattan
        query_max_y = query_y + query_search_range_manhattan

        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            self.city_halluc_bbox_table[city_name],
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        if len(overlap_indxs) == 0:
            return []

        neighborhood_lane_ids: List[int] = []
        for overlap_idx in overlap_indxs:
            lane_segment_id = self.city_halluc_tableidx_to_laneid_map[
                city_name][str(overlap_idx)]
            neighborhood_lane_ids.append(lane_segment_id)

        return neighborhood_lane_ids


class Argoverse1Sampler(object):
    """
    Sampler for argoverse dataset.

    Args:
        map_path: The path of map data.
        pred_step: Steps for traj prediction.
        traj_scale: Scale for traj feat. Needed for qat.
        max_distance: Max distance for map range.
        max_lane_num: Max num of lane vector.
        max_lane_poly: Max num of lane poly.
        max_traj_num: Max num of traj num.
        max_goals_num: Max goals num .
        use_subdivide: Whether use subdivide for goals generation.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    def __init__(
            self,
            map_path: str,
            pred_step: int = 20,
            traj_scale: int = 50,
            max_distance: float = 50.0,
            max_lane_num: int = 64,
            max_lane_poly: int = 9,
            max_traj_num: int = 32,
            max_goals_num: int = 2048,
            use_subdivide: bool = True,
    ):
        self.map_loader = ArgoverseMap(map_path)

        self.pred_step = pred_step
        self.traj_scale = traj_scale
        self.max_distance = max_distance
        self.max_lane_num = max_lane_num
        self.max_traj_num = max_traj_num
        self.use_subdivide = use_subdivide
        self.max_goals_num = max_goals_num
        self.max_lane_poly = max_lane_poly

    def _get_subdivide_points(self, polygon, threshold=1.0):
        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0])**2 +
                           (point_a[1] - point_b[1])**2)

        average_dis = 0
        point_pre = 0
        for i, point in enumerate(polygon):
            if i > 0:
                average_dis += get_dis(point, point_pre)
            point_pre = point
        average_dis /= len(polygon) - 1

        points = []
        divide_num = 1
        while average_dis / divide_num > threshold:
            divide_num += 1
        for i, point in enumerate(polygon):
            if i > 0:
                for k in range(1, divide_num):

                    def get_kth_point(point_a, point_b, ratio):
                        return (
                            point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio,
                        )

                    points.append(
                        get_kth_point(point_pre, point, k / divide_num))
            point_pre = point
        return points

    def _gen_direction(self, agent_lines):
        span = agent_lines[-6:]
        interval = 2
        angles = []
        for i in range(len(span)):
            if i + interval < len(span):
                der = span[i + interval] - span[i]
                angles.append(der)
        angles = np.stack(angles)
        der = np.mean(angles, axis=0)
        angle = -math.atan2(der[1], der[0]) + math.radians(90)
        return angle

    def _rotate(self, x, y, angle):
        x = x * math.cos(angle) - y * math.sin(angle)
        y = x * math.sin(angle) + y * math.cos(angle)
        return np.stack([x, y], axis=-1)

    def _norm(self, feat, cent, angle):

        norm_feat = self._rotate(feat[..., 0] - cent[0],
                                 feat[..., 1] - cent[1], angle)

        return norm_feat

    def _gen_traj(self, sample, data):
        coordinate = np.array(sample["coordinate"])

        feat = coordinate[:, :self.pred_step]

        labels = coordinate[0, self.pred_step:]

        mask = np.array(sample["mask"])
        feat_mask = mask[:, :self.pred_step]
        label_mask = mask[0, self.pred_step:]

        data["feat_mask"] = feat_mask
        data["label_mask"] = label_mask

        cent = feat[0, -1]
        data["cent"] = cent

        angle = self._gen_direction(feat[0])
        data["angle"] = angle

        feat = self._norm(feat, cent, angle) / self.traj_scale
        feat = feat * feat_mask[:, :, None]

        coors_diff = np.diff(feat, axis=1)
        coors_yaw = np.arctan2(coors_diff[:, :, 1], coors_diff[:, :, 0])

        labels = self._norm(labels, cent, angle)
        labels = labels * label_mask[:, None]

        timestamps = np.array(sample["time"])
        start_time = timestamps[0]
        timestamps = timestamps[1:self.pred_step] - start_time

        timestamps = timestamps.reshape(1, len(timestamps), 1).repeat(
            len(feat), axis=0)
        cat = np.array(sample["category"]).astype(dtype=np.int8)
        cat = np.eye(3)[cat - 1]
        cat = cat.reshape(-1, 1, 3).repeat(self.pred_step - 1, axis=1)
        feat = np.concatenate([feat[:, :-1], feat[:, 1:]], axis=-1)
        feat = np.concatenate(
            [feat, np.expand_dims(coors_yaw, axis=-1), timestamps, cat],
            axis=-1,
        )
        traj_mask = np.ones((self.max_traj_num)) * -100

        if feat.shape[0] > self.max_traj_num:
            feat = feat[:self.max_traj_num]

        num_traj, num_step, num_feat = feat.shape

        traj_feat = np.zeros((self.max_traj_num, num_step, num_feat))

        traj_feat[:num_traj] = feat
        traj_mask[:num_traj] = 0

        traj_feat = traj_feat.transpose((2, 1, 0))
        data["traj_feat"] = traj_feat.astype(np.float32)
        data["traj_mask"] = traj_mask.astype(np.float32)
        data["traj_labels"] = labels.astype(np.float32)
        data["end_points"] = data["traj_labels"][-1, :]

    def _gen_sub_map(self, sample, data):
        end_points = data["traj_labels"][-1]
        cent = data["cent"]
        city_name = sample["city"]
        lane_ids = self.map_loader.get_lane_ids_in_xy_bbox(
            cent[0],
            cent[1],
            city_name,
            query_search_range_manhattan=self.max_distance,
        )
        polygons = []
        lane_segments = []
        for lane_id in lane_ids:
            local_lane_centerline = (
                self.map_loader.get_lane_segment_centerline(
                    lane_id, city_name))
            polygons.append(local_lane_centerline[:, :2].copy())
            lane_segments.append(
                self.map_loader.city_lane_centerlines_dict[city_name][lane_id])
        angle = data["angle"]

        center_points = []
        for polygon in polygons:
            new_polygon = []
            for _, point in enumerate(polygon):
                new_point = self._norm(point, cent, angle)
                new_polygon.append(new_point)
            center_points.append(np.array(new_polygon))

        POLYLINE_DIRECTION = {"NONE": 0, "LEFT": -1, "RIGHT": 1}

        pid_mapping = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

        feats = []
        lane_mask = np.ones((self.max_lane_num)) * -100

        if len(lane_ids) > self.max_lane_num:
            lane_ids = lane_ids[:self.max_lane_num]
        for i, lane_id in enumerate(lane_ids):
            lane_props = lane_segments[i]
            center_point = center_points[i]
            seq_len = len(center_point) - 1
            op_feat = []
            turn_direction = POLYLINE_DIRECTION[lane_props.turn_direction]
            op_feat.append(np.ones((seq_len, 1)) * turn_direction)

            op_feat.append(np.ones((seq_len, 1)) * pid_mapping[lane_id])
            if lane_props.successors is not None and not len(
                    lane_props.successors):
                pid_succ = lane_props.successors[0]
                if pid_succ not in pid_mapping:
                    pid_succ = -1
                else:
                    pid_succ = pid_mapping[pid_succ]
            else:
                pid_succ = -1

            if lane_props.predecessors is not None and not len(
                    lane_props.predecessors):
                pid_pred = lane_props.predecessors[0]
                if pid_pred not in pid_mapping:
                    pid_pred = -1
                else:
                    pid_pred = pid_mapping[pid_pred]
            else:
                pid_pred = -1

            pid_pred_succ = np.stack([pid_pred, pid_succ], axis=-1).reshape(
                1, 2)
            pid_pred_succ = pid_pred_succ.repeat(seq_len, axis=0)
            op_feat.append(pid_pred_succ)

            pre_pre_point = np.concatenate(
                (center_point[0:1, :], center_point[:-2, :]), axis=0)
            op_feat.append(pre_pre_point)

            op_feat.append(
                np.ones((seq_len, 1)) * int(lane_props.is_intersection))
            op_feat = np.concatenate(op_feat, axis=-1)

            line_feat = np.concatenate([center_point[:-1], center_point[1:]],
                                       axis=-1)
            lane_feat = np.concatenate([line_feat, op_feat], axis=-1)
            seq_len, channels = lane_feat.shape
            feat = np.zeros((self.max_lane_poly, channels))
            feat[:seq_len] = lane_feat
            feats.append(feat)
            lane_mask[i] = 0

        feats = np.stack(feats)
        lane_num, poly_num, channels = feats.shape
        lane_feats = np.zeros((self.max_lane_num, poly_num, channels))
        lane_feats[:lane_num] = feats
        lane_feats = lane_feats.transpose((2, 1, 0))
        data["lane_feat"] = lane_feats.astype(np.float32)
        data["lane_mask"] = lane_mask.astype(np.float32)

        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round(
                (point[1] + 500) * 100)

        goals = []
        visit = {}
        points = []
        for cps in center_points:
            points.extend(cps.tolist())
            if self.use_subdivide is True:
                subdivide_points = self._get_subdivide_points(cps)
                points.extend(subdivide_points)
        for p in points:
            key = get_hash(p)
            if key not in visit:
                visit[key] = True
                goals.append(p)

        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0])**2 +
                           (point_a[1] - point_b[1])**2)

        if len(goals) > self.max_goals_num:
            logger.info(
                f"number of goals {len(goals)} exceeds {self.max_goals_num}")
            goals = goals[:self.max_goals_num]
        goals_dis = np.array([get_dis(goal, end_points) for goal in goals])
        data["goals_2d_labels"] = np.argmin(goals_dis)

        goals_2d = np.zeros((self.max_goals_num, 2)).astype(np.float32)
        goals_2d[:len(goals)] = np.array(goals).astype(dtype=np.float32)
        data["goals_2d"] = np.array(goals_2d).transpose((1, 0))

        goals_2d_mask = np.ones((self.max_goals_num)) * -100
        goals_2d_mask[:len(goals)] = 0
        data["goals_2d_mask"] = goals_2d_mask.astype(np.float32)

    def __call__(self, sample):
        data = {}

        self._gen_traj(sample, data)
        self._gen_sub_map(sample, data)
        data["file_name"] = sample["file_name"]
        instance_mask = np.concatenate([data["traj_mask"], data["lane_mask"]])
        data["instance_mask"] = instance_mask.reshape((1, -1))
        return data


class Argoverse1Dataset(Dataset):
    """
    Argoverse  dataset v1.

    Args:
        data_path: The path of the parent directory of data.
        map_path: The path of map data.
        transforms: A function transform that takes input sample \
and its target as entry and returns a transformed version.
        pred_step: Steps for traj prediction.
        max_distance: Max distance for map range.
        max_lane_num: Max num of lane vector.
        max_lane_poly: Max num of lane poly.
        max_traj_num: Max num of traj num.
        max_goals_num: Max goals num .
        use_subdivide: Whether use subdivide for goals generation.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        transforms: Optional[Callable] = None,
    """

    CATEGORY_MAP = {"AGENT": 1, "AV": 2, "OTHERS": 3}

    def __init__(
            self,
            data_path: str,
            map_path: str,
            transforms: Optional[Callable] = None,
            pred_step: int = 20,
            max_distance: float = 50.0,
            max_lane_num: int = 64,
            max_lane_poly: int = 9,
            max_traj_num: int = 32,
            max_goals_num: int = 2048,
            use_subdivide: bool = True,
            pack_type: Optional[str] = None,
            pack_kwargs: Optional[dict] = None,
    ):

        self.transforms = transforms
        self.root = data_path

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.sampler = Argoverse1Sampler(
            map_path=map_path,
            pred_step=pred_step,
            max_distance=max_distance,
            max_lane_num=max_lane_num,
            max_lane_poly=max_lane_poly,
            max_traj_num=max_traj_num,
            max_goals_num=max_goals_num,
            use_subdivide=use_subdivide,
        )
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs)
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __getstate__(self):
        state = self.__dict__
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs)
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def _decode(self, pack_file, sample):
        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        sample = pack_file.read(sample)
        sample = msgpack.unpackb(sample, object_hook=_decode_hook, raw=True)
        return sample

    def __getitem__(self, index: int) -> dict:
        sample = self._decode(self.pack_file, self.samples[index])
        sample = self.sampler(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)


class Argoverse1Parser(object):
    """
    Parser for argoverse dataset from csv format.

    Args:
        data_path: The path of the parent directory of data.
        mode: The name of the dataset directory.
    """

    def __init__(self, data_path: str, mode: str):
        self.data_path = os.path.join(data_path, mode, "data")
        self.csv_files = os.listdir(self.data_path)
        self.track_num = 0

    def _parse_csv(self, fs):
        df = pd.read_csv(fs)
        city = df["CITY_NAME"][0]
        tracks = list(df.groupby("TRACK_ID"))
        object_num = len(tracks)

        time = np.sort(np.unique(df["TIMESTAMP"].values))
        step_num = len(time)
        time_step_dict = dict(zip(time.tolist(), list(range(step_num))))

        coordinate = np.zeros([object_num, step_num, 2])
        mask = np.zeros([object_num, step_num])
        category = np.zeros([object_num])

        idx_others = 2
        time = np.sort(np.unique(df["TIMESTAMP"].values))
        time_step_dict = dict(zip(time.tolist(), list(range(50))))
        for _track_id, data in tracks:
            step = np.array(
                [time_step_dict[x] for x in data["TIMESTAMP"].values],
                dtype=np.int32,
            )
            category_id = Argoverse1Dataset.CATEGORY_MAP[data["OBJECT_TYPE"].
                                                         values[0]]
            xy = data[["X", "Y"]].values

            if category_id == 1:
                coordinate[0, step] = xy
                mask[0, step] = 1
                category[0] = category_id
            elif category_id == 2:
                coordinate[1, step] = xy
                mask[1, step] = 1
                category[1] = category_id
            else:
                coordinate[idx_others, step] = xy
                mask[idx_others, step] = 1
                category[idx_others] = category_id
                idx_others += 1
        if len(tracks) > self.track_num:
            self.track_num = len(tracks)
        sample = {
            "file_name": fs,
            "coordinate": coordinate.tolist(),
            "mask": mask.tolist(),
            "category": category.tolist(),
            "city": city,
            "time": time.tolist(),
        }
        return sample

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, index: int):
        csv_file = self.csv_files[index]
        fs = os.path.join(self.data_path, csv_file)
        sample = self._parse_csv(fs)
        return sample


class Argoverse1Packer(Packer):
    """
    Packer for converting argoverse dataset from csv format into lmdb format.

    Args:
        src_data_path: The path of the parent directory of data.
        mode: The name of the dataset directory.
        target_data_path: The target path to store lmdb dataset.
        num_workers: Num workers for reading original data.
            while num_workers <= 0 means pack by single process.
            num_workers >= 1 mean pack by num_workers process.
        pack_type: The file type for packing.
        **kwargs: Kwargs for Packer.
    """

    def __init__(
            self,
            src_data_path: str,
            mode: str,
            target_data_path: str,
            num_workers: int,
            pack_type: str,
            **kwargs,
    ):

        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        self.parser = Argoverse1Parser(src_data_path, mode)

        super(Argoverse1Packer, self).__init__(
            target_data_path,
            len(self.parser),
            pack_type,
            num_workers,
            **kwargs,
        )

    def pack_data(self, idx):
        sample = self.parser[idx]
        return msgpack.packb(sample)

    def _write(self, idx, data):
        self.pack_file.write(idx, data)


def parse_args():
    parser = argparse.ArgumentParser(description="Pack argoverse dataset.")
    parser.add_argument(
        "--src-path",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--dst-path",
        required=True,
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--is-gen-lmdb",
        default="true",
        help="Whether gen lmdb,if you have not gened it,should gen it.")
    parser.add_argument(
        "--is-gen-visual-config",
        default="false",
        help="Whether gen visualization config.")
    parser.add_argument(
        "--num-workers",
        default=10,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--num-batch",
        default=30,
        help="The number of model batch.",
    )
    args = parser.parse_args()
    return args


def save_data(file_data, file_name, save_name):
    with open(save_name, "w") as sf:
        data_float32 = file_data.astype(np.float32)
        data_float32.tofile(sf)


if __name__ == "__main__":
    args = parse_args()

    lmdb_tmp_path = args.src_path + "/val_lmdb"
    print("Process lmdb start!")
    if (args.is_gen_lmdb == "true"):
        if os.path.exists(lmdb_tmp_path):
            print(f"{lmdb_tmp_path} dir exits!")
            try:
                shutil.rmtree(lmdb_tmp_path)
                print(f"Directory '{lmdb_tmp_path}' successfully removed.")
                os.makedirs(lmdb_tmp_path, exist_ok=True)
            except OSError as e:
                print(f"Error while deleting directory '{lmdb_tmp_path}': {e}")

        directory = os.path.expanduser(args.src_path)
        print("Loading dataset from %s" % directory)

        pack_path = os.path.join(
            args.src_path,
            "%s_%s" % ("val", "lmdb"),
        )
        packer = Argoverse1Packer(
            directory,
            "val",
            pack_path,
            int(args.num_workers),
            "lmdb",
        )
        packer()
    print("Process lmdb end!")

    print("Save res start!")
    map_file_path = args.src_path + "/map_files/"
    a = Argoverse1Dataset(data_path=lmdb_tmp_path, map_path=map_file_path)

    file_base_name = args.dst_path
    if os.path.exists(file_base_name):
        if len(os.listdir(file_base_name)) != 0:
            print(f"[Warning]: Directory '{file_base_name}' is not empty, " +
                  "please rm -rf it first, or change a empty directory.")
            sys.exit()
    else:
        os.mkdir(file_base_name)

    # ===========for single batch data===========
    # file_name = ["goals_2d","goals_2d_mask","instance_mask","lane_feat","traj_feat"]
    # for indx in tqdm(range(len(a))):
    #     for i in range(len(file_name)):
    #         print(a[indx]["file_name"])
    #         save_name = file_base_name + "/" + a[indx]["file_name"].split("/")[-1].split(".")[0] + "_" + file_name[i] + ".bin"
    #         # print(save_name)
    #         with open(save_name, "w") as sf:
    #             data_float32 = a[indx][file_name[i]].astype(np.float32)
    #             data_float32.tofile(sf)

    print("Save process res start!")
    file_names = ["input3", "input4", "input2", "input1", "input0"]
    for indx in tqdm(range(0, len(a), int(args.num_batch))):
        pack_name = a[indx]["file_name"].split("/")[-1].split(".")[0] + "_"
        pack_data = []
        # print("a[indx]['goals_2d'].shape: ",a[indx]["goals_2d"].shape)
        pack_data.append(a[indx]["goals_2d"].reshape(1, 2, 1, 2048))
        # pack_data.append(a[indx]["goals_2d_mask"].reshape(1, 2048, 1, 1))
        pack_data.append(a[indx]["goals_2d_mask"].reshape(1, 1, 1, 2048))
        pack_data.append(a[indx]["instance_mask"].reshape(1, 1, 1, 96))
        pack_data.append(a[indx]["lane_feat"].reshape(1, 11, 9, 64))
        pack_data.append(a[indx]["traj_feat"].reshape(1, 9, 19, 32))

        for pack_id in range(1, int(args.num_batch)):
            real_pack_id = 0
            if ((indx + pack_id) < len(a)):
                real_pack_id = indx + pack_id
            else:
                # be sure have batch data packed
                real_pack_id = indx - pack_id

            pack_name += a[real_pack_id]["file_name"].split("/")[-1].split(
                ".")[0] + "_"
            tmp_goals_2d = a[real_pack_id]["goals_2d"].reshape(1, 2, 1, 2048)
            pack_data[0] = np.concatenate((pack_data[0], tmp_goals_2d), axis=0)
            tmp_goals_2d_mask = a[real_pack_id]["goals_2d_mask"].reshape(
                1, 1, 1, 2048)
            pack_data[1] = np.concatenate((pack_data[1], tmp_goals_2d_mask),
                                          axis=0)
            tmp_instance_mask = a[real_pack_id]["instance_mask"].reshape(
                1, 1, 1, 96)
            pack_data[2] = np.concatenate((pack_data[2], tmp_instance_mask),
                                          axis=0)
            tmp_lane_feat = a[real_pack_id]["lane_feat"].reshape(1, 11, 9, 64)
            pack_data[3] = np.concatenate((pack_data[3], tmp_lane_feat),
                                          axis=0)
            tmp_traj_feat = a[real_pack_id]["traj_feat"].reshape(1, 9, 19, 32)
            pack_data[4] = np.concatenate((pack_data[4], tmp_traj_feat),
                                          axis=0)

        pack_name = pack_name[:-1]
        for i in range(len(file_names)):
            save_name = file_base_name + "/" + pack_name + "-" + file_names[
                i] + ".bin"
            print(save_name)
            with open(save_name, "w") as sf:
                data_float32 = pack_data[i].astype(np.float32)
                data_float32.tofile(sf)
    print("Save process res end!")

    print("Save meta start!")
    meta_base_save_name = args.src_path + "/meta"
    if os.path.exists(meta_base_save_name):
        print(f"{meta_base_save_name} dir exits!")
        try:
            shutil.rmtree(meta_base_save_name)
            print(f"Directory '{meta_base_save_name}' successfully removed.")
            os.makedirs(meta_base_save_name, exist_ok=True)
        except OSError as e:
            print(
                f"Error while deleting directory '{meta_base_save_name}': {e}")
    else:
        os.mkdir(meta_base_save_name)

    for i in tqdm(range(len(a))):
        meta_save_name = meta_base_save_name + "/" + a[i]["file_name"].split(
            "/")[-1].split(".")[0] + "_traj_labels_meta.bin"
        with open(meta_save_name, "w") as sf:
            data_float32 = a[i]["traj_labels"].astype(np.float32)
            data_float32.tofile(sf)
    print("Save meta end!")

    if (args.is_gen_visual_config == "true"):
        print("Save visualization config start...")
        visualization_save_base_name = args.src_path + "/visualization"
        if os.path.exists(visualization_save_base_name):
            print(f"{visualization_save_base_name} dir exits!")
            try:
                shutil.rmtree(visualization_save_base_name)
                print(f"Directory '{visualization_save_base_name}'" +
                      " successfully removed.")
                os.makedirs(visualization_save_base_name, exist_ok=True)
            except OSError as e:
                print("Error while deleting directory" +
                      f"'{visualization_save_base_name}': {e}")
        else:
            os.mkdir(visualization_save_base_name)

        visualization_name = ["traj_mask", "lane_mask", "feat_mask"]
        for i in tqdm(range(len(a))):
            for each_visualization_name in visualization_name:
                print(each_visualization_name)
                visualization_save_name = visualization_save_base_name + \
                    "/" + a[i]["file_name"].split("/")[-1].split(
                        ".")[0] + "_" + each_visualization_name + "_vis.bin"
                print(visualization_save_name)
                with open(visualization_save_name, "w") as sf:
                    data_float32 = a[i][each_visualization_name].astype(
                        np.float32)
                    data_float32.tofile(sf)
        print("Save visualization config end!")

    print("Process dataset end!")
