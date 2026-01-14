"""QCNet Pack argoverse2."""

import argparse
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils._pytree import tree_flatten
from tqdm import tqdm

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object

# Run the QCNet Argoverse2 data preprocessing script.
# Usage: python3 qcnet_argoverse2_preprocess.py
# Parameter Description:
# --src-data-dir <source data directory>    # Original data directory
# --target-data-dir <target data directory>  # Directory to save processed .bin data    # noqa
# --pack-type <pack type>                    # Data packing format, currently only supports lmdb # noqa
# --num-samples <number of samples>          # Number of samples to process; defaults to processing all validation samples. For generating min_data, specify as needed. # noqa
# Example Command:
# python3 qcnet_argoverse2_preprocess.py --src-data-dir <path1> --target-data-dir <path2> --pack-type lmdb --num-samples # noqa

__all__ = [
    "Argoverse2Base",
    "Argoverse2Packer",
]

def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def side_to_directed_lineseg(
    query_point: torch.Tensor,
    start_point: torch.Tensor,
    end_point: torch.Tensor,
) -> str:
    cond = (end_point[0] - start_point[0]) * (
        query_point[1] - start_point[1]
    ) - (end_point[1] - start_point[1]) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


class Argoverse2Base:
    """
    Argoverse2 dataset handler.

    Args:
        num_historical_steps: Number of historical time steps.
        num_future_step: Number of future time steps.
        split: Dataset split name, e.g., 'train', 'val', 'test'.
    """

    def __init__(
        self,
        num_historical_steps: int = 60,
        num_future_steps: int = 50,
        split: str = "val",
    ):
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.split = split
        assert split in ["train", "test", "val"]

        self._agent_types = [
            "vehicle",
            "pedestrian",
            "motorcyclist",
            "cyclist",
            "bus",
            "static",
            "background",
            "construction",
            "riderless_bicycle",
            "unknown",
        ]
        self._agent_categories = [
            "TRACK_FRAGMENT",
            "UNSCORED_TRACK",
            "SCORED_TRACK",
            "FOCAL_TRACK",
        ]
        self._polygon_types = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
        self._polygon_is_intersections = [True, False, None]
        self._point_types = [
            "DASH_SOLID_YELLOW",
            "DASH_SOLID_WHITE",
            "DASHED_WHITE",
            "DASHED_YELLOW",
            "DOUBLE_SOLID_YELLOW",
            "DOUBLE_SOLID_WHITE",
            "DOUBLE_DASH_YELLOW",
            "DOUBLE_DASH_WHITE",
            "SOLID_YELLOW",
            "SOLID_WHITE",
            "SOLID_DASH_WHITE",
            "SOLID_DASH_YELLOW",
            "SOLID_BLUE",
            "NONE",
            "UNKNOWN",
            "CROSSWALK",
            "CENTERLINE",
        ]
        self._point_sides = ["LEFT", "RIGHT", "CENTER"]
        self._polygon_to_polygon_types = [
            "NONE",
            "PRED",
            "SUCC",
            "LEFT",
            "RIGHT",
        ]

    # @require_packages("av2")
    def process(self, input_dir: str):
        """
        Process data from the specified input directory.

        Args:
            input_dir: Path to the directory containing input data.

        Returns:
            pd.DataFrame: Processed data as a Pandas DataFrame.
        """
        assert (
            ArgoverseStaticMap is not None
        ), "av2 should be installed for argoverse2 data process."
        raw_file_name = os.path.basename(input_dir)
        map_dir = Path(input_dir)
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)

        df = pd.read_parquet(
            os.path.join(
                input_dir,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["agent"] = self.get_agent_features(df)
        data.update(self.get_map_features(map_api, centerlines))

        return data, map_path

    def _decode(self, sample: dict, input_dim: int):

        for k in [
            "valid_mask",
            "predict_mask",
            "type",
            "category",
            "position",
            "heading",
            "velocity",
        ]:
            sample["agent"][k] = torch.tensor(sample["agent"][k]).clone()

        for k in [
            "position",
            "orientation",
            "height",
            "type",
            "is_intersection",
        ]:
            sample["map_polygon"][k] = torch.tensor(
                sample["map_polygon"][k]
            ).clone()
        for k in [
            "position",
            "orientation",
            "magnitude",
            "height",
            "type",
            "side",
        ]:
            sample["map_point"][k] = [
                torch.tensor(p) for p in sample["map_point"][k]
            ]
        sample["map_point_to_map_polygon"]["edge_index"] = (
            torch.tensor(sample["map_point_to_map_polygon"]["edge_index"])
            .long()
            .clone()
        )
        sample["map_polygon_to_map_polygon"]["edge_index"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["edge_index"]
        ).clone()
        sample["map_polygon_to_map_polygon"]["type"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["type"]
        ).clone()

        sample["map_polygon"]["position"] = sample["map_polygon"]["position"][
            :, :input_dim
        ]
        pl_num = sample["map_polygon"]["num_nodes"]
        pl2pl_type_mat = torch.zeros([pl_num, pl_num])
        for i in range(
            sample["map_polygon_to_map_polygon"]["edge_index"].shape[1]
        ):
            src, dst = sample["map_polygon_to_map_polygon"]["edge_index"][:, i]
            pl2pl_type_mat[src][dst] = sample["map_polygon_to_map_polygon"][
                "type"
            ][i]
        sample["map_polygon_to_map_polygon"]["type_mat"] = pl2pl_type_mat
        return sample

    # @require_packages("av2")
    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:
            historical_df = df[df["timestep"] < self.num_historical_steps]
            agent_ids = list(historical_df["track_id"].unique())
            df = df[df["track_id"].isin(agent_ids)]
        else:
            agent_ids = list(df["track_id"].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index("AV")

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(
            num_agents, self.num_steps, dtype=torch.bool
        )
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )

        for track_id, track_df in df.groupby("track_id"):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df["timestep"].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[
                agent_idx, self.num_historical_steps - 1
            ]
            predict_mask[agent_idx, agent_steps] = True
            if (
                self.vector_repr
            ):  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1 : self.num_historical_steps] = (
                    valid_mask[agent_idx, : self.num_historical_steps - 1]
                    & valid_mask[agent_idx, 1 : self.num_historical_steps]
                )
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, : self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps :] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(
                track_df["object_type"].values[0]
            )
            agent_category[agent_idx] = track_df["object_category"].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["position_x"].values,
                        track_df["position_y"].values,
                    ],
                    axis=-1,
                )
            ).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(
                track_df["heading"].values
            ).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["velocity_x"].values,
                        track_df["velocity_y"].values,
                    ],
                    axis=-1,
                )
            ).float()

        if self.split == "test":
            predict_mask[
                current_valid_mask
                | (agent_category == 2)
                | (agent_category == 3),
                self.num_historical_steps :,
            ] = True

        return {
            "num_nodes": num_agents,
            "av_index": av_idx,
            "valid_mask": valid_mask.numpy().tolist(),
            "predict_mask": predict_mask.numpy().tolist(),
            "id": agent_id,
            "type": agent_type.numpy().tolist(),
            "category": agent_category.numpy().tolist(),
            "position": position.numpy().tolist(),
            "heading": heading.numpy().tolist(),
            "velocity": velocity.numpy().tolist(),
        }

    # @require_packages("av2")
    def get_map_features(
        self, map_api: ArgoverseStaticMap, centerlines: Mapping[str, Polyline]
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(
            num_polygons, self.dim, dtype=torch.float
        )
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(
                centerlines[lane_segment.id].xyz
            ).float()
            polygon_position[lane_segment_idx] = centerline[0, : self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(
                centerline[1, 1] - centerline[0, 1],
                centerline[1, 0] - centerline[0, 0],
            )
            polygon_height[lane_segment_idx] = (
                centerline[1, 2] - centerline[0, 2]
            )
            polygon_type[lane_segment_idx] = self._polygon_types.index(
                lane_segment.lane_type.value
            )
            polygon_is_intersection[
                lane_segment_idx
            ] = self._polygon_is_intersections.index(
                lane_segment.is_intersection
            )

            left_boundary = torch.from_numpy(
                lane_segment.left_lane_boundary.xyz
            ).float()
            right_boundary = torch.from_numpy(
                lane_segment.right_lane_boundary.xyz
            ).float()
            point_position[lane_segment_idx] = torch.cat(
                [
                    left_boundary[:-1, : self.dim],
                    right_boundary[:-1, : self.dim],
                    centerline[:-1, : self.dim],
                ],
                dim=0,
            )
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat(
                [
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                    torch.atan2(center_vectors[:, 1], center_vectors[:, 0]),
                ],
                dim=0,
            )
            point_magnitude[lane_segment_idx] = torch.norm(
                torch.cat(
                    [
                        left_vectors[:, :2],
                        right_vectors[:, :2],
                        center_vectors[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_height[lane_segment_idx] = torch.cat(
                [
                    left_vectors[:, 2],
                    right_vectors[:, 2],
                    center_vectors[:, 2],
                ],
                dim=0,
            )
            left_type = self._point_types.index(
                lane_segment.left_mark_type.value
            )
            right_type = self._point_types.index(
                lane_segment.right_mark_type.value
            )
            center_type = self._point_types.index("CENTERLINE")
            point_type[lane_segment_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),), left_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(right_vectors),), right_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_side[lane_segment_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[: self.dim]
            polygon_position[
                crosswalk_idx + len(cross_walk_ids)
            ] = end_position[: self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2(
                (end_position - start_position)[1],
                (end_position - start_position)[0],
            )
            polygon_orientation[
                crosswalk_idx + len(cross_walk_ids)
            ] = torch.atan2(
                (start_position - end_position)[1],
                (start_position - end_position)[0],
            )
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = (
                start_position[2] - end_position[2]
            )
            polygon_type[crosswalk_idx] = self._polygon_types.index(
                "PEDESTRIAN"
            )
            polygon_type[
                crosswalk_idx + len(cross_walk_ids)
            ] = self._polygon_types.index("PEDESTRIAN")
            polygon_is_intersection[
                crosswalk_idx
            ] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[
                crosswalk_idx + len(cross_walk_ids)
            ] = self._polygon_is_intersections.index(None)

            if (
                side_to_directed_lineseg(
                    (edge1[0] + edge1[-1]) / 2, start_position, end_position
                )
                == "LEFT"
            ):
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = (
                math.ceil(
                    torch.norm(
                        end_position - start_position, p=2, dim=-1
                    ).item()
                    / 2.0
                )
                + 1
            )
            centerline = torch.from_numpy(
                compute_midpoint_line(
                    left_ln_boundary=left_boundary.numpy(),
                    right_ln_boundary=right_boundary.numpy(),
                    num_interp_pts=int(num_centerline_points),
                )[0]
            ).float()

            point_position[crosswalk_idx] = torch.cat(
                [
                    left_boundary[:-1, : self.dim],
                    right_boundary[:-1, : self.dim],
                    centerline[:-1, : self.dim],
                ],
                dim=0,
            )
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    right_boundary.flip(dims=[0])[:-1, : self.dim],
                    left_boundary.flip(dims=[0])[:-1, : self.dim],
                    centerline.flip(dims=[0])[:-1, : self.dim],
                ],
                dim=0,
            )
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                    torch.atan2(center_vectors[:, 1], center_vectors[:, 0]),
                ],
                dim=0,
            )
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.atan2(
                        -right_vectors.flip(dims=[0])[:, 1],
                        -right_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -left_vectors.flip(dims=[0])[:, 1],
                        -left_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -center_vectors.flip(dims=[0])[:, 1],
                        -center_vectors.flip(dims=[0])[:, 0],
                    ),
                ],
                dim=0,
            )
            point_magnitude[crosswalk_idx] = torch.norm(
                torch.cat(
                    [
                        left_vectors[:, :2],
                        right_vectors[:, :2],
                        center_vectors[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat(
                    [
                        -right_vectors.flip(dims=[0])[:, :2],
                        -left_vectors.flip(dims=[0])[:, :2],
                        -center_vectors.flip(dims=[0])[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_height[crosswalk_idx] = torch.cat(
                [
                    left_vectors[:, 2],
                    right_vectors[:, 2],
                    center_vectors[:, 2],
                ],
                dim=0,
            )
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    -right_vectors.flip(dims=[0])[:, 2],
                    -left_vectors.flip(dims=[0])[:, 2],
                    -center_vectors.flip(dims=[0])[:, 2],
                ],
                dim=0,
            )
            crosswalk_type = self._point_types.index("CROSSWALK")
            center_type = self._point_types.index("CENTERLINE")
            point_type[crosswalk_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),), crosswalk_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(right_vectors),),
                        crosswalk_type,
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),),
                        crosswalk_type,
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(left_vectors),), crosswalk_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_side[crosswalk_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

        num_points = torch.tensor(
            [point.size(0) for point in point_position], dtype=torch.long
        )
        point_to_polygon_edge_index = torch.stack(
            [
                torch.arange(num_points.sum(), dtype=torch.long),
                torch.arange(num_polygons, dtype=torch.long).repeat_interleave(
                    num_points
                ),
            ],
            dim=0,
        )
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack(
                        [
                            torch.tensor(pred_inds, dtype=torch.long),
                            torch.full(
                                (len(pred_inds),),
                                lane_segment_idx,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                )
                polygon_to_polygon_type.append(
                    torch.full(
                        (len(pred_inds),),
                        self._polygon_to_polygon_types.index("PRED"),
                        dtype=torch.uint8,
                    )
                )
            succ_inds = []
            for succ in lane_segment.successors:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack(
                        [
                            torch.tensor(succ_inds, dtype=torch.long),
                            torch.full(
                                (len(succ_inds),),
                                lane_segment_idx,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                )
                polygon_to_polygon_type.append(
                    torch.full(
                        (len(succ_inds),),
                        self._polygon_to_polygon_types.index("SUCC"),
                        dtype=torch.uint8,
                    )
                )
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(
                    polygon_ids, lane_segment.left_neighbor_id
                )
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor(
                            [[left_idx], [lane_segment_idx]], dtype=torch.long
                        )
                    )
                    polygon_to_polygon_type.append(
                        torch.tensor(
                            [self._polygon_to_polygon_types.index("LEFT")],
                            dtype=torch.uint8,
                        )
                    )
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(
                    polygon_ids, lane_segment.right_neighbor_id
                )
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor(
                            [[right_idx], [lane_segment_idx]], dtype=torch.long
                        )
                    )
                    polygon_to_polygon_type.append(
                        torch.tensor(
                            [self._polygon_to_polygon_types.index("RIGHT")],
                            dtype=torch.uint8,
                        )
                    )
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(
                polygon_to_polygon_edge_index, dim=1
            )
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor(
                [[], []], dtype=torch.long
            )
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            "map_polygon": {},
            "map_point": {},
            "map_point_to_map_polygon": {},
            "map_polygon_to_map_polygon": {},
        }
        map_data["map_polygon"][
            "num_nodes"
        ] = num_polygons  # .numpy().tolist()
        map_data["map_polygon"]["position"] = polygon_position.numpy().tolist()
        map_data["map_polygon"][
            "orientation"
        ] = polygon_orientation.numpy().tolist()
        if self.dim == 3:
            map_data["map_polygon"]["height"] = polygon_height.numpy().tolist()
        map_data["map_polygon"]["type"] = polygon_type.numpy().tolist()
        map_data["map_polygon"][
            "is_intersection"
        ] = polygon_is_intersection.numpy().tolist()
        if len(num_points) == 0:
            map_data["map_point"]["num_nodes"] = 0
            map_data["map_point"]["position"] = []
            map_data["map_point"]["orientation"] = []
            map_data["map_point"]["magnitude"] = []
            if self.dim == 3:
                map_data["map_point"]["height"] = []
            map_data["map_point"]["type"] = []
            map_data["map_point"]["side"] = []
        else:
            map_data["map_point"]["num_nodes"] = num_points.sum().item()

            map_data["map_point"]["num_points"] = num_points.numpy().tolist()
            map_data["map_point"]["position"] = [
                p.numpy().tolist() for p in point_position
            ]
            map_data["map_point"]["orientation"] = [
                p.numpy().tolist() for p in point_orientation
            ]
            map_data["map_point"]["magnitude"] = [
                p.numpy().tolist() for p in point_magnitude
            ]
            if self.dim == 3:
                map_data["map_point"]["height"] = [
                    p.numpy().tolist() for p in point_height
                ]
            map_data["map_point"]["type"] = [
                p.numpy().tolist() for p in point_type
            ]
            map_data["map_point"]["side"] = [
                p.numpy().tolist() for p in point_side
            ]

        map_data["map_point_to_map_polygon"][
            "edge_index"
        ] = point_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "edge_index"
        ] = polygon_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "type"
        ] = polygon_to_polygon_type.numpy().tolist()

        return map_data


class Argoverse2Packer:
    """Argoverse2 Dataset Packer.

    Args:
        src_data_dir: The directory path where the source data is located.
        target_data_dir: The path where the packed data will be stored.
        split_name: The name of the dataset split to be packed
                 optional: ('train', 'test').
        num_workers: The number of workers to use for parallel processing.
        pack_type: The type of packing to be performed.
        num_samples: The number of samples to pack.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps for prediction.
        **kwargs: Additional keyword arguments for the packing process.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str = "val",
        num_workers: int = 10,
        pack_type: str = "lmdb",
        num_samples: Optional[int] = None,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        input_dim: int = 2,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.data_root = src_data_dir
        self.split = split_name
        assert self.split in ["train", "test", "val"]
        self._raw_file_names = sorted(
            name
            for name in os.listdir(os.path.join(self.data_root, self.split))
            if os.path.isdir(os.path.join(self.data_root, self.split, name))
        )  # "val": 24988
        self.raw_dir = os.path.join(self.data_root, self.split)
        if num_samples is None:
            num_samples = len(self._raw_file_names)
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self._num_samples = {
            "train": 199908,
            "val": 24988,
            "test": 24984,
        }[self.split]
        self.argoverse2 = Argoverse2Base(
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            split=self.split,
        )
        # super(Argoverse2Packer, self).__init__(
        #    target_data_dir, num_samples, pack_type, num_workers, **kwargs
        # )

    def __call__(self, idx):
        raw_file_name = self._raw_file_names[idx]
        df = pd.read_parquet(
            os.path.join(
                self.raw_dir,
                raw_file_name,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        map_dir = Path(self.raw_dir) / raw_file_name
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["scenario_id"] = df["scenario_id"].values[0]
        data["city"] = df["city"].values[0]
        data["agent"] = self.argoverse2.get_agent_features(df)
        data.update(self.argoverse2.get_map_features(map_api, centerlines))

        # data type: list -->torch
        sample = self.argoverse2._decode(data, self.input_dim)

        return sample, raw_file_name


ori_historical_sec = (
    5  # Length of the original historical trajectory in seconds）
)
ori_future_sec = 6  # Length of the predicted future trajectory in seconds
ori_sample_fre = (
    10  # Original sampling frequency, which is 10 steps per second
)
sample_fre = 2  # Frequency used for downsampling
num_future_steps = ori_future_sec * sample_fre
num_t2m_steps = 30 // ori_sample_fre * sample_fre  # times steps for decode
time_span = 2
B = 1
A = 30
pl = 80
pt = 50
num_historical_steps = HT = (
    ori_historical_sec * sample_fre
)  # if sample_fre==2 init HT=10
input_dim = 2
hidden_dim = D = 128
num_agent_layers = 1


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:

    angle = torch.where(angle < min_val, angle + 2 * math.pi, angle)
    angle = torch.where(angle > max_val, angle - 2 * math.pi, angle)
    return angle


def get_bin_input(deploy_inputs, sample_id, idx, target_data_dir):

    flatten_data = tree_flatten(deploy_inputs)[0]
    if idx == 0:
        print(len(flatten_data))

    for i in range(len(flatten_data)):
        if idx == 0:
            shapes = flatten_data[i].shape
            shapes = [str(s) for s in shapes]
            s = "x".join(shapes)
            print(s, end=", ")
        flatten_data[i] = flatten_data[i].numpy().tobytes()

    for i in range(len(flatten_data)):
        save_path = os.path.join(
            target_data_dir,
            f"sample_{idx}-{sample_id}",
            f"sample_{idx}-{sample_id}-input{i}.bin",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flatten_data[i])


class QCNetOEPreprocess:
    """Preprocess Module for OE version QCNet deploy.

    Args:
        input_dim: Dimension of the input data.
        num_historical_steps: Number of historical time steps.
        time_span: Time span for the model.
        num_t2m_steps: Number of time steps for the decoder's cross attention
            along the time axis.
        num_agent_layers: Number of layers for the agent module.
        num_pl2a: Number of polygons for map polygon to agent cross attention
                (works only when save_memory=True).
        num_a2a: Number of agents for agent to agent cross attention.
                (works only when save_memory=True).
        agent_num: Number of agents in the dataset.
        pl_num: Number of polygons for map polygon to agent cross attention.
        pt_num: Number of points for map polygon to agent cross attention.
        stream: Whether using stream mode for inference.
            Only works when deploy=False.
        save_memory: whether select agent and map neighbors to save memory,
            currently only works for training.
        deploy: Flag to indicate deployment mode. Default is False.
        reuse_agent_rembs: Flag to reuse agent rembs for the decoder.
        quant_infer_cold_start:Indicates whether to use cold start
            for streaming inference.
    """

    def __init__(
        self,
        input_dim: int,
        num_historical_steps: int,
        time_span: int,
        num_t2m_steps: int,
        num_agent_layers: int,
        num_pl2a: int = 32,
        num_a2a: int = 36,
        num_pl2pl: int = -1,
        agent_num: Optional[int] = None,
        pl_num: Optional[int] = None,
        pt_num: Optional[int] = None,
        mask_pl2pl_type: str = "diagonal_valid",
        stream: bool = False,
        save_memory: bool = False,
        reuse_agent_rembs: bool = True,
        deploy: bool = False,
        quant_infer_cold_start: bool = False,
    ):
        super().__init__()
        self.time_span = time_span
        self.input_dim = input_dim
        self.A = agent_num
        self.ht = num_historical_steps
        self.num_t2m_steps = num_t2m_steps
        self.pl_num = pl_num
        self.pt_num = pt_num
        self.num_pl2pl = num_pl2pl
        self.num_pl2a = num_pl2a
        self.num_a2a = num_a2a
        self.num_agent_layers = num_agent_layers
        self.save_memory = save_memory
        self.stream = stream
        self.reuse_agent_rembs = reuse_agent_rembs
        self.deploy = deploy
        self.mask_pl2pl_type = mask_pl2pl_type
        self.quant_infer_cold_start = quant_infer_cold_start

    def build_map_r_inputs(self, data: dict):
        pos_pt = data["map_point"]["position"] / 10.0
        orient_pt = data["map_point"]["orientation"]
        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]

        rel_pos_pt2pl = pos_pt - pos_pl[:, :, None]  # [B, pl, pt, 2]
        rel_orient_pt2pl = wrap_angle(
            orient_pt - orient_pl[:, :, None]
        )  # [B, pl, pt]
        theta_pt2pl = torch.atan2(rel_pos_pt2pl[..., 1], rel_pos_pt2pl[..., 0])
        angle_pt2pl = wrap_angle(
            theta_pt2pl - orient_pl[:, :, None],  # [B, pl, pt] - B, pl]
        )  # [B, pl, pt]
        dist_pt2pl = torch.norm(rel_pos_pt2pl[..., :2], dim=-1, p=2)
        r_pt2pl = [
            dist_pt2pl.unsqueeze(1),  # [B, 1, pl, pt]
            angle_pt2pl.unsqueeze(1),
            rel_orient_pt2pl.unsqueeze(1),
        ]

        B, pl_N = pos_pl.shape[:2]  # [B, pl, 2]
        # mask_pl2pl
        valid_pl = data["map_polygon"]["valid_mask"]  # [B, pl]

        if self.mask_pl2pl_type == "diagonal":
            mask_pl2pl = (
                (torch.ones([B, pl_N, pl_N]) - torch.eye(pl_N).unsqueeze(0))
                .to(pos_pl.device)
                .bool()
            )
        elif self.mask_pl2pl_type == "valid":
            mask_pl2pl = (
                valid_pl[:, :, None] & valid_pl[:, None, :]
            )  # [B, pl, pl]
        elif self.mask_pl2pl_type == "diagonal_valid":
            valid_mask_pl2pl = valid_pl[:, :, None] & valid_pl[:, None, :]
            dia_mask_pl2pl = (
                (torch.ones([B, pl_N, pl_N]) - torch.eye(pl_N).unsqueeze(0))
                .to(pos_pl.device)
                .bool()
            )
            mask_pl2pl = valid_mask_pl2pl & dia_mask_pl2pl

        rel_pos_pl2pl = (
            pos_pl[:, None, :] - pos_pl[:, :, None]
        )  # #[B, pl, pl, 2]
        dist_pl2pl = torch.norm(rel_pos_pl2pl[..., :2], dim=-1, p=2)

        B, pl_N = pos_pl.shape[:2]  # [B, pl, 2]
        pl_K = self.num_pl2pl

        if pl_K > 0 and pl_N > pl_K and self.save_memory:
            # mask_dist_pl2pl= dist_pl2pl.masked_fill(~mask_pl2pl, 1e2)

            # topk_dist_pl2pl, pl2pl_near_idx = torch.topk(
            #    mask_dist_pl2pl,
            #    k=pl_K,
            #    dim=-1,
            #    largest=False,
            # )   # [B, pl_N, pl_K]

            pl2pl_near_idx = (
                (torch.rand([B, pl_N, pl_K]) * pl_K)
                .long()
                .to(dist_pl2pl.device)
            )

            topk_dist_pl2pl = dist_pl2pl.gather(
                -1, pl2pl_near_idx
            )  # [B, pl_N, pl_K]

            valid_pl2pl = mask_pl2pl.gather(-1, pl2pl_near_idx)
            dist_pl2pl = topk_dist_pl2pl.masked_fill(~valid_pl2pl, 0)
            # dist_pl2pl = dist_pl2pl.gather(-1, pl2pl_near_idx)
            rel_orient_pl2pl = wrap_angle(
                orient_pl[:, :, None] - orient_pl[:, None, :]
            ).gather(-1, pl2pl_near_idx)
            rel_pos_pl2pl = rel_pos_pl2pl.gather(
                2, pl2pl_near_idx.unsqueeze(-1).repeat(1, 1, 1, 2)
            )  # [B, pl, pl, 2]-->[B, pl, pl_K, 2]
            theta_pl2pl = torch.atan2(
                rel_pos_pl2pl[..., 1], rel_pos_pl2pl[..., 0]
            )  # [B, pl, pl_K]
            angle_pl2pl = wrap_angle(theta_pl2pl - orient_pl[:, :, None])
            mask_pl2pl = mask_pl2pl.gather(-1, pl2pl_near_idx)  # [B, pl_K]
            type_pl2pl = data["type_pl2pl"]  # [B, pl_N, pl_K]
            data["type_pl2pl"] = type_pl2pl.gather(
                -1, pl2pl_near_idx
            )  # [B, pl_N, pl_K]

        else:
            rel_orient_pl2pl = wrap_angle(
                orient_pl[:, :, None] - orient_pl[:, None, :]
            )
            theta_pl2pl = torch.atan2(
                rel_pos_pl2pl[..., 1], rel_pos_pl2pl[..., 0]
            )
            angle_pl2pl = wrap_angle(theta_pl2pl - orient_pl[:, :, None])
            dist_pl2pl = torch.norm(rel_pos_pl2pl[..., :2], dim=-1, p=2)
            pl2pl_near_idx = None

        r_pl2pl = [
            dist_pl2pl.unsqueeze(1),
            angle_pl2pl.unsqueeze(1),
            rel_orient_pl2pl.unsqueeze(1),
        ]
        if pl2pl_near_idx is not None:
            data["pl2pl_near_idx"] = pl2pl_near_idx

        return {
            "r_pl2pl": r_pl2pl,
            "r_pt2pl": r_pt2pl,
            "mask_pl2pl": mask_pl2pl,
        }

    def build_agent_his_r_inputs(self, data: dict, ht: int, qt: int):
        B, A = data["agent"]["position"].shape[:2]
        ST = self.time_span
        bt = ht - qt - ST  

        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]
        pos_a = (
            data["agent"]["position"][:, :, bt:ht, :2] / 10.0
        )  # [B, A, HT, 2]
        head_a = data["agent"]["heading"][
            :, :, bt:ht
        ].contiguous()  # [B, A, HT]
        vel = (
            data["agent"]["velocity"][
                :, :, bt:ht, : self.input_dim
            ].contiguous()
            / 10.0
        )
        dist_vel = torch.norm(vel, p=2, dim=-1)

        motion_vector_a = torch.cat(
            [
                torch.zeros(B, A, 1, self.input_dim).to(pos_a.device),
                pos_a[:, :, 1:] - pos_a[:, :, :-1],
            ],
            dim=2,
        )  # [B, A, HT, 2]

        # [B, A, HT, D]
        theta_motion_a = torch.atan2(
            motion_vector_a[..., 1], motion_vector_a[..., 0]
        )
        angle_motion = wrap_angle(theta_motion_a - head_a)  # [B, A, HT]
        theta_vel = torch.atan2(vel[..., 1], vel[..., 0])
        angle_vel = wrap_angle(theta_vel - head_a)  # [B, A, HT]
        dist_motion_a = torch.norm(motion_vector_a[..., :2], dim=-1, p=2)
        x_a = [
            dist_motion_a.unsqueeze(1),
            angle_motion.unsqueeze(1),
            dist_vel.unsqueeze(1),
            angle_vel.unsqueeze(1),
        ]
        pos_t_key = torch.cat(
            [
                pos_a[:, :, ST - i : ST + qt - i].unsqueeze(3)
                for i in range(ST, 0, -1)
            ],
            dim=3,
        ).contiguous()  # [B, A, qt, 2, 2]

        head_t_key = torch.cat(
            [
                head_a[:, :, ST - i : ST + qt - i].unsqueeze(3)
                for i in range(ST, 0, -1)
            ],
            dim=3,
        ).contiguous()

        # [B, D, A, HT]
        pos_a = pos_a[:, :, ST : self.ht, :2].contiguous()  # [B, A, HT, 2]
        head_a = head_a[:, :, ST : self.ht].contiguous()  # [B, A, HT]
        vel = vel[:, :, ST : self.ht, : self.input_dim].contiguous()

        rel_pos_t = pos_a[:, :, :, None, :] - pos_t_key
        # [B, A, QT, ST, 2]
        rel_head_t = wrap_angle(head_a[:, :, :, None] - head_t_key)
        # [B, A, QT, ST,]
        theta_pos_t = torch.atan2(rel_pos_t[..., 1], rel_pos_t[..., 0])
        angle_agent_t = wrap_angle(
            theta_pos_t - head_a[:, :, :, None]
        )  # [B, A, HT, HT]
        diff_t = (
            torch.arange(ST)
            .reshape(1, 1, 1, ST)
            .repeat(B * A, 1, qt, 1)
            .to(pos_a.device)
        )

        dist_pos_t = torch.norm(rel_pos_t[..., :2], dim=-1, p=2)
        r_t = [
            dist_pos_t.view(B * A, 1, qt, ST),
            angle_agent_t.view(B * A, 1, qt, ST),
            rel_head_t.view(B * A, 1, qt, ST),
            diff_t.float().view(B * A, 1, qt, ST),
        ]

        pl_N = pos_pl.shape[1]
        rel_pos_pl2a = pos_pl[:, None, None, :, :] - pos_a[:, :, :, None, :]
        agent_valid = data["agent"]["valid_mask"][:, :, bt:ht][
            :, :, ST : self.ht
        ]  # [B, A, 5]
        pl_valid = data["map_polygon"]["valid_mask"]  # [B, pl]
        pl2a_valid = (
            pl_valid[:, None, None, :] & agent_valid[:, :, :, None]
        )  # [B, A, 5, pl]]

        if (
            self.num_pl2a > 0
            and pl_N > self.num_pl2a + 10
            and self.save_memory
        ):
            pl_K = self.num_pl2a
            pl_N = pl_K
            dist_pl2a = torch.norm(
                rel_pos_pl2a[..., :2], p=2, dim=-1
            )  # [B, A, T, pl]

            # rand gather
            pl_idx = (
                (torch.rand(B, A, qt, pl_K) * pl_K).long().to(dist_pl2a.device)
            )  # int64
            topk_dist_valid = dist_pl2a.gather(-1, pl_idx)  # [B, A, qt, pl_k]

            pl2a_valid = pl2a_valid.gather(-1, pl_idx)  # [B, A, 5, pl_k]
            topk_dist = topk_dist_valid.masked_fill(~pl2a_valid, 0)

            select_orient_pl = orient_pl.gather(
                -1, pl_idx.reshape(1, -1)
            ).reshape(B, A, qt, pl_K)
            topk_pos_pl2a = rel_pos_pl2a.gather(
                -2, pl_idx.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
            )
            rel_orient_pl2a = wrap_angle(
                select_orient_pl - head_a[:, :, :, None]
            )
            angle_pl2a = wrap_angle(
                torch.atan2(topk_pos_pl2a[..., 1], topk_pos_pl2a[..., 0])
                - head_a[:, :, :, None]
            )
            r_pl2a = [
                topk_dist.view(B * A, 1, qt, pl_K),
                angle_pl2a.view(B * A, 1, qt, pl_K),
                rel_orient_pl2a.view(B * A, 1, qt, pl_K),
            ]
        else:
            # [B, A, QT, pl]
            rel_orient_pl2a = wrap_angle(
                orient_pl[:, None, None, :] - head_a[:, :, :, None]
            )
            theta_pl2a = torch.atan2(
                rel_pos_pl2a[..., 1], rel_pos_pl2a[..., 0]
            )
            angle_pl2a = wrap_angle(theta_pl2a - head_a[:, :, :, None])
            dist_pl2a = torch.norm(rel_pos_pl2a, p=2, dim=-1)
            r_pl2a = [
                dist_pl2a.view(B * A, 1, qt, pl_N),
                angle_pl2a.view(B * A, 1, qt, pl_N),
                rel_orient_pl2a.view(B * A, 1, qt, pl_N),
            ]
            pl_idx = None

        pos_ta = pos_a.transpose(1, 2)  # [B, T, A, 2]
        head_ta = head_a.transpose(1, 2)  # [B, T, A, 2]
        # [B, T, A, 1] & [B, T, 1, A]
        rel_pos_a2a = (
            pos_ta[:, :, :, None, :] - pos_ta[:, :, None, :, :]
        )  # [B, T, A, A, 2]

        agent_valid = data["agent"]["valid_mask"][:, :, bt:ht][
            :, :, ST : self.ht
        ]  # [B, A, 5]
        agent_valid = agent_valid.transpose(1, 2)  # [B, 5, A]

        if self.mask_pl2pl_type == "diagonal":
            a2a_valid = (
                (
                    torch.ones([B, agent_valid.shape[1], A, A])
                    - torch.eye(A)[None, None, :, :]
                )
                .to(agent_valid.device)
                .bool()
            )
        elif self.mask_pl2pl_type == "valid":
            a2a_valid = (
                agent_valid[:, :, :, None] & agent_valid[:, :, None, :]
            )  # [B, 5, A, A]
        else:
            dia_a2a_valid = (
                (
                    torch.ones([B, agent_valid.shape[1], A, A])
                    - torch.eye(A)[None, None, :, :]
                )
                .to(agent_valid.device)
                .bool()
            )
            val_a2a = (
                agent_valid[:, :, :, None] & agent_valid[:, :, None, :]
            )  # [B, 5, A, A]
            a2a_valid = dia_a2a_valid & val_a2a

        if self.num_a2a > 0 and A > self.num_a2a + 20 and self.save_memory:
            A_K = self.num_a2a
            theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
            dist_a2a = torch.norm(rel_pos_a2a[..., :2], p=2, dim=-1)

            # rand gather
            a2a_idx = (
                (torch.rand(B, qt, A, A_K) * A_K).long().to(dist_a2a.device)
            )  # int64
            topk_dist_a2a_valid = dist_a2a.gather(-1, a2a_idx)  # [B, A, A_K]

            a2a_valid = a2a_valid.gather(-1, a2a_idx)  # [B, 5, A, A_K]
            topk_dist_a2a = topk_dist_a2a_valid.masked_fill(~a2a_valid, 0)

            rel_head_a2a = wrap_angle(
                head_ta[:, :, :, None] - head_ta[:, :, None, :]
            )

            angle_a2a = wrap_angle(theta_a2a - head_ta[:, :, :, None])
            rel_head_a2a = rel_head_a2a.gather(3, a2a_idx)
            angle_a2a = angle_a2a.gather(3, a2a_idx)
            r_a2a = [
                topk_dist_a2a.reshape(B * qt, 1, A, A_K),
                angle_a2a.reshape(B * qt, 1, A, A_K),
                rel_head_a2a.reshape(B * qt, 1, A, A_K),
            ]
        else:
            A_K = A
            rel_head_a2a = wrap_angle(
                head_ta[:, :, :, None] - head_ta[:, :, None, :]
            )  # * mask_a2a #[B,T, A, A]
            theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
            angle_a2a = wrap_angle(
                theta_a2a - head_ta[:, :, :, None]
            )  # * mask_a2a
            dist_a2a = torch.norm(rel_pos_a2a[..., :2], p=2, dim=-1)
            r_a2a = [
                dist_a2a.reshape(B * qt, 1, A, A),
                angle_a2a.reshape(B * qt, 1, A, A),
                rel_head_a2a.reshape(B * qt, 1, A, A),
            ]
            a2a_idx = None

        out = {
            "x_a": x_a,  # [B,1, A, HT] *4
            "r_pl2a": r_pl2a,  # [B * A, 1, QT, pl_N] *3
            "r_t": r_t,  # [B*A, 1, QT, ST] *4
            "r_a2a": r_a2a,  # [B*QT, 1, A, A] *3
        }
        if pl_idx is not None:
            out["pl_idx"] = pl_idx
        if a2a_idx is not None:
            out["a2a_idx"] = a2a_idx
        return out

    def build_agent_cur_r_inputs(self, data: dict, cur: int):
        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]
        pos_a = data["agent"]["position"][:, :, :cur] / 10.0  # [B, A, HT, 2]
        head_a = data["agent"]["heading"][:, :, :cur]  # [B, A, HT]
        vel = data["agent"]["velocity"][:, :, :cur, : self.input_dim] / 10.0

        B, A = head_a.shape[:2]
        pos_cur = pos_a[:, :, cur - 1]
        head_cur = data["agent"]["heading"][:, :, cur - 1]
        motion_vector_a = pos_a[:, :, -1] - pos_a[:, :, -2]

        theta_motion_a = torch.atan2(
            motion_vector_a[..., 1], motion_vector_a[..., 0]
        )
        angle_motion = wrap_angle(theta_motion_a - head_cur)  # [B, A,]
        theta_vel = torch.atan2(vel[..., cur - 1, 1], vel[..., cur - 1, 0])
        angle_vel = wrap_angle(theta_vel - head_cur)  # [B, A]
        dist_vel = torch.norm(vel[:, :, cur - 1, :2], p=2, dim=-1)
        dist_motion_a = torch.norm(motion_vector_a[..., :2], p=2, dim=-1)  # BA
        x_a = [
            dist_motion_a.reshape(B, 1, A, 1),
            angle_motion.reshape(B, 1, A, 1),
            dist_vel.reshape(B, 1, A, 1),
            angle_vel.reshape(B, 1, A, 1),
        ]
        if not self.reuse_agent_rembs:
            ST = self.time_span
        else:
            if cur == self.ht:
                ST = self.num_t2m_steps  # get r_t for decoder
            else:
                ST = self.time_span

        pos_t_key = pos_a[:, :, cur - ST - 1 : cur - 1]
        head_t_key = head_a[:, :, cur - ST - 1 : cur - 1]
        rel_pos_t = pos_cur[:, :, None] - pos_t_key
        # [B, A, QT, ST, 2]
        rel_head_t = wrap_angle(head_cur[:, :, None] - head_t_key)
        # [B, A, QT, ST,]
        theta_pos_t = torch.atan2(rel_pos_t[..., 1], rel_pos_t[..., 0])
        angle_agent_t = wrap_angle(
            theta_pos_t - head_cur[:, :, None]
        )  # [B, A, HT, HT]

        diff_t = (
            torch.arange(ST)
            .reshape(1, 1, 1, ST)
            .repeat(B, 1, A, 1)
            .to(pos_a.device)
        )

        dist_pos_t = torch.norm(rel_pos_t[..., :2], p=2, dim=-1)
        r_t = [
            dist_pos_t.view(B, 1, A, ST),
            angle_agent_t.view(B, 1, A, ST),
            rel_head_t.view(B, 1, A, ST),
            diff_t.float().view(B, 1, A, ST),
        ]

        if cur < self.ht and self.quant_infer_cold_start:
            r_t_pedding = torch.zeros(B, 1, A, self.num_t2m_steps - ST).to(
                dist_pos_t.device
            )
            for i in range(len(r_t)):
                r_t[i] = torch.cat((r_t_pedding, r_t[i]), dim=-1)

        pl_N = pos_pl.shape[1]
        rel_pos_pl2a = pos_pl[:, None, :, :] - pos_cur[:, :, None, :]
        # [B, A, pl, 2]
        rel_orient_pl2a = wrap_angle(
            orient_pl[:, None, :] - head_cur[:, :, None]
        )  # [B, A, T, pl]
        theta_pl2a = torch.atan2(rel_pos_pl2a[..., 1], rel_pos_pl2a[..., 0])
        angle_pl2a = wrap_angle(theta_pl2a - head_cur[:, :, None])
        dist_pl2a = torch.norm(rel_pos_pl2a, dim=-1, p=2)
        r_pl2a = [
            dist_pl2a.view(B, 1, A, pl_N),
            angle_pl2a.view(B, 1, A, pl_N),
            rel_orient_pl2a.view(B, 1, A, pl_N),
        ]

        # [B, A, 1] - [B, 1, A]
        rel_pos_a2a = (
            pos_cur[:, :, None, :] - pos_cur[:, None, :, :]
        )  # [B, T, A, A, 2]
        rel_head_a2a = wrap_angle(
            head_cur[:, :, None] - head_cur[:, None, :]
        )  # [B, A, A]
        theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
        angle_a2a = wrap_angle(
            theta_a2a - head_cur[:, :, None]  # head_cur[:, None, :]
        )
        dist_a2a = torch.norm(rel_pos_a2a[..., :2], dim=-1, p=2)
        r_a2a = [
            dist_a2a.reshape(B, 1, A, A),
            angle_a2a.reshape(B, 1, A, A),
            rel_head_a2a.reshape(B, 1, A, A),
        ]

        mask_a_cur = data["agent"]["valid_mask"][:, :, cur - 1]  # [B, A]
        mask_a2a_cur = data["agent"]["valid_mask_a2a"][
            :, cur - 1, :, :A
        ]  # [B, A, A]
        mask_t_key = data["agent"]["valid_mask"][
            :, :, cur - self.time_span : cur
        ]  # [B, A, 2]

        return {
            "x_a_cur": x_a,
            "r_pl2a_cur": r_pl2a,
            "r_t_cur": r_t,  # [B, 1, A, 6]
            "r_a2a_cur": r_a2a,
            "mask_a_cur": mask_a_cur,
            "mask_a2a_cur": mask_a2a_cur,
            "mask_t_key": mask_t_key,
        }

    def build_decoder_r_inputs(self, data: dict):
        B, A = data["agent"]["position"].shape[:2]
        ht = self.ht
        pt = ht - self.num_t2m_steps

        pos_m = (
            data["agent"]["position"][:, :, self.ht - 1 : self.ht] / 10.0
        )  # [B, A, 1, 2]
        head_m = data["agent"]["heading"][
            :, :, self.ht - 1 : self.ht
        ]  # [B, A, 1, 2]
        pos_t = (
            data["agent"]["position"][:, :, pt : self.ht, : self.input_dim]
            / 10.0
        )  # [B, A, HT, 2]
        head_t = data["agent"]["heading"][:, :, pt : self.ht]  # [B, A, HT]
        pos_pl = data["map_polygon"]["position"][..., : self.input_dim] / 10.0
        orient_pl = data["map_polygon"]["orientation"]

        rel_pos_t2m = pos_t - pos_m  # [B, A, HT, 2]
        rel_head_t2m = wrap_angle(head_t - head_m)  # [B, A, HT]
        theta_t2m = torch.atan2(rel_pos_t2m[..., 1], rel_pos_t2m[..., 0])
        angle_t2m = wrap_angle(theta_t2m - head_m)  # [B, A, HT]
        arange_t = torch.arange(pt, self.ht).reshape(
            [1, 1, self.num_t2m_steps]
        )
        diff_t = (
            (arange_t.to(pos_m.device) - self.ht + 1).float().repeat(B, A, 1)
        )

        dist_t2m = torch.norm(rel_pos_t2m[..., :2], dim=-1, p=2)
        r_t2m = [
            dist_t2m.unsqueeze(1),
            angle_t2m.unsqueeze(1),
            rel_head_t2m.unsqueeze(1),
            diff_t.unsqueeze(1),
        ]  # [B, A, T, 4]

        # [B, pl, 2] - [B, A, 1, 2]
        rel_pos_pl2m = pos_pl[:, None, :, :] - pos_m  # [B, A, pl, 2]
        rel_orient_pl2m = wrap_angle(
            orient_pl[:, None, :] - head_m
        )  # [B, A, pl, 2]
        theta_pl2m = torch.atan2(rel_pos_pl2m[..., 1], rel_pos_pl2m[..., 0])
        angle_pl2m = wrap_angle(theta_pl2m - head_m)
        dist_pl2m = torch.norm(rel_pos_pl2m[..., :2], p=2, dim=-1)
        r_pl2m = [
            dist_pl2m.unsqueeze(1),
            angle_pl2m.unsqueeze(1),
            rel_orient_pl2m.unsqueeze(1),
        ]

        # [B, A, 1, 2]
        rel_pos_a2m = pos_m - pos_m.squeeze(2)[:, None, :]  # [B, A, A, 2]
        rel_head_a2m = wrap_angle(
            head_m - head_m.squeeze(2)[:, None, :]
        )  # [B, A, A]
        theta_a2m = torch.atan2(rel_pos_a2m[..., 1], rel_pos_a2m[..., 0])
        angle_a2m = wrap_angle(theta_a2m - head_m)  # [B, A, A,]
        dist_a2m = torch.norm(rel_pos_a2m[..., :2], p=2, dim=-1)
        r_a2m = [
            dist_a2m.unsqueeze(1),
            angle_a2m.unsqueeze(1),
            rel_head_a2m.unsqueeze(1),
        ]

        return {"r_t2m": r_t2m, "r_pl2m": r_pl2m, "r_a2m": r_a2m}

    def __call__(
        self, collate_data: dict, cur_step: None, his_model_input: bool = False
    ):
        deploy_inputs = OrderedDict()

        map_inputs = self.build_map_r_inputs(collate_data)
        if self.deploy or self.quant_infer_cold_start:
            if not his_model_input:
                if self.quant_infer_cold_start:
                    cur = cur_step  # quant infer and cold start
                else:
                    cur = self.ht  # quant infer and hot start
                agent_inputs = self.build_agent_cur_r_inputs(collate_data, cur)
                if not self.reuse_agent_rembs:
                    decoder_inputs = self.build_decoder_r_inputs(collate_data)
            else:  # quant infer and hot start
                qt = (
                    self.num_t2m_steps
                    + self.time_span * (self.num_agent_layers - 1)
                    - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht - 1, qt=qt
                )
        else:
            if self.stream:  # predict and hot start
                qt = (
                    self.num_t2m_steps
                    + self.time_span * (self.num_agent_layers - 1)
                    - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht - 1, qt=qt
                )
                agent_inputs2 = self.build_agent_cur_r_inputs(
                    collate_data, cur=self.ht
                )
                agent_inputs.update(agent_inputs2)
            else:
                qt = self.num_t2m_steps + self.time_span * (
                    self.num_agent_layers - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht, qt=qt
                )
            if not self.reuse_agent_rembs:
                decoder_inputs = self.build_decoder_r_inputs(collate_data)
        if not self.deploy:
            collate_data["map_point"]["magnitude"] /= 10.0
            collate_data["map_polygon"].update(map_inputs)
            collate_data["agent"].update(agent_inputs)
            if not self.reuse_agent_rembs:
                collate_data["decoder"].update(decoder_inputs)
            return collate_data
        elif self.deploy:
            A = self.A
            deploy_inputs["agent"] = OrderedDict()
            deploy_inputs["agent"]["valid_mask"] = collate_data["agent"][
                "valid_mask"
            ][:1, :A, : self.ht]
            deploy_inputs["agent"]["valid_mask_a2a"] = collate_data["agent"][
                "valid_mask_a2a"
            ][:1, :, :A, :A]
            deploy_inputs["agent"]["agent_type"] = collate_data["agent"][
                "agent_type"
            ][:1, :A]

            deploy_inputs["agent"].update(agent_inputs)

            deploy_inputs["map_polygon"] = {
                "pl_type": collate_data["map_polygon"]["pl_type"][:1],
                "is_intersection": collate_data["map_polygon"][
                    "is_intersection"
                ][:1],
            }
            deploy_inputs["map_polygon"].update(map_inputs)
            deploy_inputs["map_point"] = {
                "magnitude": collate_data["map_point"]["magnitude"][:1] / 10.0,
                "pt_type": collate_data["map_point"]["pt_type"][:1],
                "side": collate_data["map_point"]["side"][:1],
                "mask": collate_data["map_point"]["mask"][:1],
            }
            if not his_model_input:
                deploy_inputs["decoder"] = OrderedDict()
                deploy_inputs["decoder"].update(
                    {
                        "mask_a2m": collate_data["decoder"]["mask_a2m"][
                            :1, :A, :A
                        ],
                        "mask_dst": collate_data["decoder"]["mask_dst"][
                            :1, :A
                        ],
                    }
                )
                if not self.reuse_agent_rembs:
                    deploy_inputs["decoder"].update(decoder_inputs)
            deploy_inputs["type_pl2pl"] = collate_data["type_pl2pl"]

            return deploy_inputs


def collate_qc_argoverse2(
    batch: dict,
    ori_historical_sec=ori_historical_sec,
    ori_future_sec=ori_future_sec,
    ori_sample_fre=ori_sample_fre,
    sample_fre=sample_fre,
    stage: str = "train",
    pl_N: int = 80,
    pt_N: int = 50,
    pt_N_downsample_nums: int = 50,
    agent_num: int = 30,
    add_noise: bool = False,
    num_historical_steps: int = 50,
):
    """
    Collate function for QCNet with Argoverse2 dataset.

    This function preprocesses a batch of data, creating tensors for agents,
    map polygons, map points, and their relationships,
    to be used as input to the QCNet model.

    Args:
        batch: Batch of data containing agent and map information.
        ori_historical_sec: Original data historical sampling duration
        ori_future_sec: Original data future sampling duration (in seconds)
        ori_sample_fre: Original data sampling frequency
        sample_fre: Current sampling frequency
        stage: The stage of processing, e.g., 'train', 'val', 'test'.
        pl_N: Number of polygons for map polygon to agent cross attention.
        pt_N: Number of points for map polygon to agent cross attention.
        pt_N_downsample_nums: Number of map points to downsample from the.
        agent_num: Number of agents in the dataset.
        add_noise: Flag to add noise to the agent's position and velocity.
        num_historical_steps: Number of historical steps.

    Returns:
        dict: A dictionary containing processed agent and map data.
    """
    return_data = {
        "agent": {},
        "map_polygon": {},
        "map_point": {},
        "map_point_to_map_polygon": {},
        "map_polygon_to_map_polygon": {},
    }
    B = len(batch)
    input_dim = 2
    FS = sample_fre  # Downsampling Frequency
    HT = ori_sample_fre * ori_historical_sec  # Original Historical Input Steps
    samp_his_steps = ori_historical_sec * FS  # Time Steps of Historical Data
    his_samp_list = torch.arange(
        HT // samp_his_steps - 1, HT, HT // samp_his_steps
    )
    fu_samp_list = torch.arange(
        HT - 1 + HT // samp_his_steps,
        (ori_historical_sec + ori_future_sec) * ori_sample_fre,
        HT // samp_his_steps,
    )
    samp_list = torch.cat((his_samp_list, fu_samp_list))
    av_cur_pos = []
    for i in range(B):
        av_idx = batch[i]["agent"]["av_index"]
        av_cur = batch[i]["agent"]["position"][av_idx, HT - 1, :input_dim]
        av_cur_pos.append(av_cur)
    max_pl = max([b["map_polygon"]["num_nodes"] for b in batch])
    if pl_N is None or (stage == "train" and pl_N >= max_pl):
        pl_N = max_pl
        pl_idx = [
            torch.arange(b["map_polygon"]["num_nodes"]).long() for b in batch
        ]
    else:
        pl_idx = []
        for i in range(B):
            pl_pos = batch[i]["map_polygon"]["position"]
            if pl_pos.shape[0] > pl_N:
                pl_dist = torch.norm(pl_pos - av_cur_pos[i], p=2, dim=-1)
                if stage != "train":
                    _, pl_k = torch.topk(-pl_dist, k=pl_N, dim=-1)
                else:
                    k1 = 20
                    _, pl_k1 = torch.topk(-pl_dist, k=k1, dim=-1)
                    all_indices = torch.arange(pl_pos.shape[0])
                    remain_idx = all_indices[~torch.isin(all_indices, pl_k1)]
                    pl_k2 = remain_idx[
                        torch.randperm(len(remain_idx))[: pl_N - k1]
                    ]

                    pl_k = torch.cat([pl_k1, pl_k2])
            else:
                pl_k = torch.arange(pl_pos.shape[0])
            pl_idx.append(pl_k)
    if pt_N is None:
        pt_N = max(
            [
                max([p.shape[0] for p in b["map_point"]["position"]])
                for b in batch
            ]
        )

    mp = {}
    mp["position"] = torch.zeros([B, pl_N, pt_N, 2])
    mp["orientation"] = torch.zeros([B, pl_N, pt_N])
    mp["magnitude"] = torch.zeros([B, pl_N, pt_N])
    mp["height"] = torch.zeros([B, pl_N, pt_N])
    mp["pt_type"] = torch.zeros([B, pl_N, pt_N]).long()
    mp["side"] = torch.zeros([B, pl_N, pt_N]).long()
    mp["mask"] = torch.zeros([B, pl_N, pt_N]).bool()

    if pt_N_downsample_nums is not None and pt_N_downsample_nums < pt_N:
        step = pt_N / pt_N_downsample_nums
        pt_index = torch.arange(0, pt_N, step).long()[:pt_N_downsample_nums]
        mp_s = {}
        mp_s["position"] = torch.zeros([B, pl_N, pt_N_downsample_nums, 2])
        mp_s["orientation"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["magnitude"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["height"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["pt_type"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).long()
        mp_s["side"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).long()
        mp_s["mask"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).bool()

    for i, b in enumerate(batch):
        pl_num = b["map_polygon"]["num_nodes"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pt = b["map_point"]

        for j in range(pl_idx[i].shape[0]):
            if pl_idx is not None:
                pt_j = pl_idx[i][j]
            pt_num = len(pt["position"][pt_j])

            mp["position"][i, j, :pt_num, :] = (
                pt["position"][pt_j][:pt_N, :input_dim]
                - av_pos0[None, :input_dim]
            )
            mp["orientation"][i, j, :pt_num] = pt["orientation"][pt_j][:pt_N]
            mp["magnitude"][i, j, :pt_num] = pt["magnitude"][pt_j][:pt_N]
            mp["height"][i, j, :pt_num] = pt["height"][pt_j][:pt_N]
            mp["pt_type"][i, j, :pt_num] = pt["type"][pt_j][:pt_N]
            mp["side"][i, j, :pt_num] = pt["side"][pt_j][:pt_N]
            mp["mask"][i, j, :pt_num] = True

            if (
                pt_N_downsample_nums is not None
                and pt_N_downsample_nums < pt_N
            ):
                mp_s["position"][i, j, :pt_N_downsample_nums, :] = mp[
                    "position"
                ][i, j, pt_index, :]
                mp_s["orientation"][i, j, :pt_N_downsample_nums] = mp[
                    "orientation"
                ][i, j, pt_index]
                mp_s["magnitude"][i, j, :pt_N_downsample_nums] = mp[
                    "magnitude"
                ][i, j, pt_index]
                mp_s["height"][i, j, :pt_N_downsample_nums] = mp["height"][
                    i, j, pt_index
                ]
                mp_s["pt_type"][i, j, :pt_N_downsample_nums] = mp["pt_type"][
                    i, j, pt_index
                ]
                mp_s["side"][i, j, :pt_N_downsample_nums] = mp["side"][
                    i, j, pt_index
                ]
                mp_s["mask"][i, j, :pt_N_downsample_nums] = mp["mask"][
                    i, j, pt_index
                ]

    if pt_N_downsample_nums is not None and pt_N_downsample_nums < pt_N:
        return_data["map_point"] = mp_s
    else:
        return_data["map_point"] = mp

    mpl = {}
    mpl["position"] = torch.zeros([B, pl_N, 2])
    mpl["orientation"] = torch.zeros([B, pl_N])
    mpl["height"] = torch.zeros([B, pl_N])
    mpl["pl_type"] = torch.zeros([B, pl_N]).long()
    mpl["is_intersection"] = torch.zeros([B, pl_N]).long()
    mpl["valid_mask"] = torch.zeros([B, pl_N]).bool()
    for i, b in enumerate(batch):
        pl_num = b["map_polygon"]["num_nodes"]
        pl = b["map_polygon"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pl_pos = pl["position"][:, :input_dim] - av_pos0[None, :input_dim]
        mpl["position"][i, :pl_num, :] = pl_pos[pl_idx[i]]
        mpl["orientation"][i, :pl_num] = pl["orientation"][pl_idx[i]]
        mpl["height"][i, :pl_num] = pl["height"][pl_idx[i]]
        mpl["pl_type"][i, :pl_num] = pl["type"][pl_idx[i]]
        mpl["is_intersection"][i, :pl_num] = pl["is_intersection"][pl_idx[i]]
        mpl["valid_mask"][i, :pl_num] = True

    return_data["map_polygon"] = mpl
    type_pl2pl = torch.zeros(B, pl_N, pl_N).long()
    for i, b in enumerate(batch):
        edge_pl = b["map_polygon_to_map_polygon"]["edge_index"]
        pl_type = b["map_polygon_to_map_polygon"]["type"]

        for j in range(min(pl_N, edge_pl.shape[1])):
            if edge_pl[1, j] < pl_N and edge_pl[0, j] < pl_N:
                type_pl2pl[i, edge_pl[1, j], edge_pl[0, j]] = pl_type[j]

    return_data["type_pl2pl"] = type_pl2pl
    max_A = max([b["agent"]["num_nodes"] for b in batch])
    if agent_num is None:
        agent_num = max_A
    if stage == "train":
        agent_num = min(agent_num, max_A, 80)
    if agent_num is None or (stage == "train" and agent_num >= max_A):
        A = max_A
        a_idx = [torch.arange(b["agent"]["num_nodes"]).long() for b in batch]
    else:
        A = agent_num
        a_idx = []
        for i in range(B):
            agent_pos = batch[i]["agent"]["position"][:, HT - 1, :input_dim]
            if agent_pos.shape[0] > A:
                a_dist = torch.norm(agent_pos - av_cur_pos[i], p=2, dim=-1)
                if stage != "train":
                    _, a_k = torch.topk(-a_dist, k=A, dim=-1)
                else:
                    k1 = 10
                    _, a_k1 = torch.topk(-a_dist, k=k1, dim=-1)
                    all_indices = torch.arange(agent_pos.shape[0])
                    remain_idx = all_indices[~torch.isin(all_indices, a_k1)]
                    a_k2 = remain_idx[
                        torch.randperm(len(remain_idx))[: A - k1]
                    ]

                    a_k = torch.cat([a_k1, a_k2])
            else:
                a_k = torch.arange(agent_pos.shape[0])
            focal_idx = (batch[i]["agent"]["category"] == 3).nonzero()

            if focal_idx[0][0] not in a_k:
                a_k = torch.cat([a_k[:-1], focal_idx[0]], dim=-1)
            assert focal_idx[0][0] in a_k
            a_idx.append(a_k)
    # Total Steps of Historical and Forecast Data After Downsampling
    T = batch[0]["agent"]["valid_mask"].shape[1] // ori_sample_fre * FS
    agent = {}
    agent["num_nodes"] = [b["agent"]["num_nodes"] for b in batch]
    agent["valid_mask"] = torch.zeros([B, A, T], dtype=torch.bool)
    agent["predict_mask"] = torch.zeros([B, A, T], dtype=torch.bool)
    agent["agent_type"] = torch.zeros([B, A, 1], dtype=torch.long)
    agent["position"] = torch.zeros([B, A, T, 2], dtype=torch.float)
    agent["heading"] = torch.zeros([B, A, T], dtype=torch.float)
    agent["velocity"] = torch.zeros([B, A, T, 2], dtype=torch.float)
    agent["category"] = torch.zeros([B, A, 1], dtype=torch.long)

    for i, _b in enumerate(batch):
        a = batch[i]["agent"]["num_nodes"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pos = (
            batch[i]["agent"]["position"][a_idx[i], :, :2] - av_pos0[None, :2]
        )
        pos = pos[:, samp_list, :]
        valid_mask = batch[i]["agent"]["valid_mask"][a_idx[i]]
        valid_mask = valid_mask[:, samp_list]
        vel = batch[i]["agent"]["velocity"][a_idx[i], :, :2]
        vel = vel[:, samp_list, :]
        if add_noise:
            noise1 = (
                torch.randn((a, samp_his_steps, 2)).to(valid_mask.device)
                * 0.01
            )
            noise2 = (
                torch.randn((a, samp_his_steps, 2)).to(valid_mask.device)
                * 0.002
            )
            pos[:, :samp_his_steps] = pos[:, :samp_his_steps] + noise1
            vel[:, :samp_his_steps] = vel[:, :samp_his_steps] + noise2
        pos = torch.where(valid_mask.unsqueeze(-1), pos, 0)
        agent["valid_mask"][i, :a] = valid_mask
        predict_mask = batch[i]["agent"]["predict_mask"][a_idx[i]]
        agent["predict_mask"][i, :a] = predict_mask[:, samp_list]
        agent["agent_type"][i, :a, 0] = batch[i]["agent"]["type"][a_idx[i]]

        agent["position"][i, :a] = pos
        heading = batch[i]["agent"]["heading"][a_idx[i]]
        agent["heading"][i, :a] = heading[:, samp_list]
        agent["velocity"][i, :a] = vel
        agent["category"][i, :a, 0] = batch[i]["agent"]["category"][a_idx[i]]

    mask_dst = agent["predict_mask"].any(dim=-1, keepdim=True)

    mask_src = agent["valid_mask"][:, :, :samp_his_steps]
    mask_ta = agent["valid_mask"].transpose(1, 2)[
        :, :samp_his_steps
    ]  # [B, T, A]
    agent["valid_mask_a2a"] = mask_ta[:, :, :, None] & mask_ta[:, :, None, :]
    return_data["agent"] = agent
    decoder = {}
    decoder["mask_dst"] = mask_dst  # [B, A, 1]
    decoder["mask_a2m"] = mask_dst[:, :, :] & mask_src[:, None, :, -1]
    return_data["decoder"] = decoder
    return return_data


def parse_args():
    parser = argparse.ArgumentParser(description="Pack argoverse2 dataset.")
    parser.add_argument(
        "-s",
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "-t",
        "--target-data-dir",
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="val", help="The mode to packed."
    )
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--num-samples",
        default=24988,
        type=int,
        help="The directory that contains unpacked image files.",
    )
    args = parser.parse_args()
    return args


def data_collate_preprocess(idx, target_data_dir, args):

    # get ori data
    packer = Argoverse2Packer(
        args.src_data_dir,
        target_data_dir,
        split_name=args.split_name,
        pack_type=args.pack_type,
        num_workers=args.num_workers,
    )
    data_ori = []
    tmp_data, sample_id = packer(idx)
    data_ori.append(tmp_data)  # {torch.Tensor}

    # collate data and downsample
    # ['agent', 'map_polygon', 'map_point', 'map_point_to_map_polygon', 'map_polygon_to_map_polygon', 'type_pl2pl', 'decoder']  # noqa
    # 'agent'：['num_nodes', 'valid_mask', 'predict_mask', 'agent_type', 'position', 'heading', 'velocity', 'category', 'valid_mask_a2a'] # noqa
    # 'map_polygon'：['position', 'orientation', 'height', 'pl_type', 'is_intersection', 'valid_mask']  # noqa
    # 'map_point'：['position', 'orientation', 'magnitude', 'height', 'pt_type', 'side', 'mask']    # noqa
    # 'map_point_to_map_polygon'：none
    # 'map_polygon_to_map_polygon'：none
    # 'type_pl2pl'：tensor
    # 'decoder'：['mask_dst', 'mask_a2m']
    data_ori_collate = collate_qc_argoverse2(
        batch=data_ori,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        pl_N=80,
        pt_N=50,
        pt_N_downsample_nums=50,
        agent_num=30,
        add_noise=False,
        num_historical_steps=50,
    )

    # data_for_postprocess
    data_for_postprocess = OrderedDict()
    data_for_postprocess["agent_position"] = data_ori_collate["agent"][
        "position"
    ]  # 35 (1 30 22 2) fpew
    data_for_postprocess["agent_heading"] = data_ori_collate["agent"][
        "heading"
    ]  # 36 (1 30 22) fp32
    data_for_postprocess["agent_category"] = data_ori_collate["agent"][
        "category"
    ]  # 37 (1 30 1) int 64
    data_for_postprocess["agent_predict_mask"] = data_ori_collate["agent"][
        "predict_mask"
    ]  # 38 (1 30, 22) torch.bool

    # cold start preprocess
    preprocess = QCNetOEPreprocess(
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        agent_num=A,
        pl_num=pl,
        pt_num=pt,
        stream=True,
        deploy=True,
        quant_infer_cold_start=True,
    )

    cur_step = (
        num_historical_steps - num_t2m_steps - time_span
    )  # current steps of stream infer (init 2)
    x_a_decoder_zeros = torch.zeros(
        B, A, num_t2m_steps - 1, D
    )  # x_a for decoder init zeros

    temporal_keys = ["x_a_cur", "r_pl2a_cur", "r_t_cur", "r_a2a_cur"]
    temporal_mask = ["mask_a_cur", "mask_a2a_cur", "mask_t_key"]
    temp_inputs = {}
    for key in temporal_keys:
        temp_inputs[key] = []
    for key in temporal_mask:
        temp_inputs[key] = []

    for _ in range(num_t2m_steps + time_span):
        cur_step += 1
        cur_inputs = preprocess(
            data_ori_collate, cur_step, his_model_input=False
        )
        cur_inputs["agent"]["x_a_mid_emb"] = torch.zeros(
            B, A, time_span, D
        )  # [1, 30, 2, 128]
        cur_inputs["agent"]["x_a_his"] = x_a_decoder_zeros  # [1, 30, 5, 128]

        for key in temporal_keys:
            temp_inputs[key].append(cur_inputs["agent"][key])
        for key in temporal_mask:
            temp_inputs[key].append(cur_inputs["agent"][key])

    for key in temporal_keys:

        dist_motion = torch.cat(
            [
                temp_inputs[key][j][0].unsqueeze(0)
                for j in range(len(temp_inputs[key]))
            ],
            dim=0,
        )
        angle_motion = torch.cat(
            [
                temp_inputs[key][j][1].unsqueeze(0)
                for j in range(len(temp_inputs[key]))
            ],
            dim=0,
        )

        dist_vel = torch.cat(
            [
                temp_inputs[key][j][2].unsqueeze(0)
                for j in range(len(temp_inputs[key]))
            ],
            dim=0,
        )
        if len(temp_inputs[key][0]) == 4:
            angle_vel = torch.cat(
                [
                    temp_inputs[key][j][3].unsqueeze(0)
                    for j in range(len(temp_inputs[key]))
                ],
                dim=0,
            )
            cur_inputs["agent"][key] = [
                dist_motion,
                angle_motion,
                dist_vel,
                angle_vel,
            ]  # "x_a_cur": (8, B, 1, A, 1) * 4  "r_t_cur": (8, B, 1, A, num_t2m_steps) * 4 # noqa
        else:
            cur_inputs["agent"][key] = [
                dist_motion,
                angle_motion,
                dist_vel,
            ]  # "r_pl2a_cur": (8, B, 1, A, pl) * 3  "r_a2a_cur": (8, B, 1, A, A) * 3   # noqa

    for key in temporal_mask:
        cur_inputs["agent"][key] = torch.cat(
            [
                temp_inputs[key][j].unsqueeze(0)
                for j in range(len(temp_inputs[key]))
            ],
            dim=0,
        )  # "mask_a_cur": [8, 1, 30], "mask_a2a_cur":[8, 1, 30, 30], "mask_t_key":[8, 1, 30, 2]    # noqa

    # data after preprocess
    # ['agent', 'map_polygon', 'map_point', 'decoder', 'type_pl2pl']    # noqa
    # 'agent'：['valid_mask', 'valid_mask_a2a', 'agent_type', 'x_a_cur', 'r_pl2a_cur', 'r_t_cur', 'r_a2a_cur',"mask_a_cur", "mask_a2a_cur", "mask_t_key", 'x_a_mid_emb', 'x_a_his'] # noqa
    # 'map_polygon'：['pl_type', 'is_intersection', 'r_pl2pl', 'r_pt2pl', 'mask_pl2pl'] # noqa
    # 'map_point'：['magnitude', 'pt_type', 'side', 'mask'] # noqa
    # 'decoder'：['mask_a2m', 'mask_dst']   # noqa
    # 'type_pl2pl'：tensor

    cur_inputs.update(data_for_postprocess)  # 35 + 4 = 39

    get_bin_input(cur_inputs, sample_id, idx, target_data_dir)


if __name__ == "__main__":
    args = parse_args()

    target_data_dir = args.target_data_dir

    num_samples = args.num_samples

    start_time = time.time()

    for idx in tqdm(range(num_samples), dynamic_ncols=True):
        # for idx in trange(num_samples):
        data_collate_preprocess(idx, target_data_dir, args)
    # pbar.set_description(f'Step-{simstep}: ')

    #  python3 qcnet_argoverse2_preprocess.py --src-data-dir  ../tmp_ori_data/argoverse-2  --target-data-dir qcnet_val_data_bin_1105 --pack-type lmdb --num-samples 10  # noqa
