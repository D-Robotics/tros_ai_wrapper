// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_centerpoint_post_process_method.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/box3d_nms.h"
#include "utils/tensor_utils.h"

DEFINE_AND_REGISTER_METHOD(QATCenterPointPostProcessMethod);

void HeatmapDequantizeScale(int *valid_shape, int *aligned_shape, float *scale,
                            int topk, int32_t *input,
                            std::vector<ScoresData> &output) {
  int kWC = aligned_shape[2] * aligned_shape[3];
  int kHW = valid_shape[1] * valid_shape[2];
  for (int c = 0; c < valid_shape[3]; ++c) {
    std::priority_queue<ScoresData, std::vector<ScoresData>,
                        std::greater<ScoresData>>
        queue;
    for (int h = 0; h < valid_shape[1]; ++h) {
      for (int w = 0; w < valid_shape[2]; w++) {
        float data_tmp =
            quanti_scale(input[h * kWC + w * aligned_shape[3] + c], scale[c]);
        float value = 1.0f / (std::exp(-data_tmp) + 1.0f);
        queue.push(ScoresData(value, c, w, h));
        if (queue.size() > topk) {
          queue.pop();
        }
      }
    }
    while (!queue.empty()) {
      output.emplace_back(queue.top());
      queue.pop();
    }
  }
}

std::vector<std::vector<float>> xywhr2xyxyr(std::vector<Lidar3D> &Bboxes) {
  std::vector<std::vector<float>> boxes(Bboxes.size(), std::vector<float>(5));
  for (int i = 0; i < Bboxes.size(); i++) {
    boxes[i][0] = Bboxes[i].xs - Bboxes[i].dim_1 * 0.5f;
    boxes[i][1] = Bboxes[i].ys - Bboxes[i].dim_0 * 0.5f;
    boxes[i][2] = Bboxes[i].xs + Bboxes[i].dim_1 * 0.5f;
    boxes[i][3] = Bboxes[i].ys + Bboxes[i].dim_0 * 0.5f;
    boxes[i][4] = 0.0f - Bboxes[i].rot - 3.141592653589793 / 2;
  }
  return boxes;
}

int QATCenterPointPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATCenterPointPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("norm_bbox")) {
    norm_bbox_ = document["norm_bbox"].GetBool();
  }

  if (document.HasMember("topk")) {
    topk_ = document["topk"].GetInt();
  }

  if (document.HasMember("out_size_factor")) {
    out_size_factor_ = document["out_size_factor"].GetInt();
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("nms_thr")) {
    nms_thr_ = document["nms_thr"].GetFloat();
  }

  if (document.HasMember("pre_max_size")) {
    pre_max_size_ = document["pre_max_size"].GetInt();
  }

  if (document.HasMember("post_max_size")) {
    post_max_size_ = document["post_max_size"].GetInt();
  }

  if (document.HasMember("pc_range")) {
    auto pc_range_value = document["pc_range"].GetArray();
    pc_range_.resize(pc_range_value.Size());
    for (int i = 0; i < pc_range_value.Size(); i++) {
      pc_range_[i] = pc_range_value[i].GetFloat();
    }
  }

  if (document.HasMember("voxel_size")) {
    auto voxel_size_value = document["voxel_size"].GetArray();
    voxel_size_.resize(voxel_size_value.Size());
    for (int i = 0; i < voxel_size_value.Size(); i++) {
      voxel_size_[i] = voxel_size_value[i].GetFloat();
    }
  }

  if (document.HasMember("post_center_range")) {
    auto post_center_range_value = document["post_center_range"].GetArray();
    post_center_range_.resize(post_center_range_value.Size());
    for (int i = 0; i < post_center_range_value.Size(); i++) {
      post_center_range_[i] = post_center_range_value[i].GetFloat();
    }
  }

  return 0;
}

PerceptionPtr QATCenterPointPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATCenterPointPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
    Perception *perception) {
  perception->type = Perception::LIDAR3D;
  for (int i = 0; i < tensors.size(); i++) {
    hbUCPMemFlush(&(tensors[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
  }

  int kHW = width_ * height_;
  int task_cls{0};

  for (int i = 0; i < 6; i++) {
    // heatmap: score, cls, id
    std::vector<ScoresData> heatmaps;
    int32_t *heatmap_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 5].sysMem.virAddr);
    float *heatmap_scale = tensors[i * 6 + 5].properties.scale.scaleData;
    int32_t *heatmap_shape =
        tensors[i * 6 + 5].properties.validShape.dimensionSize;
    auto heatmap_aligned_shape =
        properies2alignshape(tensors[i * 6 + 5].properties);
    int32_t num_cls = heatmap_shape[3];
    HeatmapDequantizeScale(heatmap_shape, heatmap_aligned_shape.data(),
                           heatmap_scale, topk_, heatmap_data, heatmaps);

    std::priority_queue<ScoresData, std::vector<ScoresData>,
                        std::greater<ScoresData>>
        queue;

    for (auto &heat : heatmaps) {
      queue.push(heat);
      if (queue.size() > topk_) {
        queue.pop();
      }
    }

    std::vector<ScoresData> heatmap;
    while (!queue.empty()) {
      if (queue.top().value > score_threshold_) {
        heatmap.emplace_back(queue.top());
      }
      queue.pop();
    }
    std::reverse(heatmap.begin(), heatmap.end());

    int32_t score_size = heatmap.size();

    auto stride2alignshape = [](const hbDNNTensorProperties &properties,
                                int32_t index) {
      return properties.stride[index - 1] / properties.stride[index];
    };
    // rot
    int32_t *rot_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 3].sysMem.virAddr);
    float *rot_scale = tensors[i * 6 + 3].properties.scale.scaleData;
    int32_t rot_aligned_c = stride2alignshape(tensors[i * 6 + 3].properties, 3);

    // height
    int32_t *height_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 1].sysMem.virAddr);
    float *height_scale = tensors[i * 6 + 1].properties.scale.scaleData;
    int32_t height_aligned_c =
        stride2alignshape(tensors[i * 6 + 1].properties, 3);

    // dim
    int32_t *dim_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 2].sysMem.virAddr);
    float *dim_scale = tensors[i * 6 + 2].properties.scale.scaleData;
    int32_t dim_aligned_c = stride2alignshape(tensors[i * 6 + 2].properties, 3);

    // vel
    int32_t *vel_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 4].sysMem.virAddr);
    float *vel_scale = tensors[i * 6 + 4].properties.scale.scaleData;
    int32_t vel_aligned_c = stride2alignshape(tensors[i * 6 + 4].properties, 3);

    // reg
    int32_t *reg_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 0].sysMem.virAddr);
    float *reg_scale = tensors[i * 6 + 0].properties.scale.scaleData;
    int32_t reg_aligned_c = stride2alignshape(tensors[i * 6 + 0].properties, 3);

    std::vector<float> height;
    std::vector<float> xs;
    std::vector<float> ys;
    float xs_weight = out_size_factor_ * voxel_size_[0];
    float ys_weight = out_size_factor_ * voxel_size_[1];
    int32_t reg_aligned_WC = width_ * reg_aligned_c;
    int32_t height_aligned_WC = width_ * height_aligned_c;
    for (int k = 0; k < score_size; k++) {
      int32_t w = heatmap[k].w;
      int32_t h = heatmap[k].h;
      int32_t reg_idx_0 = h * reg_aligned_WC + w * reg_aligned_c + 0;
      int32_t reg_idx_1 = h * reg_aligned_WC + w * reg_aligned_c + 1;
      int32_t height_idx = h * height_aligned_WC + w * height_aligned_c + 0;

      float reg_0 = quanti_scale(reg_data[reg_idx_0], reg_scale[0]);
      float reg_1 = quanti_scale(reg_data[reg_idx_1], reg_scale[1]);
      height.emplace_back(
          quanti_scale(height_data[height_idx], height_scale[0]));
      xs.emplace_back((reg_0 + heatmap[k].w) * xs_weight + pc_range_[0]);
      ys.emplace_back((reg_1 + heatmap[k].h) * ys_weight + pc_range_[1]);
    }

    // mask
    std::vector<int32_t> indexes;
    std::vector<Lidar3D> Bbox;
    std::vector<float> scores;
    std::vector<int32_t> cls;
    for (int32_t k = 0; k < score_size; k++) {
      if (xs[k] >= post_center_range_[0] && ys[k] >= post_center_range_[1] &&
          height[k] >= post_center_range_[2] &&
          xs[k] <= post_center_range_[3] && ys[k] <= post_center_range_[4] &&
          height[k] <= post_center_range_[5]) {
        indexes.emplace_back(k);
      }
    }

    Bbox.resize(indexes.size());
    scores.resize(indexes.size());
    cls.resize(indexes.size());

    int32_t rot_aligned_WC = width_ * rot_aligned_c;
    int32_t dim_aligned_WC = width_ * dim_aligned_c;
    int32_t vel_aligned_WC = width_ * vel_aligned_c;

    for (int32_t j = 0; j < indexes.size(); j++) {
      int32_t k = indexes[j];
      int32_t w = heatmap[k].w;
      int32_t h = heatmap[k].h;
      int32_t rot_idx_0 = h * rot_aligned_WC + w * rot_aligned_c + 0;
      int32_t rot_idx_1 = h * rot_aligned_WC + w * rot_aligned_c + 1;
      float rots = quanti_scale(rot_data[rot_idx_0], rot_scale[0]);
      float rotc = quanti_scale(rot_data[rot_idx_1], rot_scale[1]);

      Bbox[j].xs = xs[k];
      Bbox[j].ys = ys[k];
      Bbox[j].height = height[k];

      int32_t dim_idx_0 = h * dim_aligned_WC + w * dim_aligned_c + 0;
      int32_t dim_idx_1 = h * dim_aligned_WC + w * dim_aligned_c + 1;
      int32_t dim_idx_2 = h * dim_aligned_WC + w * dim_aligned_c + 2;
      float dim_0_tmp = quanti_scale(dim_data[dim_idx_0], dim_scale[0]);
      float dim_1_tmp = quanti_scale(dim_data[dim_idx_1], dim_scale[1]);
      float dim_2_tmp = quanti_scale(dim_data[dim_idx_2], dim_scale[2]);
      Bbox[j].dim_0 = norm_bbox_ ? std::exp(dim_0_tmp) : dim_0_tmp;
      Bbox[j].dim_1 = norm_bbox_ ? std::exp(dim_1_tmp) : dim_1_tmp;
      Bbox[j].dim_2 = norm_bbox_ ? std::exp(dim_2_tmp) : dim_2_tmp;

      Bbox[j].rot = std::atan2(rots, rotc);
      int32_t vel_idx_0 = h * vel_aligned_WC + w * vel_aligned_c + 0;
      int32_t vel_idx_1 = h * vel_aligned_WC + w * vel_aligned_c + 1;
      Bbox[j].vel_0 = quanti_scale(vel_data[vel_idx_0], vel_scale[0]);
      Bbox[j].vel_1 = quanti_scale(vel_data[vel_idx_1], vel_scale[1]);

      scores[j] = heatmap[k].value;
      cls[j] = heatmap[k].c;
    }

    std::vector<int32_t> selected_idxs;

    if (Bbox.size() > 0) {
      std::vector<std::vector<float>> bboxes_for_nms = xywhr2xyxyr(Bbox);
      Lidar3DNMS(selected_idxs, bboxes_for_nms, scores, nms_thr_, pre_max_size_,
                 post_max_size_);

      if (selected_idxs.size() > 0) {
        for (int32_t s = 0; s < selected_idxs.size(); s++) {
          int k = selected_idxs[s];
          perception->lidar3d.emplace_back(
              LidarDetection3D{Bbox[k], scores[k], cls[k] + task_cls});
        }
      }
    }
    task_cls += num_cls;
  }

  if (perception->lidar3d.size() == 0) {
    perception->lidar3d.resize(6, LidarDetection3D{Lidar3D(), 0.0f, 0});
  }
  return 0;
}
