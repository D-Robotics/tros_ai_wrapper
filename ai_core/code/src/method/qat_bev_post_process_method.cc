// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bev_post_process_method.h"

#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevPostProcessMethod);

std::vector<std::vector<std::string>> DetTask = {
    {"car"},
    {"truck", "construction_vehicle"},
    {"bus", "trailer"},
    {"barrier"},
    {"motorcycle", "bicycle"},
    {"pedestrian", "traffic_cone"}};

std::unordered_map<std::string, int32_t> ClsMap = {{"car", 0},
                                                   {"truck", 1},
                                                   {"trailer", 2},
                                                   {"bus", 3},
                                                   {"construction_vehicle", 4},
                                                   {"bicycle", 5},
                                                   {"motorcycle", 6},
                                                   {"pedestrian", 7},
                                                   {"traffic_cone", 8},
                                                   {"barrier", 9}};

static void HeatMaskAndTopk(int32_t *ori_heat, int32_t *pool_heat, float *scale,
                            int32_t *valid_shape, int32_t topk,
                            std::vector<ScoresData> &output) {
  int32_t kHW = valid_shape[2] * valid_shape[3];
  output.reserve(valid_shape[1] * topk);
  for (int32_t c = 0; c < valid_shape[1]; c++) {
    std::vector<ScoresData> output_c;
    for (int32_t h = 0; h < valid_shape[2]; h++) {
      for (int32_t w = 0; w < valid_shape[3]; w++) {
        int32_t curr_pos = c * kHW + h * valid_shape[3] + w;
        if (ori_heat[curr_pos] == pool_heat[curr_pos]) {
          float data_tmp = quanti_scale(ori_heat[curr_pos], scale[c]);
          float value = 1.0f / (std::exp(-data_tmp) + 1.0f);
          output_c.emplace_back(ScoresData(value, c, w, h));
        }
      }
    }
    std::stable_sort(output_c.begin(), output_c.end(), CompareScoresData());
    std::move(output_c.begin(), output_c.begin() + topk,
              std::back_inserter(output));
  }
}

int QATBevPostProcessMethod::InitFromJsonString(const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevPostProcessMethod Json string:"
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

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("max_pool_kernel")) {
    kernel_ = document["max_pool_kernel"].GetInt();
  }

  if (document.HasMember("bev_size")) {
    auto bev_size_value = document["bev_size"].GetArray();
    bev_size_.resize(bev_size_value.Size());
    for (int i = 0; i < bev_size_value.Size(); i++) {
      bev_size_[i] = bev_size_value[i].GetFloat();
    }
  }

  if (document.HasMember("ori_shape")) {
    auto ori_value = document["ori_shape"].GetArray();
    ori_shape_.resize(ori_value.Size());
    for (int i = 0; i < ori_value.Size(); i++) {
      ori_shape_[i] = ori_value[i].GetInt();
    }
  }

  if (document.HasMember("resize_shape")) {
    auto resize_value = document["resize_shape"].GetArray();
    resize_shape_.resize(resize_value.Size());
    for (int i = 0; i < resize_value.Size(); i++) {
      resize_shape_[i] = resize_value[i].GetInt();
    }
  }

  return 0;
}

PerceptionPtr QATBevPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevPostProcessMethod::PostProcess(std::vector<hbDNNTensor> &tensors,
                                         ImageTensor *image_tensor,
                                         Perception *perception) {
  perception->type = Perception::BEV;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];

  VLOG(EXAMPLE_DEBUG) << "ori_image_height: " << image_tensor->ori_image_height;
  VLOG(EXAMPLE_DEBUG) << "ori_image_width: " << image_tensor->ori_image_width;
  VLOG(EXAMPLE_DEBUG) << "resize_height: " << image_tensor->resize_height;
  VLOG(EXAMPLE_DEBUG) << "resize_width: " << image_tensor->resize_width;

  // for ipm_4d
  bool is_temporal = WorkflowPlugin::GetInstance()->IsTemporalModel();
  VLOG(EXAMPLE_DEBUG) << "Is temporal model: " << is_temporal
                      << "; output tensor count:" << tensors.size();
  if (is_temporal && tensors.size() == 38) {
    VLOG(EXAMPLE_DEBUG) << "Frame id: " << image_tensor->frame_id;
    int32_t associated_prev_num =
        WorkflowPlugin::GetInstance()->GetAssociatedPrevNum();

    hbDNNTensor &temporal_fusion = tensors[37];
    WorkflowPlugin::GetInstance()->UpdateTemporalTensor<int8_t *>(
        &temporal_fusion, associated_prev_num - 1, 0);
  }

  // segment
  int32_t *seg_data = reinterpret_cast<int32_t *>(tensors[0].sysMem.virAddr);
  int32_t seg_channel = tensors[0].properties.validShape.dimensionSize[1];
  int32_t seg_height = tensors[0].properties.validShape.dimensionSize[2];
  int32_t seg_width = tensors[0].properties.validShape.dimensionSize[3];
  float *seg_scale = tensors[0].properties.scale.scaleData;

  int32_t seg_height_aligned =
      tensors[0].properties.stride[1] / tensors[0].properties.stride[2];
  int32_t seg_width_aligned =
      tensors[0].properties.stride[2] / tensors[0].properties.stride[3];

  int32_t seg_kHW = seg_height * seg_width;
  perception->bevSeg.num_classes = seg_channel;
  perception->bevSeg.seg.resize(seg_kHW);
  perception->bevSeg.height = seg_height;
  perception->bevSeg.width = seg_width;

  int32_t hw_aligned = seg_height_aligned * seg_width_aligned;

  for (int32_t h = 0; h < seg_height; h++) {
    int32_t h_idx = h * seg_width_aligned;
    int32_t h_k = h * seg_width;
    for (int32_t w = 0; w < seg_width; w++) {
      int32_t idx = h_idx + w;
      int32_t k = h_k + w;
      int32_t *data_tmp = seg_data + idx;
      float max_val = std::numeric_limits<float>::lowest();
      int32_t top_index = -1;
      for (int c = 0; c < seg_channel; c++) {
        float data = quanti_scale(data_tmp[c * hw_aligned], seg_scale[c]);
        if (data > max_val) {
          max_val = data;
          top_index = c;
        }
      }
      perception->bevSeg.seg[k] = top_index;
    }
  }

  // detection
  int32_t pad_val = std::floor((kernel_ - 1) / 2);
  std::vector<int32_t> const max_pool_kernel{kernel_, kernel_};
  std::vector<int32_t> const pad{pad_val, pad_val};
  std::vector<int32_t> const stride{1, 1};

  int kHW = width_ * height_;

  for (int i = 0; i < 6; i++) {
    int32_t *heatmap_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 6].sysMem.virAddr);
    float *heatmap_scale = tensors[i * 6 + 6].properties.scale.scaleData;
    int32_t *heatmap_shape =
        tensors[i * 6 + 6].properties.validShape.dimensionSize;
    int32_t num_cls = heatmap_shape[1];
    int32_t heatmap_size =
        heatmap_shape[1] * heatmap_shape[2] * heatmap_shape[3];

    std::vector<int32_t> pool_heat(heatmap_size);
    max_pool2d_nchw(pool_heat.data(), heatmap_data, heatmap_shape,
                    max_pool_kernel, stride, pad);

    std::vector<ScoresData> heatmaps_topk;
    HeatMaskAndTopk(heatmap_data, pool_heat.data(), heatmap_scale,
                    heatmap_shape, topk_, heatmaps_topk);

    std::vector<ScoresData> heatmap;
    std::stable_sort(heatmaps_topk.begin(), heatmaps_topk.end(),
                     CompareScoresData());
    for (int32_t idx = 0; idx < topk_; idx++) {
      if (heatmaps_topk[idx].value > score_threshold_) {
        heatmap.emplace_back(heatmaps_topk[idx]);
      }
    }

    // rot
    int32_t *rot_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 4].sysMem.virAddr);
    float *rot_scale = tensors[i * 6 + 4].properties.scale.scaleData;
    // height
    int32_t *height_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 2].sysMem.virAddr);
    float *height_scale = tensors[i * 6 + 2].properties.scale.scaleData;
    // dim
    int32_t *dim_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 3].sysMem.virAddr);
    float *dim_scale = tensors[i * 6 + 3].properties.scale.scaleData;
    // vel
    int32_t *vel_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 5].sysMem.virAddr);
    float *vel_scale = tensors[i * 6 + 5].properties.scale.scaleData;
    // reg
    int32_t *reg_data =
        reinterpret_cast<int32_t *>(tensors[i * 6 + 1].sysMem.virAddr);
    float *reg_scale = tensors[i * 6 + 1].properties.scale.scaleData;

    std::vector<float> xs;
    std::vector<float> ys;

    int32_t score_size = heatmap.size();
    for (int k = 0; k < score_size; k++) {
      int32_t w = heatmap[k].w;
      int32_t h = heatmap[k].h;
      int32_t idx_0 = 0 * kHW + h * width_ + w;
      int32_t idx_1 = 1 * kHW + h * width_ + w;

      float reg_0 = quanti_scale(reg_data[idx_0], reg_scale[0]);
      float reg_1 = quanti_scale(reg_data[idx_1], reg_scale[1]);

      xs.emplace_back((reg_0 + heatmap[k].w) * out_size_factor_);
      ys.emplace_back((reg_1 + heatmap[k].h) * out_size_factor_);
    }

    // grid mask
    std::vector<int32_t> indexes;
    float grid_size_0 = bev_size_[0] * 2 / bev_size_[2];
    float grid_size_1 = bev_size_[1] * 2 / bev_size_[2];
    if (score_size > pre_max_size_) {
      score_size = pre_max_size_;
    }

    if (score_size > post_max_size_) {
      score_size = post_max_size_;
    }

    for (int32_t k = 0; k < score_size; k++) {
      if (xs[k] > 0 && ys[k] > 0 && xs[k] < grid_size_1 &&
          ys[k] < grid_size_0) {
        indexes.emplace_back(k);
      }
    }

    float max_x = bev_size_[1] - bev_size_[2] / 2;
    float max_y = bev_size_[0] - bev_size_[2] / 2;

    std::vector<EgoBbox> Bbox;
    std::vector<float> scores;
    std::vector<int32_t> cls;

    Bbox.resize(indexes.size());
    scores.resize(indexes.size());
    cls.resize(indexes.size());

    for (int32_t j = 0; j < indexes.size(); j++) {
      int32_t k = indexes[j];
      int32_t w = heatmap[k].w;
      int32_t h = heatmap[k].h;
      int32_t idx_0 = 0 * kHW + h * width_ + w;
      int32_t idx_1 = 1 * kHW + h * width_ + w;
      int32_t idx_2 = 2 * kHW + h * width_ + w;

      float rots = quanti_scale(rot_data[idx_0], rot_scale[0]);
      float rotc = quanti_scale(rot_data[idx_1], rot_scale[1]);

      Bbox[j].xs = xs[k] * bev_size_[2] - max_x;
      Bbox[j].ys = ys[k] * bev_size_[2] - max_y;

      float dim_0_tmp = quanti_scale(dim_data[idx_0], dim_scale[0]);
      float dim_1_tmp = quanti_scale(dim_data[idx_1], dim_scale[1]);
      float dim_2_tmp = quanti_scale(dim_data[idx_2], dim_scale[2]);
      float dim_0 = norm_bbox_ ? std::exp(dim_0_tmp) : dim_0_tmp;
      float dim_1 = norm_bbox_ ? std::exp(dim_1_tmp) : dim_1_tmp;
      float dim_2 = norm_bbox_ ? std::exp(dim_2_tmp) : dim_2_tmp;
      Bbox[j].dim_0 = dim_0 * bev_size_[2];
      Bbox[j].dim_1 = dim_1 * bev_size_[2];
      Bbox[j].dim_2 = dim_2 * bev_size_[2];

      Bbox[j].rot = std::atan2(rots, rotc);
      Bbox[j].vel_0 =
          quanti_scale(vel_data[idx_0], vel_scale[0]) * bev_size_[2];
      Bbox[j].vel_1 =
          quanti_scale(vel_data[idx_1], vel_scale[1]) * bev_size_[2];
      Bbox[j].height =
          quanti_scale(height_data[idx_0], height_scale[0]) * bev_size_[2];

      scores[j] = heatmap[k].value;
      cls[j] = ClsMap[DetTask[i][heatmap[k].c]];

      // printf("index: [%ld/%ld] score: %.2f\n", j, indexes.size(), scores[j]);

      perception->bevDet3d.emplace_back(
          BevDetection3D{Bbox[j], scores[j], cls[j]});
    }
  }

  if (perception->bevDet3d.size() == 0) {
    perception->bevDet3d.emplace_back(BevDetection3D{EgoBbox(), 1.0f, 1});
  }

  return 0;
}
