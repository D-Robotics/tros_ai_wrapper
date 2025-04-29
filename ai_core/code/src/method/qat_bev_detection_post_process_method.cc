// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bev_detection_post_process_method.h"

#include <arm_neon.h>

#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevDetectionPostProcessMethod);

static void SigmoidDequantize(std::vector<ScoreInfo> &output, int32_t *input,
                              float *input_scale, int32_t *input_shape) {
  int32_t channel = input_shape[1];
  int32_t feature_size = input_shape[2] * input_shape[3];

  int32_t idx{0};
  for (int32_t i = 0; i < feature_size; i++) {
    for (int32_t c = 0; c < channel; c++) {
      float data_tmp = input[c * feature_size + i] * input_scale[c];
      float score = 1.0f / (std::exp(-data_tmp) + 1.0f);
      output.emplace_back(std::make_pair(score, idx++));
    }
  }
}

static void DeNormalize(std::vector<BevDetection3D> &bevDet3d,
                        std::vector<float> &scores,
                        std::vector<int32_t> &labels,
                        std::vector<int32_t> &reg_idx,
                        std::vector<float> &bev_range, float *reference_points,
                        int32_t *reg_pred, int32_t *reg_shape,
                        float *reg_scale) {
  int32_t bbox_param_size = reg_shape[1];
  int32_t feature_size = reg_shape[2] * reg_shape[3];
  int32_t bbox_num = reg_idx.size();
  for (int32_t i = 0; i < bbox_num; i++) {
    float reg_0 = reg_pred[0 * feature_size + reg_idx[i]] * reg_scale[0] +
                  reference_points[0 * feature_size + reg_idx[i]];
    float reg_1 = reg_pred[1 * feature_size + reg_idx[i]] * reg_scale[1] +
                  reference_points[1 * feature_size + reg_idx[i]];
    float reg_2 = reg_pred[2 * feature_size + reg_idx[i]] * reg_scale[2] +
                  reference_points[2 * feature_size + reg_idx[i]];
    bevDet3d[i].bbox.xs = bev_range[0] + (bev_range[3] - bev_range[0]) * 1.0f /
                                             (std::exp(-reg_0) + 1.0f);
    bevDet3d[i].bbox.ys = bev_range[1] + (bev_range[4] - bev_range[1]) * 1.0f /
                                             (std::exp(-reg_1) + 1.0f);
    bevDet3d[i].bbox.height = bev_range[2] + (bev_range[5] - bev_range[2]) *
                                                 1.0f /
                                                 (std::exp(-reg_2) + 1.0f);

    bevDet3d[i].bbox.dim_0 =
        std::exp(reg_pred[3 * feature_size + reg_idx[i]] * reg_scale[3]);
    bevDet3d[i].bbox.dim_1 =
        std::exp(reg_pred[4 * feature_size + reg_idx[i]] * reg_scale[4]);
    bevDet3d[i].bbox.dim_2 =
        std::exp(reg_pred[5 * feature_size + reg_idx[i]] * reg_scale[5]);

    float rot_sin = reg_pred[6 * feature_size + reg_idx[i]] * reg_scale[6];
    float rot_cos = reg_pred[7 * feature_size + reg_idx[i]] * reg_scale[7];
    bevDet3d[i].bbox.rot = std::atan2(rot_sin, rot_cos);

    if (bbox_param_size == 10) {
      bevDet3d[i].bbox.vel_0 =
          reg_pred[8 * feature_size + reg_idx[i]] * reg_scale[8];
      bevDet3d[i].bbox.vel_1 =
          reg_pred[9 * feature_size + reg_idx[i]] * reg_scale[9];
    }

    bevDet3d[i].score = scores[i];
    bevDet3d[i].label = labels[i];
  }
}

int QATBevDetectionPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevDetectionPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("topk")) {
    topk_ = document["topk"].GetInt();
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
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

  if (document.HasMember("bev_range")) {
    auto bev_range_value = document["bev_range"].GetArray();
    bev_range_.resize(bev_range_value.Size());
    for (int i = 0; i < bev_range_value.Size(); i++) {
      bev_range_[i] = bev_range_value[i].GetFloat();
    }
  }

  if (document.HasMember("reference_points_path")) {
    reference_points_path_ = document["reference_points_path"].GetString();
  }

  return 0;
}

PerceptionPtr QATBevDetectionPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevDetectionPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
    Perception *perception) {
  perception->type = Perception::BEV;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];

  for (int32_t i = 0; i < tensors.size(); i++) {
    hbUCPMemFlush(&(tensors[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // read reference points
  std::string reference_points_file =
      reference_points_path_ + "/" + image_tensor->image_name;
  int32_t data_length = 0;
  char *data = nullptr;
  auto ret = read_binary_file(reference_points_file, &data, &data_length);
  if (ret != 0) {
    VLOG(EXAMPLE_SYSTEM) << "read reference points error";
    return -1;
  }
  float *reference_points = reinterpret_cast<float *>(data);

  int32_t *cls_pred = reinterpret_cast<int32_t *>(tensors[0].sysMem.virAddr);
  int32_t *cls_shape = tensors[0].properties.validShape.dimensionSize;
  float *cls_scale = tensors[0].properties.scale.scaleData;
  int32_t *reg_pred = reinterpret_cast<int32_t *>(tensors[1].sysMem.virAddr);
  int32_t *reg_shape = tensors[1].properties.validShape.dimensionSize;
  float *reg_scale = tensors[1].properties.scale.scaleData;

  int32_t num_classes = cls_shape[1];

  std::vector<ScoreInfo> scores_info;
  SigmoidDequantize(scores_info, cls_pred, cls_scale, cls_shape);

  std::sort(scores_info.begin(), scores_info.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first > b.first;
            });

  std::vector<int32_t> reg_idx;
  std::vector<int32_t> label;
  std::vector<float> scores;
  int32_t score_idx{-1};
  for (int32_t i = 0; i < topk_; i++) {
    if (score_threshold_ > 0) {
      if (scores_info[i].first > score_threshold_) {
        score_idx = scores_info[i].second;
        scores.emplace_back(scores_info[i].first);
      }
    } else {
      score_idx = scores_info[i].second;
      scores.emplace_back(scores_info[i].first);
    }
    reg_idx.emplace_back(score_idx / num_classes);
    label.emplace_back(score_idx % num_classes);
  }

  perception->bevDet3d.resize(reg_idx.size());
  DeNormalize(perception->bevDet3d, scores, label, reg_idx, bev_range_,
              reference_points, reg_pred, reg_shape, reg_scale);

  delete[] data;

  return 0;
}
