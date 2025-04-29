// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevformer_post_process_method.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevformerPostProcessMethod);

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float expf(float x) { return std::exp(x); }

inline float atan2f(float y, float x) { return std::atan2(y, x); }

void bbox_convert(std::vector<BevDetection3D>& bevDet3d,
                  std::vector<float>& scores, std::vector<int32_t>& labels,
                  std::vector<std::vector<float>> Bbox3D) {
  for (int32_t i{0}; i < scores.size(); i++) {
    bevDet3d[i].score = scores[i];
    bevDet3d[i].label = labels[i];

    bevDet3d[i].bbox.xs = Bbox3D[i][0];
    bevDet3d[i].bbox.ys = Bbox3D[i][1];
    bevDet3d[i].bbox.height = Bbox3D[i][2];

    bevDet3d[i].bbox.dim_0 = Bbox3D[i][3];
    bevDet3d[i].bbox.dim_1 = Bbox3D[i][4];
    bevDet3d[i].bbox.dim_2 = Bbox3D[i][5];

    bevDet3d[i].bbox.rot = Bbox3D[i][6];
    bevDet3d[i].bbox.vel_0 = Bbox3D[i][7];
    bevDet3d[i].bbox.vel_1 = Bbox3D[i][8];
  }
}

std::vector<std::vector<float>> denormalize_bbox(
    const std::vector<std::vector<float>>& normalized_bboxes) {
  size_t num_bboxes = normalized_bboxes.size();
  std::vector<std::vector<float>> denormalized_bboxes(num_bboxes,
                                                      std::vector<float>(10));

  for (size_t i = 0; i < num_bboxes; ++i) {
    const auto& bbox = normalized_bboxes[i];
    float rot_sine = bbox[6];
    float rot_cosine = bbox[7];
    float rot = atan2f(rot_sine, rot_cosine);

    float cx = bbox[0];
    float cy = bbox[1];
    float cz = bbox[4];

    float w = expf(bbox[2]);
    float bl = expf(bbox[3]);
    float h = expf(bbox[5]);

    float vx = 0.0f;
    float vy = 0.0f;
    if (bbox.size() > 8) {
      vx = bbox[8];
      vy = bbox[9];
    }

    denormalized_bboxes[i] = {cx, cy, cz, w, bl, h, rot, vx, vy};
  }

  return denormalized_bboxes;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& vec) {
  std::vector<float> flat;
  for (const auto& sub_vec : vec) {
    flat.insert(flat.end(), sub_vec.begin(), sub_vec.end());
  }
  return flat;
}

std::pair<std::vector<float>, std::vector<int>> topk(
    const std::vector<float>& vec, int k) {
  std::vector<int> indices(vec.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&vec](int a, int b) { return vec[a] > vec[b]; });

  std::vector<float> topk_values(k);
  std::vector<int> topk_indices(k);
  for (int i = 0; i < k; ++i) {
    topk_values[i] = vec[indices[i]];
    topk_indices[i] = indices[i];
  }
  return {topk_values, topk_indices};
}

int QATBevformerPostProcessMethod::InitFromJsonString(
    const std::string& config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevformerPostProcessMethod Json string:"
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

  if (document.HasMember("bev_range")) {
    auto bev_range = document["bev_range"].GetArray();
    bev_range_.resize(bev_range.Size());
    for (int i = 0; i < bev_range.Size(); i++) {
      bev_range_[i] = bev_range[i].GetFloat();
    }
  }

  if (document.HasMember("post_center_range")) {
    auto post_center_range = document["post_center_range"].GetArray();
    post_center_range_.resize(post_center_range.Size());
    for (int i = 0; i < post_center_range.Size(); i++) {
      post_center_range_[i] = post_center_range[i].GetFloat();
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

PerceptionPtr QATBevformerPostProcessMethod::DoProcess(
    ImageTensor* image_tensor, TensorVectorPtr& output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevformerPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor>& tensors, ImageTensor* image_tensor,
    Perception* perception) {
  perception->type = Perception::BEV;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];
  image_tensor->is_pad_resize = true;

  for (int i = 0; i < tensors.size(); i++) {
    hbUCPMemFlush(&(tensors[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  hbDNNTensor& outputs_classes = tensors[1];
  hbDNNTensor& reference_out = tensors[2];
  hbDNNTensor& bbox_outputs = tensors[3];
  hbDNNTensor& bev_embed = tensors[0];

  int32_t* outputs_classes_data =
      static_cast<int32_t*>(outputs_classes.sysMem.virAddr);
  int16_t* reference_out_data =
      static_cast<int16_t*>(reference_out.sysMem.virAddr);
  int32_t* bbox_outputs_data =
      static_cast<int32_t*>(bbox_outputs.sysMem.virAddr);
  int8_t* bev_embed_data = static_cast<int8_t*>(bev_embed.sysMem.virAddr);

  // scale data
  float* cls_scale = outputs_classes.properties.scale.scaleData;
  float* ref_scale = reference_out.properties.scale.scaleData;
  float* bbox_scale = bbox_outputs.properties.scale.scaleData;
  float* emb_scale = bev_embed.properties.scale.scaleData;

  int32_t num_boxes =
      bbox_outputs.properties.validShape.dimensionSize[1];  // 900
  int32_t bbox_dim = bbox_outputs.properties.validShape.dimensionSize[2];  // 10
  int32_t ref_dim = reference_out.properties.validShape.dimensionSize[2];  // 3

  int32_t class_dim_stride = outputs_classes.properties.stride[1] /
                             outputs_classes.properties.stride[2];  // 16
  int32_t ref_dim_stride = reference_out.properties.stride[1] /
                           reference_out.properties.stride[2];  // 4
  int32_t bbox_dim_stride = bbox_outputs.properties.stride[1] /
                            bbox_outputs.properties.stride[2];  // 16

  bool is_temporal = WorkflowPlugin::GetInstance()->IsTemporalModel();
  VLOG(EXAMPLE_DEBUG) << "Is temporal model: " << is_temporal
                      << "; output tensor count:" << tensors.size();

  // bevformer
  if (is_temporal && tensors.size() == 4) {
    VLOG(EXAMPLE_DEBUG) << "Frame id: " << image_tensor->frame_id;
    int32_t associated_prev_num =
        WorkflowPlugin::GetInstance()->GetAssociatedPrevNum();

    auto output_size{bev_embed.properties.alignedByteSize / sizeof(int8_t)};
    if (quant_data_.virAddr == nullptr) {
      hbUCPMallocCached(&quant_data_, output_size * sizeof(float), 0);
    }
    float* quant_data_ptr = reinterpret_cast<float*>(quant_data_.virAddr);
    for (int32_t i{0}; i < output_size; i++) {
      quant_data_ptr[i] = bev_embed_data[i] * (*emb_scale);
    }
    WorkflowPlugin::GetInstance()->UpdateTemporalQuantTensor<float*>(
        &quant_data_, associated_prev_num - 1, 0);
  }

  std::vector<std::vector<float>> all_cls_scores(
      num_boxes, std::vector<float>(num_classes_));
  std::vector<std::vector<float>> all_bbox_preds(num_boxes,
                                                 std::vector<float>(bbox_dim));

  for (int i = 0; i < num_boxes; ++i) {
    for (int j = 0; j < num_classes_; ++j) {
      float score =
          static_cast<float>(outputs_classes_data[i * class_dim_stride + j]) *
          cls_scale[j];
      all_cls_scores[i][j] = sigmoid(score);
    }

    std::vector<float> tmp(bbox_dim_stride);
    for (int j = 0; j < bbox_dim; ++j) {
      tmp[j] = static_cast<float>(bbox_outputs_data[i * bbox_dim_stride + j]) *
               (bbox_scale[j]);
    }

    int16_t* reference = reference_out_data + i * ref_dim_stride;

    tmp[0] += static_cast<float>(reference[0]) * (*ref_scale);
    tmp[1] += static_cast<float>(reference[1]) * (*ref_scale);
    tmp[0] = 1.0f / (1.0f + std::exp(-tmp[0]));
    tmp[1] = 1.0f / (1.0f + std::exp(-tmp[1]));

    tmp[4] += static_cast<float>(reference[2]) * (*ref_scale);
    tmp[4] = 1.0f / (1.0f + std::exp(-tmp[4]));

    tmp[0] = tmp[0] * (bev_range_[3] - bev_range_[0]) + bev_range_[0];
    tmp[1] = tmp[1] * (bev_range_[4] - bev_range_[1]) + bev_range_[1];
    tmp[4] = tmp[4] * (bev_range_[5] - bev_range_[2]) + bev_range_[2];

    for (int j = 0; j < bbox_dim; ++j) {
      all_bbox_preds[i][j] = tmp[j];
    }
  }

  std::vector<float> flat_scores = flatten(all_cls_scores);

  auto top_res = topk(flat_scores, topk_);
  auto scores_topk = top_res.first;
  auto indices_topk = top_res.second;

  std::vector<float> final_scores(topk_);
  std::vector<int> final_labels(topk_);
  std::vector<int> bbox_indices(topk_);

  for (int i = 0; i < topk_; ++i) {
    final_scores[i] = flat_scores[indices_topk[i]];
    final_labels[i] = indices_topk[i] % num_classes_;
    bbox_indices[i] = indices_topk[i] / num_classes_;
  }

  std::vector<std::vector<float>> selected_bboxes(
      topk_, std::vector<float>(all_bbox_preds[0].size()));
  for (int i = 0; i < topk_; ++i) {
    selected_bboxes[i] = all_bbox_preds[bbox_indices[i]];
  }

  auto final_box_preds = denormalize_bbox(selected_bboxes);

  // threshold mask
  std::vector<bool> thresh_mask(topk_, false);
  if (score_threshold_ > 0) {
    for (int i = 0; i < topk_; ++i) {
      thresh_mask[i] = final_scores[i] > score_threshold_;
    }
    float tmp_score = score_threshold_;
    while (std::none_of(thresh_mask.begin(), thresh_mask.end(),
                        [](bool v) { return v; })) {
      tmp_score *= 0.9;
      if (tmp_score < 0.01) {
        std::fill(thresh_mask.begin(), thresh_mask.end(), true);
        break;
      }
      for (int i = 0; i < topk_; ++i) {
        thresh_mask[i] = final_scores[i] >= tmp_score;
      }
    }
  } else {
    std::fill(thresh_mask.begin(), thresh_mask.end(), true);
  }

  std::vector<std::vector<float>> boxes3d;
  std::vector<float> scores;
  std::vector<int32_t> labels;

  // filter boxes
  std::vector<bool> mask(final_box_preds.size(), true);

  for (int32_t i{0}; i < final_box_preds.size(); ++i) {
    const auto& box = final_box_preds[i];
    bool in_range{true};

    for (int32_t j{0}; j < 3; ++j) {
      if (box[j] < post_center_range_[j] ||
          box[j] > post_center_range_[j + 3]) {
        in_range = false;
        break;
      }
    }

    if (!in_range) {
      mask[i] = false;
    }
  }

  for (int32_t i{0}; i < mask.size(); ++i) {
    if (mask[i] && thresh_mask[i]) {
      boxes3d.push_back(final_box_preds[i]);
      scores.push_back(final_scores[i]);
      labels.push_back(final_labels[i]);
    }
  }

  perception->bevDet3d.resize(scores.size());
  bbox_convert(perception->bevDet3d, scores, labels, boxes3d);
  return 0;
}
