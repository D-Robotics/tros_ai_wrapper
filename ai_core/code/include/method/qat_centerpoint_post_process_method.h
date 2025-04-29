// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_POST_PROCESS_METHOD_H_

#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

struct ScoresData {
  float value;
  int c;
  int w;
  int h;

  ScoresData(float score, int channel, float x, float y)
      : value(score), c(channel), w(x), h(y) {}

  friend std::ostream &operator<<(std::ostream &os, const ScoresData &scores) {
    os << std::fixed << std::setprecision(6) << scores.value << ", " << scores.c
       << ", " << scores.h << ", " << scores.w << std::endl;
    return os;
  }

  friend bool operator>(const ScoresData &lhs, const ScoresData &rhs) {
    return (lhs.value >= rhs.value);
  }
};

struct CompareScoresData {
  bool operator()(const ScoresData &lhs, const ScoresData &rhs) const {
    return lhs.value >= rhs.value;
  }
};

void NMS(std::vector<int> &keep_res, std::vector<std::vector<float>> &boxes,
         std::vector<float> &scores, float thresh, int per_max_size,
         int post_max_size);

void NmsRotated(std::vector<int> &keep, std::vector<std::vector<float>> &dets,
                std::vector<float> &scores, float iou_threshold);

/**
 * Method for post processing
 */
class QATCenterPointPostProcessMethod : public PostProcessMethod {
 public:
  /**
   * Init post process from json string
   * @param[in] config: config json string
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATCenterPointPostProcessMethod() override = default;

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

 private:
  bool norm_bbox_{true};
  int topk_{500};
  int out_size_factor_{4};
  float score_threshold_{0.1};
  int pre_max_size_{1000};
  int post_max_size_{83};
  float nms_thr_{0.2};
  std::vector<float> pc_range_{-51.2, -51.2};
  std::vector<float> voxel_size_{0.2, 0.2};
  std::vector<float> post_center_range_{-61.2, -61.2, -10.0, 61.2, 61.2, 10.0};

  int height_{128};
  int width_{128};
  int aligned_c_{4};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_POST_PROCESS_METHOD_H_
