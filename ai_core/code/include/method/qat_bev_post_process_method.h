// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEV_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEV_POST_PROCESS_METHOD_H_

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

/**
 * Method for post processing
 */
class QATBevPostProcessMethod : public PostProcessMethod {
 public:
  /**
   * Init post process from json string
   * @param[in] config: config json string
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATBevPostProcessMethod() override = default;

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

 private:
  int height_{128};
  int width_{128};
  std::vector<int> ori_shape_{900, 1600};
  std::vector<int> resize_shape_{540, 960};

  bool norm_bbox_{true};
  int topk_{50};
  int out_size_factor_{1};
  float score_threshold_{0.1};

  int kernel_{3};
  int post_max_size_{100};
  int pre_max_size_{1000};
  std::vector<float> bev_size_{51.2, 51.2, 0.8};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEV_POST_PROCESS_METHOD_H_
