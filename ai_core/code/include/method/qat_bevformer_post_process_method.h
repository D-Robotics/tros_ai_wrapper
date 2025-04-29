// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFORMER_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFORMER_POST_PROCESS_METHOD_H_

#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

typedef std::pair<float, int> ScoreInfo;

float sigmoid(float x);
std::vector<float> flatten(const std::vector<std::vector<float>> &vec);
std::pair<std::vector<float>, std::vector<int>> topk(
    const std::vector<float> &vec, int k);

/**
 * Method for post processing
 */
class QATBevformerPostProcessMethod : public PostProcessMethod {
 public:
  /**
   * Init post process from json string
   * @param[in] config: config json string
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  virtual ~QATBevformerPostProcessMethod() {
    if (quant_data_.virAddr != nullptr) {
      hbUCPFree(&quant_data_);
    }
  }

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

 private:
  std::vector<int> ori_shape_{900, 1600};
  std::vector<int> resize_shape_{480, 800};

  int topk_{300};
  int num_classes_{10};
  float score_threshold_{-1.0f};
  hbUCPSysMem quant_data_{0, nullptr, 0};
  std::vector<float> bev_range_{-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
  std::vector<float> post_center_range_{-61.2f, -61.2f, -10.0f,
                                        61.2f,  61.2f,  10.0f};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFORMER_POST_PROCESS_METHOD_H_
