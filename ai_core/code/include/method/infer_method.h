// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_INFER_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_INFER_METHOD_H_

#include <memory>
#include <string>
#include <vector>

#include "base/common_def.h"
#include "glog/logging.h"
#include "hobot/dnn/hb_dnn.h"
#include "method.h"
#include "method/method_data.h"
#include "utils/pool.h"

/**
 * Method for prediction
 */
class InferMethod : public Method {
 public:
  /**
   * Init predict method from config string
   * @param[in] config: config json string
   *    config should be in the json format
   *    for example:
   *    {
   *        "core": 1 # 1 for single core, 2 for dual core
   *        "model_file": ".bin",
   *        "output_name": "tensors",
   *        "tensor_pool_size": 8 # output tensors pool size
   *    }
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  hbDNNHandle_t GetModelHandle();

  TensorVectorPtr DoProcess(ImageTensor *image_tensor);

  void Rlease();

  ~InferMethod() = default;

 private:
  hbDNNPackedHandle_t packed_dnn_handle_ = nullptr;
  int core_{0};
  hbDNNHandle_t dnn_handle_{nullptr};
  static bool is_releated_model_;
  static bool is_temporal_model_;
  static bool
      is_temporal_multiframe_model_;  // for trajectory prediction (QCNet)
  static int input_num_;

  int32_t tensor_pool_size_{8};
  std::shared_ptr<Pool<TensorVector>> tensor_vector_pool_;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_INFER_METHOD_H_
