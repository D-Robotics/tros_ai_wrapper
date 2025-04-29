// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_PREPROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_PREPROCESS_METHOD_H_

#include <string>

#include "method.h"
#include "method_data.h"

/**
 * Method for post processing
 */
class PreProcessMethod : public Method {
 public:
  virtual int32_t DoProcess(std::string path, int32_t input_count,
                            ImageTensor *image_tensor) = 0;

  ~PreProcessMethod() override = default;

  void SetModelHandle(hbDNNHandle_t dnn_handle) { dnn_handle_ = dnn_handle; }

 protected:
  hbDNNHandle_t dnn_handle_;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_PREPROCESS_METHOD_H_
