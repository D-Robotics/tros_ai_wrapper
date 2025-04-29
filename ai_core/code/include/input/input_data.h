// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_INPUT_DATA_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_INPUT_DATA_H_

#include <ostream>
#include <string>
#include <vector>

#include "base/common_def.h"
#include "glog/logging.h"
#include "hobot/dnn/hb_dnn.h"
#include "utils/utils.h"

typedef struct ImageTensor {
  hbDNNTensor tensor;
  std::vector<hbDNNTensor> tensors;
  int32_t frame_id = 0;
  std::string image_name;
  int ori_image_width;
  int ori_image_height;
  int resize_width;
  int resize_height;
  int input_width;
  int input_height;
  std::string ori_image_path;
  uint64_t pre_duration{0U};
  int32_t frames_per_sample_infer{
      0};  // Number of frames for single sample inference

  int num_img = 1;
  std::vector<std::string> ori_image_path_list;

  // TODO(@horizon.ai):
  bool is_pad_resize = false;

  inline int32_t height() {
    int height{input_height};
    VLOG(EXAMPLE_DEBUG) << "height: " << height;
    return height;
  }

  inline int32_t width() {
    int width{input_width};
    VLOG(EXAMPLE_DEBUG) << "width: " << width;
    return width;
  }

  inline int32_t ori_height() const { return ori_image_height; }

  inline int32_t ori_width() const { return ori_image_width; }

  friend std::ostream &operator<<(std::ostream &os, ImageTensor &image_tensor) {
    os << "{"
       << R"("image_name")"
       << ":\"" << image_tensor.image_name << "\", "
       << R"("image_width")"
       << ":" << image_tensor.ori_image_width << ", "
       << R"("image_height")"
       << ":" << image_tensor.ori_image_height << "}";
    return os;
  }
} ImageTensor;

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_INPUT_DATA_H_
