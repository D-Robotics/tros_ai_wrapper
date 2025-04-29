// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_RAW_DATA_ITERATOR_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_RAW_DATA_ITERATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "data_iterator.h"
#include "method/preprocess_method.h"

class RawDataIterator : public DataIterator {
 public:
  RawDataIterator() : DataIterator("raw_data_iterator") {}

  /**
   * Init raw data iterator from file
   * @param[in] config_string: config string, should be in the json format
   * @return 0 if success
   */
  int Init(std::string config_string) override;

  /**
   * Next Image Data read from file system
   * @param[out] image_tensor: image tensor
   * @return 0 if success
   */
  bool Next(ImageTensor *image_tensor) override;

  /**
   * Release image_tensor
   * @param[in] image_tensor: image tensor to be released
   */
  void Release(ImageTensor *image_tensor) override;

  /**
   * Check if has next image
   * @return 0 if finish
   */
  bool HasNext() override;

  ~RawDataIterator() override;

 private:
  int LoadConfig(std::string &config_string) override;

  int InitPreProcess(const std::string &config_string);

 private:
  std::shared_ptr<PreProcessMethod> preprocess_method_;

  std::string image_list_file_;
  std::vector<std::string> image_files_;
  hbDNNDataType data_type_ = HB_DNN_TENSOR_TYPE_F32;
  bool cache_able_{false};
  bool loop_able_{false};
  int time_diff_ms_{-1};
  int max_cache_size_{10};
  int max_frame_count_{INT32_MAX};
  int send_index_{0};
  int input_count_{1};
  std::vector<ImageTensor> cache_;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_INPUT_RAW_DATA_ITERATOR_H_
