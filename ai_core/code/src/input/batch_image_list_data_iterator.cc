// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/batch_image_list_data_iterator.h"

#include <iostream>

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "utils/data_transformer.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_DATA_ITERATOR(batch_image, BatchImageListDataIterator)

/**
 * parse image list from config file
 * @param[in] list_file image list config file
 * @param[out] files list of images
 * @return
 */
static bool ParseImageList(const std::string &list_file,
                           std::vector<std::string> &files) {
  std::ifstream lst_ifstream(list_file);
  if (!lst_ifstream) {
    VLOG(EXAMPLE_SYSTEM) << "open image list file : " << list_file
                         << " failed!";
    return false;
  }
  std::string line;
  while (std::getline(lst_ifstream, line)) {
    std::istringstream ss(line);
    std::string temp;
    std::getline(ss, temp, ' ');
    if (!temp.empty()) {
      files.push_back(std::move(temp));
    }
  }
  lst_ifstream.close();
  return true;
}

int BatchImageListDataIterator::Init(std::string config_string) {
  int ret_code = DataIterator::Init(config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (check_model_input(tensors_, {}) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Check pyramid model failed";
    return -1;
  }

  size_t input_count{tensors_.size()};
  if (input_count / 2 != image_list_file_.size()) {
    VLOG(EXAMPLE_SYSTEM) << "Get input file error!";
    return -1;
  }
  image_files_.resize(input_count / 2);

  for (size_t i{0}; i < input_count / 2; i++) {
    if (!ParseImageList(image_list_file_[i], image_files_[i])) {
      VLOG(EXAMPLE_SYSTEM) << "Parse image list error!";
      return -1;
    }
  }

  if (cache_able_) {
    for (int i = 0; i < max_cache_size_; i++) {
      if (i >= image_files_[0].size()) break;
      ImageTensor image_tensor;
      image_tensor.tensors = tensors_;

      // NV12_0 + NV12_1: Y_0 + UV_0 + Y_1 + UV_1
      for (size_t cnt{0}; cnt < input_count; cnt += 2) {
        if (prepare_pyramid_input_tensor(&image_tensor, cnt)) {
          VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
          return -1;
        }
        fill_image_to_tensor(image_files_[cnt / 2][i], &image_tensor, cnt,
                             transformers_);
      }
      cache_.push_back(image_tensor);
    }
  }

  return 0;
}

bool BatchImageListDataIterator::Next(ImageTensor *image_tensor) {
  if (last_frame_id + 1 >= max_frame_count_) {
    is_finish_ = true;
    return false;
  }

  if (cache_able_) {
    if (send_index_ >= cache_.size()) {
      is_finish_ = true;
      return false;
    }
    *image_tensor = cache_[send_index_++];
    if (loop_able_) {
      send_index_ = send_index_ % cache_.size();
    }

  } else {
    if (send_index_ >= image_files_[0].size()) {
      is_finish_ = true;
      return false;
    }

    image_tensor->tensors = tensors_;

    // NV12_0 + NV12_1: Y_0 + UV_0 + Y_1 + UV_1
    for (size_t cnt{0}; cnt < tensors_.size(); cnt += 2) {
      if (prepare_pyramid_input_tensor(image_tensor, cnt)) {
        VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
        return -1;
      }
      fill_image_to_tensor(image_files_[cnt / 2][send_index_], image_tensor,
                           cnt, transformers_);
    }

    send_index_++;

    if (loop_able_) {
      send_index_ = send_index_ % image_files_[0].size();
    }
  }

  image_tensor->frame_id = NextFrameId();
  return true;
}

void BatchImageListDataIterator::Release(ImageTensor *image_tensor) {
  if (!cache_able_) {
    for (int i = 0; i < image_tensor->tensors.size(); i++) {
      release_tensor(&(image_tensor->tensors[i]));
    }
  }
}

int BatchImageListDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("image_list_file")) {
    auto file_list = document["image_list_file"].GetArray();
    image_list_file_.resize(file_list.Size());
    for (int i = 0; i < file_list.Size(); i++) {
      image_list_file_[i] = file_list[i].GetString();
    }
  }

  if (document.HasMember("need_pre_load")) {
    cache_able_ = document["need_pre_load"].GetBool();
  }

  if (document.HasMember("need_loop")) {
    loop_able_ = document["need_loop"].GetBool();
  }

  if (document.HasMember("max_frame_count")) {
    max_frame_count_ = document["max_frame_count"].GetInt();
  }

  if (document.HasMember("max_cache")) {
    max_cache_size_ = document["max_cache"].GetInt();
  }

  std::string transformers = "default_transformers";
  if (document.HasMember("transformers")) {
    transformers = document["transformers"].GetString();
  }
  transformers_ = get_transformers(transformers);
  if (!transformers_) {
    VLOG(EXAMPLE_SYSTEM) << "Get transformers: " << transformers << " failed";
    return -1;
  }

  return 0;
}

BatchImageListDataIterator::~BatchImageListDataIterator() {}

bool BatchImageListDataIterator::HasNext() { return !is_finish_; }
