// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/image_list_data_iterator.h"

#include <iostream>

#include "glog/logging.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/data_transformer.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_DATA_ITERATOR(image, ImageListDataIterator)

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

int ImageListDataIterator::Init(std::string config_string) {
  int ret_code = DataIterator::Init(config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (check_model_input(tensors_, {}) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Check pyramid model failed";
    return -1;
  }

  if (cache_able_) {
    for (int i = 0; i < max_cache_size_; i++) {
      if (i >= image_files_.size()) break;
      ImageTensor image_tensor;
      image_tensor.tensors = tensors_;
      VLOG(EXAMPLE_DEBUG) << "[" << i << "]: " << image_files_[i];

      // preprocessed_image only have one input: pyramid(Y+UV)
      image_tensor.input_height =
          image_tensor.tensors[0].properties.validShape.dimensionSize[1];
      image_tensor.input_width =
          image_tensor.tensors[0].properties.validShape.dimensionSize[2];

      if (prepare_pyramid_input_tensor(&image_tensor, 0)) {
        VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
        return -1;
      }
      fill_image_to_tensor(image_files_[i], &image_tensor, 0, transformers_);

      cache_.push_back(image_tensor);
    }
  }

  return 0;
}

bool ImageListDataIterator::Next(ImageTensor *image_tensor) {
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
    if (send_index_ >= image_files_.size()) {
      is_finish_ = true;
      return false;
    }
    std::string image_file = image_files_[send_index_++];

    image_tensor->tensors = tensors_;

    // preprocessed_image only have one input: pyramid(Y+UV)
    image_tensor->input_height =
        image_tensor->tensors[0].properties.validShape.dimensionSize[1];
    image_tensor->input_width =
        image_tensor->tensors[0].properties.validShape.dimensionSize[2];
    if (prepare_pyramid_input_tensor(image_tensor, 0)) {
      VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
      return -1;
    }
    fill_image_to_tensor(image_file, image_tensor, 0, transformers_);

    if (loop_able_) {
      send_index_ = send_index_ % image_files_.size();
    }
  }

  image_tensor->frame_id = NextFrameId();
  return true;
}

void ImageListDataIterator::Release(ImageTensor *image_tensor) {
  if (!cache_able_) {
    for (size_t i{0}; i < image_tensor->tensors.size(); i++) {
      release_tensor(&(image_tensor->tensors[i]));
    }
  }
}

int ImageListDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("image_list_file")) {
    std::string image_list_file = document["image_list_file"].GetString();
    if (!ParseImageList(image_list_file, image_files_)) {
      VLOG(EXAMPLE_SYSTEM) << "Parse image list error!";
      return -1;
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

ImageListDataIterator::~ImageListDataIterator() {}

bool ImageListDataIterator::HasNext() { return !is_finish_; }
