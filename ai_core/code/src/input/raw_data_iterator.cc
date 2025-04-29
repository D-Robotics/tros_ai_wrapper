// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/raw_data_iterator.h"

#include <iostream>

#include "glog/logging.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_DATA_ITERATOR(raw_data, RawDataIterator)

/**
 * parse image list from config file
 * @param[in] list_file image list config file
 * @param[out] files list of images
 * @return
 */
static bool ParseFileList(const std::string &list_file,
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

int RawDataIterator::Init(std::string config_string) {
  int ret_code = DataIterator::Init(config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (!ParseFileList(image_list_file_, image_files_)) {
    VLOG(EXAMPLE_SYSTEM) << "Parse image list error!";
    return -1;
  }

  hbDNNHandle_t dnn_handle = WorkflowPlugin::GetInstance()->GetModelHandle();
  ret_code = hbDNNGetInputCount(&input_count_, dnn_handle);
  if (ret_code != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Get input count error!";
    return -1;
  }

  if (cache_able_) {
    for (int i = 0; i < max_cache_size_; i++) {
      if (i >= image_files_.size()) break;
      ImageTensor image_tensor;
      VLOG(EXAMPLE_DEBUG) << "[" << i << "]: " << image_files_[i];
      std::string bin_file = image_files_[i];

      image_tensor.ori_image_path = bin_file;
      image_tensor.image_name = get_file_name(image_files_[i]);

      preprocess_method_->DoProcess(bin_file, input_count_, &image_tensor);

      // cache
      cache_.push_back(image_tensor);
    }
  }

  return 0;
}

int RawDataIterator::InitPreProcess(const std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());
  auto pp_type = document["method_type"].GetString();
  preprocess_method_ =
      std::shared_ptr<PreProcessMethod>(dynamic_cast<PreProcessMethod *>(
          MethodFactory::GetInstance()->GetMethod(pp_type)));
  preprocess_method_->SetModelHandle(
      WorkflowPlugin::GetInstance()->GetModelHandle());
  VLOG(EXAMPLE_DEBUG) << "preprocess method config: "
                      << document["method_config"].GetString();
  std::ifstream ifs(document["method_config"].GetString());
  if (!ifs.good()) {
    VLOG(EXAMPLE_SYSTEM) << "open preprocess config file : "
                         << document["method_config"].GetString()
                         << " failed!";
    return -1;
  }
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document preprocess_config;
  preprocess_config.ParseStream(isw);

  return preprocess_method_->InitFromJsonString(
      json_to_string(preprocess_config));
}

bool RawDataIterator::Next(ImageTensor *image_tensor) {
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
    image_tensor->pre_duration = 0U;
    if (loop_able_) {
      send_index_ = send_index_ % cache_.size();
    }
  } else {
    if (send_index_ >= image_files_.size()) {
      is_finish_ = true;
      return false;
    }

    std::string bin_file = image_files_[send_index_++];

    image_tensor->ori_image_path = bin_file;
    image_tensor->image_name = get_file_name(bin_file);

    preprocess_method_->DoProcess(bin_file, input_count_, image_tensor);

    if (time_diff_ms_ > 0) {
      static auto last_send_time = std::chrono::system_clock::now();
      auto time_diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now() -
                               last_send_time)
                               .count();
      if (time_diff_ms > time_diff_ms_) {
        last_send_time = std::chrono::system_clock::now();
      } else {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(time_diff_ms_ - time_diff_ms));
        last_send_time = std::chrono::system_clock::now();
      }
    }

    if (loop_able_) {
      send_index_ = send_index_ % image_files_.size();
    }
  }
  image_tensor->frame_id = NextFrameId();
  return true;
}

void RawDataIterator::Release(ImageTensor *image_tensor) {
  if (!cache_able_) {
    for (int32_t i = 0; i < input_count_; i++) {
      release_tensor(&(image_tensor->tensors[i]));
    }
  }
}

int RawDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("preprocess")) {
    VLOG(EXAMPLE_DEBUG) << "Using config preprocess";
    // HB_CHECK_SUCCESS(InitPreProcess(json_to_string(document["preprocess"])),
    //                  "Parsing preprocess config failed.");
    if (InitPreProcess(json_to_string(document["preprocess"])) < 0) {
      VLOG(EXAMPLE_SYSTEM) << "Parsing preprocess config failed";
      return -1;
    }
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Missing config for preprocess.";
    return -1;
  }

  if (document.HasMember("data_type")) {
    data_type_ = static_cast<hbDNNDataType>(document["data_type"].GetInt());
  }

  if (document.HasMember("image_list_file")) {
    image_list_file_ = document["image_list_file"].GetString();
  }

  if (document.HasMember("need_pre_load")) {
    cache_able_ = document["need_pre_load"].GetBool();
  }

  if (document.HasMember("need_loop")) {
    loop_able_ = document["need_loop"].GetBool();
  }

  if (document.HasMember("time_diff_ms")) {
    time_diff_ms_ = document["time_diff_ms"].GetInt();
  }

  if (document.HasMember("max_frame_count")) {
    max_frame_count_ = document["max_frame_count"].GetInt();
  }

  VLOG(EXAMPLE_SYSTEM) << "loop_able: " << loop_able_ << ", time_diff_ms: " << time_diff_ms_;

  return 0;
}

RawDataIterator::~RawDataIterator() {}

bool RawDataIterator::HasNext() { return !is_finish_; }
