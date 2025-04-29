// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/preprocessed_image_iterator.h"

#include "glog/logging.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_DATA_ITERATOR(preprocessed_image, PreprocessedImageIterator)

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
    VLOG(EXAMPLE_SYSTEM) << "open file list file : " << list_file << " failed!";
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

int PreprocessedImageIterator::Init(std::string config_string) {
  int ret_code = DataIterator::Init(config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (is_pyramid_input_) {
    if (check_model_input(tensors_, {}) != 0) {
      VLOG(EXAMPLE_SYSTEM) << "Check pyramid model failed";
      return -1;
    }
  } else {
    hbDNNHandle_t dnn_handle = WorkflowPlugin::GetInstance()->GetModelHandle();
    int32_t input_count{0};
    if (hbDNNGetInputCount(&input_count, dnn_handle)) {
      VLOG(EXAMPLE_SYSTEM) << "Get input count failed!";
      return -1;
    }

    tensors_.resize(input_count);
    for (int32_t idx = 0; idx < input_count; ++idx) {
      hbDNNTensorProperties &model_properties = tensors_[idx].properties;
      if (hbDNNGetInputTensorProperties(&model_properties, dnn_handle, idx)) {
        VLOG(EXAMPLE_SYSTEM) << "Get InputTensorProperties failed!";
        return -1;
      }
    }
  }

  if (cache_able_) {
    for (int i = 0; i < max_cache_size_; i++) {
      if (i >= image_files_.size()) break;
      ImageTensor image_tensor;
      image_tensor.tensors = tensors_;

      // preprocessed_image only have one input: ddr or pyramid(Y+UV)
      int input_height;
      int input_width;
      image_tensor.ori_image_path = image_files_[i];
      ParsePathParams(image_files_[i], image_tensor.image_name,
                      image_tensor.ori_image_height,
                      image_tensor.ori_image_width, input_height, input_width);
      image_tensor.is_pad_resize = true;
      image_tensor.input_height = input_height;
      image_tensor.input_width = input_width;
      image_tensor.tensors = tensors_;

      VLOG(EXAMPLE_DEBUG) << "[" << i << "]: " << image_files_[i];

      // read bin file
      int32_t data_length = 0;
      char *data = nullptr;
      auto ret = read_binary_file(image_files_[i], &data, &data_length);
      if (ret != 0) {
        return false;
      }

      int32_t calculation_length{0};
      if (is_pyramid_input_) {
        calculation_length =
            image_tensor.input_height * image_tensor.input_width * 3 / 2;
      } else {
        // If it is DDR input, then there is no padding

        calculation_length = image_tensor.tensors[0].properties.alignedByteSize;
      }

      // compare
      if (data_length != calculation_length) {
        VLOG(EXAMPLE_SYSTEM) << "input bin file length is not match"
                             << ", height: " << image_tensor.input_height
                             << ", width: " << image_tensor.input_width
                             << "; required: " << calculation_length
                             << ", given: " << data_length;
        delete[] data;
        return -1;
      }

      if (is_pyramid_input_) {
        if (prepare_pyramid_input_tensor(&image_tensor, 0)) {
          VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
          return -1;
        }
        fill_preprocessed_image_to_tensor(&image_tensor, data, 0);
      } else {
        hbUCPMallocCached(&(image_tensor.tensors[0].sysMem), calculation_length,
                          0);
        memcpy(reinterpret_cast<char *>(image_tensor.tensors[0].sysMem.virAddr),
               data, image_tensor.tensors[0].sysMem.memSize);
        hbUCPMemFlush(&(image_tensor.tensors[0].sysMem),
                      HB_SYS_MEM_CACHE_CLEAN);
      }

      // cache
      cache_.push_back(image_tensor);
      delete[] data;
    }
  }

  return 0;
}

bool PreprocessedImageIterator::Next(ImageTensor *image_tensor) {
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

    std::string bin_file = image_files_[send_index_++];

    // preprocessed_image only have one input: ddr or pyramid(Y+UV)
    int input_height;
    int input_width;
    image_tensor->ori_image_path = bin_file;
    ParsePathParams(bin_file, image_tensor->image_name,
                    image_tensor->ori_image_height,
                    image_tensor->ori_image_width, input_height, input_width);
    image_tensor->is_pad_resize = true;
    image_tensor->input_height = input_height;
    image_tensor->input_width = input_width;
    image_tensor->tensors = tensors_;

    // read bin file
    int32_t data_length = 0;
    char *data = nullptr;
    auto ret = read_binary_file(bin_file, &data, &data_length);
    if (ret != 0) {
      return false;
    }

    int32_t calculation_length{0};
    if (is_pyramid_input_) {
      calculation_length =
          image_tensor->input_height * image_tensor->input_width * 3 / 2;
    } else {
      // If it is DDR input, then there is no padding
      calculation_length = image_tensor->tensors[0].properties.alignedByteSize;
    }

    // compare
    if (data_length != calculation_length) {
      VLOG(EXAMPLE_SYSTEM) << "input bin file length is not match"
                           << ", height: " << image_tensor->input_height
                           << ", width: " << image_tensor->input_width
                           << "; required: " << calculation_length
                           << ", given: " << data_length;
      delete[] data;
      return -1;
    }

    if (is_pyramid_input_) {
      if (prepare_pyramid_input_tensor(image_tensor, 0)) {
        VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
        return -1;
      }
      fill_preprocessed_image_to_tensor(image_tensor, data, 0);
    } else {
      hbUCPMallocCached(&(image_tensor->tensors[0].sysMem), calculation_length,
                        0);
      memcpy(reinterpret_cast<char *>(image_tensor->tensors[0].sysMem.virAddr),
             data, image_tensor->tensors[0].sysMem.memSize);
      hbUCPMemFlush(&(image_tensor->tensors[0].sysMem), HB_SYS_MEM_CACHE_CLEAN);
    }

    delete[] data;

    if (loop_able_) {
      send_index_ = send_index_ % image_files_.size();
    }
  }
  image_tensor->frame_id = NextFrameId();
  return true;
}

void PreprocessedImageIterator::Release(ImageTensor *image_tensor) {
  if (!cache_able_) {
    for (size_t i{0}; i < image_tensor->tensors.size(); i++) {
      release_tensor(&(image_tensor->tensors[i]));
    }
  }
}

int PreprocessedImageIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("is_pyramid_input")) {
    is_pyramid_input_ = document["is_pyramid_input"].GetBool();
  }

  if (document.HasMember("image_list_file")) {
    std::string image_list_file = document["image_list_file"].GetString();
    if (!ParseFileList(image_list_file, image_files_)) {
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

  return 0;
}

void PreprocessedImageIterator::ParsePathParams(std::string &input_file,
                                                std::string &image_name,
                                                int &org_h, int &org_w,
                                                int &dst_h, int &dst_w) {
  // {name}_{org_h}_{org_w}_{dst_h}_{dst_w}.bin
  std::string filename = get_file_name(input_file);

  std::vector<std::string> tokens;
  rsplit(filename, '.', tokens, 2);
  std::string field1 = tokens[1];

  tokens.resize(0);
  rsplit(field1, '_', tokens, 5);

  dst_w = std::stoi(tokens[0]);
  dst_h = std::stoi(tokens[1]);
  org_w = std::stoi(tokens[2]);
  org_h = std::stoi(tokens[3]);
  image_name = tokens[4];
}

PreprocessedImageIterator::~PreprocessedImageIterator() {}

bool PreprocessedImageIterator::HasNext() { return !is_finish_; }
