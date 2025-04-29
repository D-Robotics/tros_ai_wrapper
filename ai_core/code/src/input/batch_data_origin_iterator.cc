// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/batch_data_origin_iterator.h"

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_DATA_ITERATOR(batch_data_origin, BatchDataOriginIterator)

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

int BatchDataOriginIterator::Init(std::string config_string) {
  int ret_code = DataIterator::Init(config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (check_model_input(tensors_, is_pyramid_input_) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Check pyramid model failed";
    return -1;
  }

  for (int i = 0; i < is_pyramid_input_.size(); i++) {
    if (is_pyramid_input_[i] && (i % 2)) {
      data_list_files_.insert(data_list_files_.begin() + i,
                              data_list_files_[i - 1]);
    }
    VLOG(EXAMPLE_SYSTEM) << i << ": " << data_list_files_[i];
  }

  input_count_ = tensors_.size();
  if (input_count_ != data_list_files_.size()) {
    VLOG(EXAMPLE_SYSTEM) << "Get input file error!";
    return -1;
  }

  data_files_.resize(input_count_);
  for (int i = 0; i < input_count_; i++) {
    if (!ParseFileList(data_list_files_[i], data_files_[i])) {
      VLOG(EXAMPLE_SYSTEM) << "Parse image list error!";
      return -1;
    }
  }

  if (cache_able_) {
    for (int i = 0; i < max_cache_size_; i++) {
      if (i >= data_files_[0].size()) break;
      ImageTensor image_tensor;
      std::string input_name = get_file_name(data_files_[0][i]);
      std::size_t pos = 0;
      if ((pos = input_name.find("-")) != std::string::npos) {
        input_name = input_name.substr(0, pos);
      }
      for (int k = 1; k < data_files_.size(); k++) {
        std::string tmp_input_name = get_file_name(data_files_[k][i]);
        if ((pos = tmp_input_name.find("-")) != std::string::npos) {
          tmp_input_name = tmp_input_name.substr(0, pos);
        }
        if (tmp_input_name != input_name) {
          VLOG(EXAMPLE_SYSTEM) << "input file error!";
          return -1;
        }
      }

      image_tensor.image_name = input_name;
      image_tensor.tensors = tensors_;

      for (size_t cnt{0}; cnt < input_count_; cnt++) {
        if (is_pyramid_input_[cnt]) {
          if (prepare_pyramid_input_tensor(&image_tensor, cnt)) {
            VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
            return -1;
          }

          // read bin file
          int32_t data_length = 0;
          char *data = nullptr;
          auto ret = read_binary_file(data_files_[cnt][i], &data, &data_length);
          if (ret != 0) {
            return false;
          }

          image_tensor.ori_image_path_list.push_back(data_files_[cnt][i]);

          int32_t calculation_length =
              image_tensor.tensors[cnt].properties.validShape.dimensionSize[1] *
              image_tensor.tensors[cnt].properties.validShape.dimensionSize[2] *
              3 / 2;

          // compare
          if (data_length != calculation_length) {
            VLOG(EXAMPLE_SYSTEM) << "input bin file length is not match"
                                 << ", height: "
                                 << image_tensor.tensors[cnt]
                                        .properties.validShape.dimensionSize[1]
                                 << ", width: "
                                 << image_tensor.tensors[cnt]
                                        .properties.validShape.dimensionSize[2]
                                 << "; required: " << calculation_length
                                 << ", given: " << data_length;
            delete[] data;
            return -1;
          }

          fill_preprocessed_image_to_tensor(&image_tensor, data, cnt);
          delete[] data;
          cnt += 1;
        } else {
          // prepare ddr input
          prepare_batch_tensor_and_quanti(&image_tensor.tensors[cnt],
                                          data_files_[cnt][i]);
        }
      }

      image_tensor.input_height =
          image_tensor.tensors[0].properties.validShape.dimensionSize[1];
      image_tensor.input_width =
          image_tensor.tensors[0].properties.validShape.dimensionSize[2];
      image_tensor.num_img =
          std::count(is_pyramid_input_.begin(), is_pyramid_input_.end(), true) /
          2;

      // cache
      cache_.push_back(image_tensor);
    }
  }

  return 0;
}

bool BatchDataOriginIterator::Next(ImageTensor *image_tensor) {
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
    if (send_index_ >= data_files_[0].size()) {
      is_finish_ = true;
      return false;
    }
    std::string input_name = get_file_name(data_files_[0][send_index_]);
    std::size_t pos = 0;
    if ((pos = input_name.find("-")) != std::string::npos) {
      input_name = input_name.substr(0, pos);
    }
    for (int i = 1; i < data_files_.size(); i++) {
      std::string tmp_input_name = get_file_name(data_files_[i][send_index_]);
      if ((pos = tmp_input_name.find("-")) != std::string::npos) {
        tmp_input_name = tmp_input_name.substr(0, pos);
      }
      if (tmp_input_name != input_name) {
        std::cout << tmp_input_name << std::endl;
        std::cout << input_name << std::endl;
        VLOG(EXAMPLE_SYSTEM) << "input file error!";
        return false;
      }
    }
    image_tensor->image_name = input_name;
    image_tensor->tensors = tensors_;

    for (size_t cnt{0}; cnt < input_count_; cnt++) {
      if (is_pyramid_input_[cnt]) {
        if (prepare_pyramid_input_tensor(image_tensor, cnt)) {
          VLOG(EXAMPLE_SYSTEM) << "prepare pyramid input tensor failed";
          return -1;
        }

        // read bin file
        int32_t data_length = 0;
        char *data = nullptr;
        auto ret = read_binary_file(data_files_[cnt][send_index_], &data,
                                    &data_length);
        if (ret != 0) {
          return false;
        }

        image_tensor->ori_image_path_list.push_back(
            data_files_[cnt][send_index_]);

        int32_t calculation_length =
            image_tensor->tensors[cnt].properties.validShape.dimensionSize[1] *
            image_tensor->tensors[cnt].properties.validShape.dimensionSize[2] *
            3 / 2;

        // compare
        if (data_length != calculation_length) {
          VLOG(EXAMPLE_SYSTEM) << "input bin file length is not match"
                               << ", height: "
                               << image_tensor->tensors[cnt]
                                      .properties.validShape.dimensionSize[1]
                               << ", width: "
                               << image_tensor->tensors[cnt]
                                      .properties.validShape.dimensionSize[2]
                               << "; required: " << calculation_length
                               << ", given: " << data_length;
          delete[] data;
          return -1;
        }

        fill_preprocessed_image_to_tensor(image_tensor, data, cnt);
        delete[] data;
        cnt += 1;
      } else {
        // prepare ddr input
        prepare_batch_tensor_and_quanti(&image_tensor->tensors[cnt],
                                        data_files_[cnt][send_index_]);
      }
    }

    image_tensor->input_height =
        image_tensor->tensors[0].properties.validShape.dimensionSize[1];
    image_tensor->input_width =
        image_tensor->tensors[0].properties.validShape.dimensionSize[2];
    image_tensor->num_img =
        std::count(is_pyramid_input_.begin(), is_pyramid_input_.end(), true) /
        2;

    send_index_++;

    if (loop_able_) {
      send_index_ = send_index_ % data_files_[0].size();
    }
  }
  image_tensor->frame_id = NextFrameId();
  return true;
}

void BatchDataOriginIterator::Release(ImageTensor *image_tensor) {
  if (!cache_able_) {
    for (int i = 0; i < input_count_; i++) {
      release_tensor(&(image_tensor->tensors[i]));
    }
  }
}

int BatchDataOriginIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("is_pyramid_input")) {
    rapidjson::Value &value = document["is_pyramid_input"];
    auto array = value.GetArray();
    for (int i = 0; i < array.Size(); ++i) {
      is_pyramid_input_.push_back(array[i].GetBool());
    }
  }

  if (document.HasMember("data_list_file")) {
    auto file_list = document["data_list_file"].GetArray();
    data_list_files_.resize(file_list.Size());
    for (int i = 0; i < file_list.Size(); i++) {
      data_list_files_[i] = file_list[i].GetString();
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

BatchDataOriginIterator::~BatchDataOriginIterator() {}

bool BatchDataOriginIterator::HasNext() { return !is_finish_; }
