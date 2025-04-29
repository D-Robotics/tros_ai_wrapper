// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/tensor_utils.h"

#include <memory.h>

#include <fstream>
#include <iostream>

#include "glog/logging.h"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp_sys.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "plugin/workflow_plugin.h"
#include "utils/image_utils.h"
#include "utils/utils.h"

int32_t check_model_input(std::vector<hbDNNTensor> &model_input_tensors,
                          std::vector<bool> is_pyramid_input) {
  hbDNNHandle_t dnn_handle = WorkflowPlugin::GetInstance()->GetModelHandle();
  int32_t input_count{0};
  int32_t ret = hbDNNGetInputCount(&input_count, dnn_handle);
  if (ret) {
    VLOG(EXAMPLE_SYSTEM) << "Get input count failed!";
    return -1;
  }

  if (is_pyramid_input.empty()) {
    is_pyramid_input.resize(input_count, true);
  }

  if (input_count != is_pyramid_input.size()) {
    VLOG(EXAMPLE_SYSTEM)
        << "Input count should be equal to is_pyramid_input.size(), "
        << "input_count: " << input_count
        << ", is_pyramid_input.size: " << is_pyramid_input.size();
    return -1;
  }

  model_input_tensors.resize(input_count);

  for (int32_t idx = 0; idx < input_count; ++idx) {
    hbDNNTensorProperties &model_properties =
        model_input_tensors[idx].properties;
    ret = hbDNNGetInputTensorProperties(&model_properties, dnn_handle, idx);

    if (is_pyramid_input[idx] && model_properties.alignedByteSize != -1) {
      VLOG(EXAMPLE_SYSTEM)
          << "The pyramid alignedByteSize should be -1, but given: "
          << model_properties.alignedByteSize;
      return -1;
    }
  }
  return 0;
}

int32_t prepare_pyramid_input_tensor(ImageTensor *image_tensor,
                                     int32_t y_index) {
  hbDNNTensorProperties &y_properties{
      image_tensor->tensors[y_index].properties};
  hbDNNTensorProperties &uv_properties{
      image_tensor->tensors[y_index + 1].properties};

  int32_t w_stride = ALIGN_32(y_properties.validShape.dimensionSize[2]);

  y_properties.stride[3] = sizeof(uint8_t);
  y_properties.stride[2] =
      y_properties.validShape.dimensionSize[3] * y_properties.stride[3];
  y_properties.stride[1] = w_stride * y_properties.stride[2];
  y_properties.stride[0] =
      y_properties.validShape.dimensionSize[1] * y_properties.stride[1];

  uv_properties.stride[3] = sizeof(uint8_t);
  uv_properties.stride[2] =
      uv_properties.validShape.dimensionSize[3] * uv_properties.stride[3];
  uv_properties.stride[1] = w_stride / 2 * uv_properties.stride[2];
  uv_properties.stride[0] =
      uv_properties.validShape.dimensionSize[1] * uv_properties.stride[1];

  VLOG(EXAMPLE_DEBUG) << "0-stride: " << y_properties.stride[0] << ", "
                      << y_properties.stride[1] << ", "
                      << y_properties.stride[2] << ", "
                      << y_properties.stride[3];

  VLOG(EXAMPLE_DEBUG) << "1-stride: " << uv_properties.stride[0] << ", "
                      << uv_properties.stride[1] << ", "
                      << uv_properties.stride[2] << ", "
                      << uv_properties.stride[3];

  hbUCPMallocCached(&image_tensor->tensors[y_index].sysMem,
                    y_properties.stride[0], 0);
  hbUCPMallocCached(&image_tensor->tensors[y_index + 1].sysMem,
                    uv_properties.stride[0], 0);

  return 0;
}

void fill_image_to_tensor(std::string &path, ImageTensor *image_tensor,
                          int32_t y_index, transformers_func transformers) {
  int32_t height =
      image_tensor->tensors[y_index].properties.validShape.dimensionSize[1];
  int32_t width =
      image_tensor->tensors[y_index].properties.validShape.dimensionSize[2];
  int32_t stride = image_tensor->tensors[y_index].properties.stride[1];

  image_tensor->image_name = get_file_name(path);
  image_tensor->ori_image_path = path;
  int &ori_width = image_tensor->ori_image_width;
  int &ori_height = image_tensor->ori_image_height;

  // read and resize
  cv::Mat mat;
  cv::Mat bgr_mat = cv::imread(path);
  ori_width = bgr_mat.cols;
  ori_height = bgr_mat.rows;

  mat.create(height, width, bgr_mat.type());

  transformers(image_tensor, mat, bgr_mat);

  cv::Mat nv12;
  bgr_to_nv12(mat, nv12);
  uint8_t *data = nv12.data;

  // Y data
  uint8_t *y_data = reinterpret_cast<uint8_t *>(
      image_tensor->tensors[y_index].sysMem.virAddr);
  for (int32_t h = 0; h < height; ++h) {
    auto *raw = y_data + h * stride;
    for (int32_t w = 0; w < width; ++w) {
      *raw++ = *data++;
    }
  }

  // UV data
  uint8_t *uv_data = reinterpret_cast<uint8_t *>(
      image_tensor->tensors[y_index + 1].sysMem.virAddr);
  for (int32_t h = 0; h < height / 2; ++h) {
    auto *raw = uv_data + h * stride;
    for (int32_t w = 0; w < width; ++w) {
      *raw++ = *data++;
    }
  }

  hbUCPMemFlush(&(image_tensor->tensors[y_index].sysMem),
                HB_SYS_MEM_CACHE_CLEAN);
  hbUCPMemFlush(&(image_tensor->tensors[y_index + 1].sysMem),
                HB_SYS_MEM_CACHE_CLEAN);
}

void fill_preprocessed_image_to_tensor(ImageTensor *image_tensor, char *data,
                                       int32_t y_index) {
  int32_t height =
      image_tensor->tensors[y_index].properties.validShape.dimensionSize[1];
  int32_t width =
      image_tensor->tensors[y_index].properties.validShape.dimensionSize[2];
  int32_t w_stride = image_tensor->tensors[0].properties.stride[1];

  uint8_t *image_data = reinterpret_cast<uint8_t *>(data);
  // Y data
  uint8_t *y_data = reinterpret_cast<uint8_t *>(
      image_tensor->tensors[y_index].sysMem.virAddr);
  for (int32_t h = 0; h < height; ++h) {
    auto *raw = y_data + h * w_stride;
    for (int32_t w = 0; w < width; ++w) {
      *raw++ = *image_data++;
    }
  }
  // UV data
  uint8_t *uv_data = reinterpret_cast<uint8_t *>(
      image_tensor->tensors[y_index + 1].sysMem.virAddr);
  for (int32_t h = 0; h < height / 2; ++h) {
    auto *raw = uv_data + h * w_stride;
    for (int32_t w = 0; w < width; ++w) {
      *raw++ = *image_data++;
    }
  }

  hbUCPMemFlush(&(image_tensor->tensors[y_index].sysMem),
                HB_SYS_MEM_CACHE_CLEAN);
  hbUCPMemFlush(&(image_tensor->tensors[y_index + 1].sysMem),
                HB_SYS_MEM_CACHE_CLEAN);
}

void flush_tensor(hbDNNTensor *tensor) {
  hbUCPMemFlush(&(tensor->sysMem), HB_SYS_MEM_CACHE_CLEAN);
}

void release_tensor(hbDNNTensor *tensor) { hbUCPFree(&(tensor->sysMem)); }

void prepare_output_tensor(std::vector<hbDNNTensor> &output,
                           hbDNNHandle_t dnn_handle) {
  int out_num;
  hbDNNGetOutputCount(&out_num, dnn_handle);
  output.resize(out_num);
  for (int i = 0; i < out_num; ++i) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
    int out_aligned_size = output_properties.alignedByteSize;
    hbUCPSysMem &mem = output[i].sysMem;
    hbUCPMallocCached(&mem, out_aligned_size, 0);
  }
}

void release_output_tensor(std::vector<hbDNNTensor> &output) {
  for (auto tensor : output) {
    hbUCPFree(&(tensor.sysMem));
  }
}

int get_element_size(int type) {
  switch (type) {
    case HB_DNN_TENSOR_TYPE_S8:
    case HB_DNN_TENSOR_TYPE_U8:
      return 1;
    case HB_DNN_TENSOR_TYPE_F16:
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
      return 2;
    case HB_DNN_TENSOR_TYPE_F32:
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
      return 4;
    case HB_DNN_TENSOR_TYPE_F64:
    case HB_DNN_TENSOR_TYPE_S64:
    case HB_DNN_TENSOR_TYPE_U64:
      return 8;
    default:
      VLOG(EXAMPLE_DETAIL)
          << "GetElementSize failed! input tensor type not support";
  }
  return -1;
}

void release_multi_tensor(std::vector<hbDNNTensor> &tensors) {
  for (auto tensor : tensors) {
    hbUCPFree(&(tensor.sysMem));
  }
}

void flush_multi_tensor(std::vector<hbDNNTensor> &tensors) {
  for (auto tensor : tensors) {
    hbUCPMemFlush(&(tensor.sysMem), HB_SYS_MEM_CACHE_CLEAN);
  }
}

void alloc_input_tenosr_mem(int input_count, hbDNNHandle_t dnn_handle,
                            hbDNNTensor *input_tensor) {
  hbDNNTensor *input = input_tensor;
  for (int i = 0; i < input_count; i++) {
    hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
    int aligned_size = input[i].properties.alignedByteSize;
    hbUCPMallocCached(&input[i].sysMem, aligned_size, 0);
  }
}

int prepare_batch_tensor(hbDNNTensor *tensor, std::string &file_path) {
  auto &tensor_property = tensor->properties;
  // read file length
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // read file into tensor memory
  auto &mem = tensor->sysMem;
  auto aligned_size =
      tensor_property.stride[0] * tensor_property.validShape.dimensionSize[0];
  if (length != aligned_size) {
    VLOG(EXAMPLE_SYSTEM) << "len " << length << " of file " << file_path
                         << " is not equal to " << aligned_size
                         << ", maybe need padding here";
    return -1;
  }
  hbUCPMallocCached(&mem, aligned_size, 0);
  ifs.read(reinterpret_cast<char *>(mem.virAddr), aligned_size);
  ifs.close();
  // flush
  hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

template <typename T>
int quanti_tensor(void **res, std::vector<float> &input, int *valid_shape,
                  int *aligned_shape, float *scale, float min_val,
                  float max_val) {
  int valid_n = valid_shape[0];
  int valid_c = valid_shape[1];
  int valid_h = valid_shape[2];
  int valid_w = valid_shape[3];

  int aligned_n = valid_shape[0];
  int aligned_c = aligned_shape[1];
  int aligned_h = aligned_shape[2];
  int aligned_w = aligned_shape[3];

  int count = 0;

  for (int n = 0; n < valid_n; n++) {
    int input_nchw = n * aligned_c * aligned_h * aligned_w;
    for (int c = 0; c < valid_c; c++) {
      int input_chw = input_nchw + c * aligned_h * aligned_w;
      for (int h = 0; h < valid_h; h++) {
        int input_hw = input_chw + h * aligned_w;
        for (int w = 0; w < valid_w; w++) {
          float tmp = std::floor(input[count++] / scale[0] + 0.5);
          reinterpret_cast<T *>(*res)[input_hw + w] =
              static_cast<T>(std::min(std::max(tmp, min_val), max_val));
        }
      }
    }
  }

  return 0;
}

int prepare_batch_tensor_and_quanti(hbDNNTensor *tensor,
                                    std::string &file_path) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto aligned_shape = properies2alignshape(tensor_property);

  // read file length
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // read file into tensor memory
  auto type = tensor_property.tensorType;
  auto &mem = tensor->sysMem;
  auto aligned_size = tensor_property.alignedByteSize;
  if (aligned_size < 0) {
    aligned_size =
        tensor_property.stride[0] * tensor_property.validShape.dimensionSize[0];
  }
  switch (type) {
    case HB_DNN_TENSOR_TYPE_S8: {
      auto valid_size = sizeof(int8_t);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(float);
      if (real_size * sizeof(int8_t) == valid_size) {
        HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                         "hbUCPMallocCached failed");
        memset(mem.virAddr, 0, aligned_size);
        std::vector<float> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);

        quanti_tensor<int8_t>(
            &mem.virAddr, tmp_data, tensor->properties.validShape.dimensionSize,
            aligned_shape.data(), tensor->properties.scale.scaleData, -128.0f,
            127.0f, tensor->properties.validShape.numDimensions);
      } else {
        VLOG(EXAMPLE_SYSTEM)
            << "input_file length does not match! file length: " << length
            << "; input valid size: " << valid_size
            << "; input aligned size: " << aligned_size;
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }
    case HB_DNN_TENSOR_TYPE_S16: {
      auto valid_size = sizeof(int16_t);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(float);
      if (real_size * sizeof(int16_t) == valid_size) {
        HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                         "hbUCPMallocCached failed");
        memset(mem.virAddr, 0, aligned_size);
        std::vector<float> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);

        quanti_tensor<int16_t>(
            &mem.virAddr, tmp_data, tensor->properties.validShape.dimensionSize,
            aligned_shape.data(), tensor->properties.scale.scaleData, -32768.0f,
            32767.0f, tensor->properties.validShape.numDimensions);
      } else {
        VLOG(EXAMPLE_SYSTEM)
            << "input_file length does not match! file length: " << length
            << "; input valid size: " << valid_size
            << "; input aligned size: " << aligned_size;
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }
    case HB_DNN_TENSOR_TYPE_BOOL8: {
      auto valid_size = sizeof(uint8_t);
      HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                       "hbUCPMallocCached failed");
      memset(mem.virAddr, 0, aligned_size);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(bool);
      if (real_size * sizeof(uint8_t) == valid_size) {
        std::vector<uint8_t> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);
        std::vector<uint32_t> data_dim(
            tensor->properties.validShape.numDimensions);
        std::vector<int64_t> stride(data_dim.size());
        for (int i = 0; i < data_dim.size(); i++) {
          data_dim[i] = valid_shape.dimensionSize[i];
          stride[i] = tensor_property.stride[i];
        }
        int32_t result = add_padding(tensor->sysMem.virAddr, tmp_data.data(),
                                     data_dim.size(), data_dim.data(),
                                     stride.data(), sizeof(uint8_t));
      } else {
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }
    case HB_DNN_TENSOR_TYPE_S64: {
      HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                       "hbUCPMallocCached failed");
      memset(mem.virAddr, 0, aligned_size);
      int32_t real_size = length / sizeof(int64_t);
      std::vector<int64_t> tmp_data(real_size);
      ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);
      std::vector<uint32_t> data_dim(
          tensor->properties.validShape.numDimensions);
      std::vector<int64_t> stride(data_dim.size());
      for (int i = 0; i < data_dim.size(); i++) {
        data_dim[i] = valid_shape.dimensionSize[i];
        stride[i] = tensor_property.stride[i];
      }
      add_padding(tensor->sysMem.virAddr, tmp_data.data(), data_dim.size(),
                  data_dim.data(), stride.data(), sizeof(int64_t));
      ifs.close();
      break;
    }
    default: {
      VLOG(EXAMPLE_SYSTEM)
          << "Quanti only support int8_t, int16_t, BOOL8, int64.";
      ifs.close();
      break;
    }
  }
  ifs.close();
  // flush
  hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

int prepare_temporal_tensor(hbDNNTensor *tensor) {
  // feature map need padding outsize
  auto &tensor_property = tensor->properties;
  auto &mem = tensor->sysMem;

  auto aligned_size =
      tensor_property.stride[0] * tensor_property.validShape.dimensionSize[0];

  int ret = hbUCPMallocCached(&mem, aligned_size, 0);
  if (ret) {
    return ret;
  }
  memset(tensor->sysMem.virAddr, 0, tensor->sysMem.memSize);
  flush_tensor(tensor);

  return 0;
}

std::vector<int32_t> properies2alignshape(
    const hbDNNTensorProperties &properties) {
  std::vector<int32_t> aligned_shape(properties.validShape.numDimensions);
  aligned_shape[0] = properties.validShape.dimensionSize[0];
  for (int i = 1; i < properties.validShape.numDimensions; ++i) {
    aligned_shape[i] = properties.stride[i - 1] / properties.stride[i];
  }
  return aligned_shape;
}

int prepare_batch_RelPos_and_quanti(hbDNNTensor *tensor, std::string &file_path,
                                    int frame_id) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto aligned_shape = properies2alignshape(tensor_property);

  // read file length
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int file_length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  int infer_fram_nums_ = WorkflowPlugin::GetInstance()->GetInferFramNums();
  if (infer_fram_nums_ == 0) {
    std::cerr << "frame_id cannot be zero." << std::endl;
    return -1;
  }

  int length = file_length / infer_fram_nums_;  // infer_fram_nums_;

  int valid_n = 0;
  int valid_c = 0;
  int valid_h = 0;
  int valid_w = 0;
  std::streamoff offset = 0;
  if (valid_shape.numDimensions >= 1) {
    valid_n = valid_shape.dimensionSize[0];
  }
  if (valid_shape.numDimensions >= 2) {
    valid_c = valid_shape.dimensionSize[1];
  }
  if (valid_shape.numDimensions >= 3) {
    valid_h = valid_shape.dimensionSize[2];
    if (valid_shape.numDimensions >= 4) {
      valid_w = valid_shape.dimensionSize[3];
      offset =
          frame_id * (valid_n * valid_c * valid_h * valid_w * sizeof(float));
    } else {
      offset = frame_id * (valid_n * valid_c * valid_h * sizeof(float));
    }
  }
  ifs.seekg(offset, std::ios::beg);
  // read file into tensor memory
  auto type = tensor_property.tensorType;
  auto &mem = tensor->sysMem;
  auto aligned_size = tensor_property.alignedByteSize;
  switch (type) {
    case HB_DNN_TENSOR_TYPE_S8: {
      auto valid_size = sizeof(int8_t);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(float);
      if (real_size * sizeof(int8_t) == valid_size) {
        HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                         "hbUCPMallocCached failed");
        memset(mem.virAddr, 0, aligned_size);
        std::vector<float> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);

        quanti_tensor<int8_t>(
            &mem.virAddr, tmp_data, tensor->properties.validShape.dimensionSize,
            aligned_shape.data(), tensor->properties.scale.scaleData, -128.0f,
            127.0f, tensor->properties.validShape.numDimensions);
      } else {
        VLOG(EXAMPLE_SYSTEM)
            << " input_file length does not match! bin file length";
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }
    case HB_DNN_TENSOR_TYPE_S16: {
      auto valid_size = sizeof(int16_t);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(float);
      if (real_size * sizeof(int16_t) == valid_size) {
        HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                         "hbUCPMallocCached failed");
        memset(mem.virAddr, 0, aligned_size);
        std::vector<float> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);

        quanti_tensor<int16_t>(
            &mem.virAddr, tmp_data, tensor->properties.validShape.dimensionSize,
            aligned_shape.data(), tensor->properties.scale.scaleData, -32768.0f,
            32767.0f, tensor->properties.validShape.numDimensions);
      } else {
        VLOG(EXAMPLE_SYSTEM)
            << "input_file length does not match! file length: " << length;
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }
    // The input includes a mask matrix as a boolean input.
    default: {
      VLOG(EXAMPLE_SYSTEM) << "Quanti only support int8_t, int16_t.";
      VLOG(EXAMPLE_SYSTEM) << "  Type value: " << type;
      break;
    }
  }
  ifs.close();
  // flush
  hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

int prepare_batch_temporal_mask(hbDNNTensor *tensor, std::string &file_path,
                                int frame_id) {
  auto &tensor_property = tensor->properties;
  auto &valid_shape = tensor_property.validShape;
  auto aligned_shape = properies2alignshape(tensor_property);

  // read file length
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int file_length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  int infer_fram_nums_ = WorkflowPlugin::GetInstance()->GetInferFramNums();
  if (infer_fram_nums_ == 0) {
    std::cerr << "frame_id cannot be zero." << std::endl;
    return -1;
  }

  int length = file_length / infer_fram_nums_;  // infer_fram_nums_;

  int valid_n = 0;
  int valid_c = 0;
  int valid_h = 0;
  int valid_w = 0;
  std::streamoff offset = 0;
  if (valid_shape.numDimensions >= 1) {
    valid_n = valid_shape.dimensionSize[0];
  }
  if (valid_shape.numDimensions >= 2) {
    valid_c = valid_shape.dimensionSize[1];
  }
  if (valid_shape.numDimensions >= 3) {
    valid_h = valid_shape.dimensionSize[2];
    if (valid_shape.numDimensions >= 4) {
      valid_w = valid_shape.dimensionSize[3];
      offset =
          frame_id * (valid_n * valid_c * valid_h * valid_w * sizeof(uint8_t));
    } else {
      offset = frame_id * (valid_n * valid_c * valid_h * sizeof(uint8_t));
    }
  }
  ifs.seekg(offset, std::ios::beg);
  // read file into tensor memory
  auto type = tensor_property.tensorType;
  auto &mem = tensor->sysMem;
  auto aligned_size = tensor_property.alignedByteSize;
  switch (type) {
    case HB_DNN_TENSOR_TYPE_BOOL8: {
      auto valid_size = sizeof(uint8_t);
      HB_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                       "hbUCPMallocCached failed");
      memset(mem.virAddr, 0, aligned_size);
      for (int32_t i{0}; i < valid_shape.numDimensions; i++) {
        valid_size *= valid_shape.dimensionSize[i];
      }
      int32_t real_size = length / sizeof(uint8_t);
      if (real_size * sizeof(uint8_t) == valid_size) {
        std::vector<uint8_t> tmp_data(real_size);
        ifs.read(reinterpret_cast<char *>(tmp_data.data()), length);
        std::vector<uint32_t> data_dim(
            tensor->properties.validShape.numDimensions);
        std::vector<int64_t> stride(data_dim.size());
        for (int i = 0; i < data_dim.size(); i++) {
          data_dim[i] = valid_shape.dimensionSize[i];
          stride[i] = tensor_property.stride[i];
        }
        int32_t result = add_padding(tensor->sysMem.virAddr, tmp_data.data(),
                                     data_dim.size(), data_dim.data(),
                                     stride.data(), sizeof(uint8_t));
      } else {
        VLOG(EXAMPLE_SYSTEM)
            << " input_file length does not match! bin file length";
        ifs.close();
        return -1;
      }
      ifs.close();
      break;
    }

    // The input includes a mask matrix as a boolean input.
    default: {
      VLOG(EXAMPLE_SYSTEM) << "Quanti only support int8_t, int16_t.";
      VLOG(EXAMPLE_SYSTEM) << "  Type value: " << type;
      break;
    }
  }
  ifs.close();
  // flush
  hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

void PrintFlattenVector(const std::vector<float> &vec, int valid_n, int valid_c,
                        int valid_h, int valid_w) {
  for (int n = 0; n < valid_n; ++n) {
    for (int c = 0; c < valid_c; ++c) {
      for (int h = 0; h < valid_h; ++h) {
        for (int w = 0; w < valid_w; ++w) {
          int index = n * valid_c * valid_h * valid_w + c * valid_h * valid_w +
                      h * valid_w + w;
          std::cout << vec[index] << " ";
        }
      }
      std::cout << "\n";
    }
  }
}
