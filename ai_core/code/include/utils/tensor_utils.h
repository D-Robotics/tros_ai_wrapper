// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_TENSOR_UTILS_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_TENSOR_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"
#include "input/input_data.h"
#include "utils/data_transformer.h"

int32_t check_model_input(std::vector<hbDNNTensor> &model_input_tensors,
                          std::vector<bool> is_pyramid_input);

int32_t prepare_pyramid_input_tensor(ImageTensor *image_tensor,
                                     int32_t y_index);

void fill_image_to_tensor(std::string &path, ImageTensor *image_tensor,
                          int32_t y_index, transformers_func transformers);

void fill_preprocessed_image_to_tensor(ImageTensor *image_tensor, char *data,
                                       int32_t y_index);

int32_t add_padding(void *output, const void *input, uint32_t dim_num,
                    const uint32_t *dim, const int64_t *stride,
                    uint32_t element_size);

void add_padding_core(void *output_ptr, const void *input_ptr, uint32_t dim_num,
                      const uint32_t *dim, const int64_t *stride,
                      uint32_t element_size);

/**
 * Align by 32
 */

#define ALIGN_32(v) ((v + (32 - 1)) / 32 * 32)

/**
 * Flush tensor
 * @param[in] tensor: Tensor to be flushed
 */
void flush_tensor(hbDNNTensor *tensor);

/**
 * Free tensor
 * @param tensor: Tensor to be released
 */
void release_tensor(hbDNNTensor *tensor);

/**
 * Prepare output tensor
 * @param output
 * @param model
 */
void prepare_output_tensor(std::vector<hbDNNTensor> &output,
                           hbDNNHandle_t dnn_handle);

/**
 * Release output tensor
 * @param output
 */
void release_output_tensor(std::vector<hbDNNTensor> &output);

/**
 * release multi tensor
 * @param[out] output
 */
void release_multi_tensor(std::vector<hbDNNTensor> &tensors);

/**
 * flush multi tensor
 * @param[out] output
 */
void flush_multi_tensor(std::vector<hbDNNTensor> &tensors);

/**
 * prepare input tenosr
 * @param[out] output
 */
void alloc_input_tenosr_mem(int input_count, hbDNNHandle_t dnn_handle,
                            hbDNNTensor *input_tensor);

/**
 * get element size
 * @param[out] type
 */
int get_element_size(int type);

/**
 * prepare input tensor
 * @param[out] tensor
 * @param[in] file_path
 */
int prepare_batch_tensor(hbDNNTensor *tensor, std::string &file_path);

/**
 * quanti input tensor
 * @param[out/in] input
 * @param[in] scale
 */
template <typename T>
int quanti_tensor(void **res, std::vector<float> &input, int *valid_shape,
                  int *aligned_shape, float *scale, float min_val,
                  float max_val);

/**
 * quanti input tensor
 * @param[out/in] input
 * @param[in] scale
 */
template <typename T>
int quanti_tensor(void **res, const std::vector<float> &input,
                  const int *valid_shape, const int *aligned_shape,
                  const float *scale, float min_val, float max_val,
                  int valid_shape_size) {
  std::vector<int> valid_strides(valid_shape_size, 1);
  std::vector<int> aligned_strides(valid_shape_size, 1);
  for (int i = valid_shape_size - 2; i >= 0; --i) {
    valid_strides[i] = valid_strides[i + 1] * valid_shape[i + 1];
  }
  for (int i = valid_shape_size - 2; i >= 0; --i) {
    aligned_strides[i] = aligned_strides[i + 1] * aligned_shape[i + 1];
  }

  int total_aligned_size = 1;
  for (int i = 0; i < valid_shape_size; ++i) {
    total_aligned_size *= aligned_shape[i];
  }

  T *output_data = reinterpret_cast<T *>(*res);

  for (int i = 0; i < total_aligned_size; ++i) {
    int input_index = 0;
    int output_index = i;
    bool is_valid = true;

    for (int j = 0; j < valid_shape_size; ++j) {
      int idx = output_index / aligned_strides[j];
      output_index %= aligned_strides[j];
      if (idx >= valid_shape[j]) {
        is_valid = false;
        break;
      }
      input_index += idx * valid_strides[j];
    }

    if (is_valid && input_index < input.size()) {
      float tmp = std::floor(input[input_index] / scale[0] + 0.5f);
      output_data[i] =
          static_cast<T>(std::min(std::max(tmp, min_val), max_val));
    }
  }

  return 0;
}

/**
 * prepare input tensor and quanti, only suppory float32 origin data
 * @param[out] tensor
 * @param[in] file_path
 */
int prepare_batch_tensor_and_quanti(hbDNNTensor *tensor,
                                    std::string &file_path);

/**
 * prepare input tensor
 * @param[out] tensor
 */
int prepare_temporal_tensor(hbDNNTensor *tensor);

/**
 * prepare input RelPos tensor
 * @param[out] tensor
 */
int prepare_batch_RelPos_and_quanti(hbDNNTensor *tensor, std::string &file_path,
                                    int frame_id);
int prepare_batch_temporal_mask(hbDNNTensor *tensor, std::string &file_path,
                                int frame_id);
void PrintFlattenVector(const std::vector<float> &vec, int valid_n, int valid_c,
                        int valid_h, int valid_w);
/**
 * prepare input tensor
 * @param[in] properties
 * @return aligned shape
 */
std::vector<int32_t> properies2alignshape(
    const hbDNNTensorProperties &properties);

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_TENSOR_UTILS_H_
