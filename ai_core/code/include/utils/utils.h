// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_UTILS_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_UTILS_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "base/common_def.h"
#include "hobot/dnn/hb_dnn.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"

#define TEMPORAL_INPUT "temporal_input"

using GetOriginalCoordinateFunc =
    std::function<float(float, float, float, float)>;

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);

#define METHOD_CHECK_SUCCESS(value, errmsg, res)                     \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return res;                                                    \
    }                                                                \
  } while (0);

#define LOGE_AND_RETURN_IF_NULL(ptr, code)  \
  {                                         \
    if (ptr == nullptr) {                   \
      std::cout << #ptr " is null pointer"; \
      return code;                          \
    }                                       \
  }
/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @param[out] length: bin length
 * @return 0 if success otherwise -1
 */
int32_t read_binary_file(std::string &file_path, char **bin, int32_t *length);

/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @return 0 if success otherwise -1
 */
int32_t read_binary_file(std::string &file_path, char *bin);

/**
 * Parse data_type enum to string
 * @param[in] data_type enum
 * @return data_type string
 */
std::string data_type_enum_to_string(int32_t data_type);

/**
 * Get filename
 * @param path: file path
 * @return filename
 */
std::string get_file_name(std::string &path);

/**
 * Split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void split(std::string &str, char sep, std::vector<std::string> &tokens,
           int32_t limit = -1);

/**
 * Reverse split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void rsplit(std::string &str, char sep, std::vector<std::string> &tokens,
            int32_t limit = -1);

/**
 * NHWC to NCHW
 * @param[out] out_data0: channel/planar 0 data
 * @param[out] out_data1: channel/planar 1 data
 * @param[out] out_data2: channel/planar 2 data
 * @param[in] in_data:
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nhwc_to_nchw(uint8_t *out_data0, uint8_t *out_data1, uint8_t *out_data2,
                  uint8_t *in_data, int32_t height, int32_t width);
/**
 * NCHW to NHWC
 * @param[out] out_data:
 * @param[in] in_data0: channel/planar 0 data
 * @param[in] in_data1: channel/planar 1 data
 * @param[in] in_data2: channel/planar 2 data
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nchw_to_nhwc(uint8_t *out_data, uint8_t *in_data0, uint8_t *in_data1,
                  uint8_t *in_data2, int32_t height, int32_t width);

/**
 * Test whether it's the same shape
 * @param[in] lhs:
 * @param[in] rhs:
 * @return true if the shape is equal
 */
bool operator==(hbDNNTensorShape &lhs, hbDNNTensorShape &rhs);

/**
 * Quanti shift
 * @param[in] data
 * @param[in] shift
 * @return shift value
 */
float quanti_shift(int32_t data, uint32_t shift);

/**
 * Quanti scale
 * @param[in] data
 * @param[in] scale
 * @return scale value
 */
float quanti_scale(int32_t data, float scale);

/**
 * Read classification name list from config file
 * @param[in] cls_name_file classification name list file
 * @param[out] cls_names classification name list
 */
void initClsName(const std::string &cls_name_file,
                 std::vector<std::string> &cls_names);

/**
 * Dump rapidjson to string
 * @param[in] rapidjson::Value val
 * @return json string
 */
std::string json_to_string(rapidjson::Value &val);

/**
 * nhwc2nchw
 * @param[in] aligned_shape
 * @param[in] input
 * @param[out] output
 */
template <typename T>
void nhwc2nchw(int32_t *aligned_shape, T *input, T *output);

/**
 * nchw2nhwc this nchw_to_nhwc function should use vaildshape
 * @param[in] vaild_shape
 * @param[in] input
 * @param[out] output
 */
void nchw2nhwc(int32_t *vaild_shape, float *input, float *output);

/**
 * ponits_img2cam_3d struct, only support 3D
 */
typedef struct {
  float v[3];
} PointsImg2cam3D;

/**
 * Ponits_img to camera
 * @param[out] points3D::camera res,shape must be qual points
 * @param[in] points:: bbox_pred_res input
 * @param[in] cam2img
 */
void points_img2cam(std::vector<PointsImg2cam3D> &points3D,
                    std::vector<std::vector<float>> &points,
                    std::vector<std::vector<float>> &cam2img);
typedef struct {
  float v[2];
} Pointscam3D2Img;

/**
 * Ponits_camera to img
 * @param[out] points:: 2d res (N, 2)
 * @param[in] points3D::camera 3d input, points in shape (N, 3)
 * @param[in] cam2img
 * @param[in] whe_sub:: res whether sub 1
 */
void points_cam2img(std::vector<Pointscam3D2Img> &points,
                    std::vector<std::vector<float>> &points3D,
                    std::vector<std::vector<float>> &cam2img,
                    bool whe_sub = false);
/**
 * Convert the boxes to  in clockwise order, in the form of (x0y0z0, x0y0z1,
 * x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
 * @param[out] res::corners res (N,8,3)
 * @param[in] points:: (N,9)
 */
void cam2imgcorner(std::vector<std::vector<std::vector<float>>> &res,
                   std::vector<std::vector<float>> &points);

/**
 * box cxcywh to xyxy (another name: box_center_to_corner)
 * @param[out] box_cxcywh
 * @param[in] box_cxcywh
 */
void box_cxcywh_to_xyxy(std::vector<std::vector<float>> &box_cxcywh);

/**
 * inverse_sigmod
 * @param[out] res
 * @param[in] pred_boxes
 * @param[in] eps
 */
void inverse_sigmod(std::vector<float> &res,
                    std::vector<std::vector<float>> &pred_boxes,
                    float eps = 1e-5);

/**
 * im2col_process
 * @param[out] data_col
 * @param[in] data_im
 * @param[in] height
 * @param[in] width
 * @param[in] pad_w
 * @param[in] stride_h
 * @param[in] stride_w
 * @param[in] dilation_W
 * @param[in] kernel_col
 * @param[in] input_row
 * @param[in] output_w
 */
void im2col_process(float *&data_col, float *data_im, int32_t height,
                    int32_t width, int32_t pad_w, int32_t stride_h,
                    int32_t stride_w, int32_t dilation_w, int32_t kernel_col,
                    int32_t &input_row, int32_t output_w);

/**
 * im2col or unfold
 * @param[out] data_col
 * @param[in] data_im
 * @param[in] channels
 * @param[in] height
 * @param[in] width
 * @param[in] kernel_h
 * @param[in] kernel_w
 * @param[in] pad_h
 * @param[in] pad_w
 * @param[in] stride_h
 * @param[in] stride_w
 * @param[in] dilation_h
 * @param[in] dilation_w
 */
void im2col(float *data_col, float *data_im, int32_t channels, int32_t height,
            int32_t width, int32_t kernel_h, int32_t kernel_w, int32_t pad_h,
            int32_t pad_w, int32_t stride_h, int32_t stride_w,
            int32_t dilation_h, int32_t dilation_w);

/**
 * get original coordinate
 */
GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
    bool align_corners);

/**
 * bilinear upsample
 *
 */
int32_t bilinear_upsample(float *output_data, float *input_data,
                          int32_t channels, int32_t output_height,
                          int32_t output_width, int32_t input_height,
                          int32_t input_width, float scale_h, float scale_w,
                          GetOriginalCoordinateFunc get_original_coordinate);

/**
 * max_pool2d per channel
 * @param[out] out_data
 * @param[in] in_data
 * @param[in] ph: pooled height index
 * @param[in] pw: pooled width index
 * @param[in] c
 * @param[in] kernel
 * @param[in] stride
 * @param[in] pad
 * @param[in] height
 * @param[in] width
 * @param[in] pooled_height
 * @param[in] pooled_width
 */
template <typename T>
void max_pool2d_nchw_calculate(
    T *const out_data, T const *const in_data, int32_t const ph,
    int32_t const pw, uint32_t const c, std::vector<int32_t> const &kernel,
    std::vector<int32_t> const &stride, std::vector<int32_t> const &pad,
    int32_t const height, int32_t const width, int32_t const pooled_height,
    int32_t const pooled_width) {
  static_cast<void>(pooled_height);
  int32_t const kernel_h{kernel[0U]};
  int32_t const kernel_w{kernel[1U]};
  int32_t const pad_h{pad[0U]};
  int32_t const pad_w{pad[1U]};
  int32_t const stride_h{stride[0U]};
  int32_t const stride_w{stride[1U]};
  uint32_t const in_data_offset{static_cast<uint32_t>(height) *
                                static_cast<uint32_t>(width)};

  int32_t hstart{ph * stride_h - pad_h};
  int32_t wstart{pw * stride_w - pad_w};
  int32_t const hend{std::min(hstart + kernel_h, height)};
  int32_t const wend{std::min(wstart + kernel_w, width)};
  hstart = std::max(hstart, 0);
  wstart = std::max(wstart, 0);
  int32_t const pool_index{ph * pooled_width + pw};
  T max_val{std::numeric_limits<T>::lowest()};
  int64_t h_index{-1};
  int64_t w_index{-1};
  for (int32_t h{hstart}; h < hend; ++h) {
    for (int32_t w{wstart}; w < wend; ++w) {
      int32_t const in_index{h * width + w};
      if (in_data[in_index] > max_val) {
        max_val = in_data[in_index];
        h_index = h;
        w_index = w;
      }
    }
  }
  out_data[pool_index] = max_val;
}

int32_t add_padding(void *output, const void *input, uint32_t dim_num,
                    const uint32_t *dim, const int64_t *stride,
                    uint32_t element_size);

void add_padding_core(void *output_ptr, const void *input_ptr, uint32_t dim_num,
                      const uint32_t *dim, const int64_t *stride,
                      uint32_t element_size);
template <typename T>
int32_t get_prod_size(T const *dim, uint32_t dim_num);
/**
 * max_pool2d for nchw
 * @param[out] out_data
 * @param[in] in_data
 * @param[in] shape: valid shape
 * @param[in] kernel
 * @param[in] stride
 * @param[in] pad
 */
template <typename T>
void max_pool2d_nchw(T *out_data, T const *in_data, int32_t const *shape,
                     std::vector<int32_t> const &kernel,
                     std::vector<int32_t> const &stride,
                     std::vector<int32_t> const &pad) {
  int32_t const height{static_cast<int32_t>(shape[2U])};
  int32_t const width{static_cast<int32_t>(shape[3U])};
  int32_t const pooled_height{static_cast<int32_t>(shape[2U])};
  int32_t const pooled_width{static_cast<int32_t>(shape[3U])};

  int32_t const in_data_offset{shape[2U] * shape[3U]};
  int32_t const out_data_offset{shape[2U] * shape[3U]};
  int32_t const total_channels{shape[0U] * shape[1U]};

  for (uint32_t c{0U}; c < total_channels; ++c) {
    for (int32_t ph{0}; ph < pooled_height; ++ph) {
      for (int32_t pw{0}; pw < pooled_width; ++pw) {
        max_pool2d_nchw_calculate(out_data, in_data, ph, pw, c, kernel, stride,
                                  pad, height, width, pooled_height,
                                  pooled_width);
      }
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/**
 * Read data from file
 * @param[out] res
 * @param[in] file_name
 */
void LoadFromFile(std::vector<float> &res, std::string file_name);

/**
 * transpose array，transpose(2,1,0)
 * @param[out] transposedArray
 * @param[in] originalArray
 * @param[in] sizeX
 * @param[in] sizeY
 * @param[in] sizeZ
 */
template <typename T>
void transposeArray(T *transposedArray, const T *originalArray, int sizeX,
                    int sizeY, int sizeZ) {
  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      for (int z = 0; z < sizeZ; ++z) {
        int originalIndex = x * sizeY * sizeZ + y * sizeZ + z;
        int transposedIndex = z * sizeX * sizeY + y * sizeX + x;
        transposedArray[transposedIndex] = originalArray[originalIndex];
      }
    }
  }
}

void bilinear_upsample_w(
    float *const dx1, float *const dx2, uint64_t *const in_x1,
    uint64_t *const in_x2, std::vector<float> &x_original,
    int32_t const input_width, int32_t const output_width, float const scale_w,
    GetOriginalCoordinateFunc const &get_original_coordinate);

void bilinear_upsample_h(
    uint64_t *const input_width_mul_y1, uint64_t *const input_width_mul_y2,
    float *const dy1, float *const dy2, std::vector<float> &y_original,
    int32_t const output_height, int32_t const output_width,
    int32_t const input_height, int32_t const input_width, float const scale_h,
    GetOriginalCoordinateFunc const &get_original_coordinate);

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_UTILS_H_
