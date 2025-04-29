// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "glog/logging.h"

int read_binary_file(std::string &file_path, char **bin, int *length) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  *length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  *bin = new char[sizeof(char) * (*length)];
  ifs.read(*bin, *length);
  ifs.close();
  return 0;
}

int read_binary_file(std::string &file_path, char *bin) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    VLOG(EXAMPLE_SYSTEM) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  ifs.read(bin, length);
  ifs.close();
  return 0;
}

std::string data_type_enum_to_string(int32_t data_type) {
  switch (data_type) {
    case HB_DNN_TENSOR_TYPE_U8:
      return "HB_DNN_TENSOR_TYPE_U8";
    case HB_DNN_TENSOR_TYPE_S8:
      return "HB_DNN_TENSOR_TYPE_S8";
    case HB_DNN_TENSOR_TYPE_F32:
      return "HB_DNN_TENSOR_TYPE_F32";
    case HB_DNN_TENSOR_TYPE_S32:
      return "HB_DNN_TENSOR_TYPE_S32";
    case HB_DNN_TENSOR_TYPE_U32:
      return "HB_DNN_TENSOR_TYPE_U32";
    case HB_DNN_TENSOR_TYPE_MAX:
    default:
      return "HB_DNN_TENSOR_TYPE_MAX";
  }
}

std::string get_file_name(std::string &path) {
  int slash_pos = path.rfind('/');
  return path.substr(slash_pos + 1);
}

void split(std::string &str, char sep, std::vector<std::string> &tokens,
           int limit) {
  int pos = -1;
  while (true) {
    int next_pos = str.find(sep, pos + 1);
    if (next_pos == std::string::npos) {
      tokens.emplace_back(str.substr(pos + 1));
      break;
    }
    tokens.emplace_back(str.substr(pos + 1, next_pos - pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(next_pos + 1));
      break;
    }
    pos = next_pos;
  }
}

void rsplit(std::string &str, char sep, std::vector<std::string> &tokens,
            int limit) {
  int pos = str.size();
  while (true) {
    int prev_pos = str.rfind(sep, pos - 1);
    if (prev_pos == std::string::npos) {
      tokens.emplace_back(str.substr(0, pos));
      break;
    }

    tokens.emplace_back(str.substr(prev_pos + 1, pos - prev_pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(0, prev_pos));
      break;
    }

    pos = prev_pos;
  }
}

void nhwc_to_nchw(uint8_t *out_data0, uint8_t *out_data1, uint8_t *out_data2,
                  uint8_t *in_data, int height, int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = *(in_data++);
      *out_data1++ = *(in_data++);
      *out_data2++ = *(in_data++);
    }
  }
}

void nchw_to_nhwc(uint8_t *out_data, uint8_t *in_data0, uint8_t *in_data1,
                  uint8_t *in_data2, int height, int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data++ = *(in_data0++);
      *out_data++ = *(in_data1++);
      *out_data++ = *(in_data2++);
    }
  }
}

bool operator==(hbDNNTensorShape &lhs, hbDNNTensorShape &rhs) {
  if (lhs.numDimensions != rhs.numDimensions) return false;
  for (int i = 0; i < lhs.numDimensions; i++) {
    if (lhs.dimensionSize[i] != rhs.dimensionSize[i]) {
      return false;
    }
  }
  return true;
}

float quanti_shift(int32_t data, uint32_t shift) {
  return static_cast<float>(data) / static_cast<float>(1 << shift);
}

float quanti_scale(int32_t data, float scale) { return data * scale; }

void initClsName(const std::string &cls_name_file,
                 std::vector<std::string> &cls_names) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    std::string line;
    while (std::getline(fi, line)) {
      cls_names.push_back(line);
    }
  } else {
    VLOG(EXAMPLE_SYSTEM) << "can not open cls name file";
  }
}

std::string json_to_string(rapidjson::Value &val) {
  rapidjson::StringBuffer buffer;
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  val.Accept(writer);
  return buffer.GetString();
}

// nhwc_to_nchw
template <typename T>
void nhwc2nchw(int *aligned_shape, T *input, T *output) {
  int N = aligned_shape[0];
  int H = aligned_shape[1];
  int W = aligned_shape[2];
  int C = aligned_shape[3];
  for (int n = 0; n < aligned_shape[0]; ++n) {
    for (int c = 0; c < aligned_shape[3]; ++c) {
      for (int h = 0; h < aligned_shape[1]; ++h) {
        for (int w = 0; w < aligned_shape[2]; ++w) {
          int old_index = n * H * W * C + h * W * C + w * C + c;
          int new_index = n * C * H * W + c * H * W + h * W + w;
          output[new_index] = input[old_index];
        }
      }
    }
  }
}

void nchw2nhwc(int *vaild_shape, float *input, float *output) {
  int N = vaild_shape[0];
  int H = vaild_shape[1];
  int W = vaild_shape[2];
  int C = vaild_shape[3];
  for (int n = 0; n < vaild_shape[0]; ++n) {
    for (int c = 0; c < vaild_shape[3]; ++c) {
      for (int h = 0; h < vaild_shape[1]; ++h) {
        for (int w = 0; w < vaild_shape[2]; ++w) {
          int old_index = n * H * W * C + h * W * C + w * C + c;
          int new_index = n * C * H * W + c * H * W + h * W + w;
          output[old_index] = input[new_index];
        }
      }
    }
  }
}

inline std::vector<std::vector<float>> torch_eye(int N) {
  std::vector<std::vector<float>> res(N, std::vector<float>(N, 0));
  for (int i = 0; i < N; i++) {
    res[i][i] = 1;
  }
  return res;
}

struct MatInverse {
  int m = 4, n = 4;
  float mat[10][10] = {0};

  MatInverse() {}
  MatInverse(int mm, int nn) {
    m = mm;
    n = nn;
  }
  void create(std::vector<std::vector<float>> &v);
  void eye();

  bool inv(MatInverse a);
  std::vector<std::vector<float>> inv_vector();
};

void MatInverse::create(std::vector<std::vector<float>> &v) {
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      mat[i][j] = v[i - 1][j - 1];
    }
  }
}

void MatInverse::eye() {
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      if (i == j)
        mat[i][j] = 1;
      else
        mat[i][j] = 0;
    }
  }
}

bool MatInverse::inv(MatInverse a) {
  if (a.n != a.m) {
    VLOG(EXAMPLE_SYSTEM) << "The matrix is not reversible.";
    return false;
  }
  m = a.m;
  n = a.n;
  eye();

  for (int i = 1; i <= m; i++) {
    int k;
    for (k = i; k <= m; k++) {
      if (fabs(a.mat[k][i]) > 1e-10) break;
    }
    if (k <= m) {
      if (k != i) {
        for (int j = 1; j <= n; j++) {
          a.mat[0][j] = a.mat[i][j];
          a.mat[i][j] = a.mat[k][j];
          a.mat[k][j] = a.mat[0][j];
          mat[0][j] = mat[i][j];
          mat[i][j] = mat[k][j];
          mat[k][j] = mat[0][j];
        }
      }
      double b = a.mat[i][i];
      for (int j = 1; j <= n; j++) {
        a.mat[i][j] /= b;
        mat[i][j] /= b;
      }
      for (int j = i + 1; j <= m; j++) {
        b = -a.mat[j][i];
        for (k = 1; k <= n; k++) {
          a.mat[j][k] += b * a.mat[i][k];
          mat[j][k] += b * mat[i][k];
        }
      }
    } else {
      VLOG(EXAMPLE_SYSTEM) << "The matrix is not reversible.";
      return false;
    }
  }

  for (int i = m; i > 1; i--) {
    for (int j = i - 1; j >= 1; j--) {
      double b = -a.mat[j][i];
      a.mat[j][i] = 0;
      for (int k = 1; k <= n; k++) {
        mat[j][k] += b * mat[i][k];
      }
    }
  }
  return true;
}

std::vector<std::vector<float>> MatInverse::inv_vector() {
  std::vector<std::vector<float>> res(m, std::vector<float>(n, 0));
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      res[i - 1][j - 1] = mat[i][j];
    }
  }
  return res;
}

void points_img2cam(std::vector<PointsImg2cam3D> &points3D,
                    std::vector<std::vector<float>> &points,
                    std::vector<std::vector<float>> &cam2img) {
  // cam2img shape can be [3, 3], [3, 4] or [4, 4], this only support [3, 3]
  assert(cam2img.size() == cam2img[0].size());
  assert(cam2img.size() <= 4);
  assert(cam2img[0].size() <= 4);
  assert(cam2img.size() == 3);
  // assert(points[0].size() == 3); index only can use 0,1,2
  int max_index = 3;
  std::vector<std::vector<float>> xys(points.size(), std::vector<float>(2, 0));

  std::vector<std::vector<float>> depths(points.size(),
                                         std::vector<float>(1, 0));
  for (int i = 0; i < points.size(); i++) {
    xys[i][0] = points[i][0];
    xys[i][1] = points[i][1];
    depths[i][0] = points[i][2];
  }
  std::vector<std::vector<float>> unnormed_xys(points.size(),
                                               std::vector<float>(3, 0));

  for (int i = 0; i < points.size(); i++) {
    for (int j = 0; j < 3; j++) {
      if (j < 2)
        unnormed_xys[i][j] = xys[i][j] * depths[i][0];
      else
        unnormed_xys[i][j] = depths[i][0];
    }
  }

  std::vector<std::vector<float>> pad_cam2img{torch_eye(4)};
  for (int i = 0; i < cam2img.size(); i++) {
    for (int j = 0; j < cam2img[0].size(); j++) {
      pad_cam2img[i][j] = cam2img[i][j];
    }
  }

  std::vector<std::vector<float>> pad_cam2img_inverse{4,
                                                      std::vector<float>(4, 0)};
  MatInverse tmp_pad_cam2img_inverse(4, 4);
  tmp_pad_cam2img_inverse.create(pad_cam2img);

  MatInverse tmp_pad_cam2img_inverse2;
  tmp_pad_cam2img_inverse2.inv(tmp_pad_cam2img_inverse);
  pad_cam2img_inverse = tmp_pad_cam2img_inverse2.inv_vector();

  // transpose (4,4)
  std::vector<std::vector<float>> inv_pad_cam2img(
      pad_cam2img_inverse[0].size(),
      std::vector<float>(pad_cam2img_inverse.size(), 0));
  for (int i{0}; i < pad_cam2img_inverse[0].size(); i++) {
    for (int j{0}; j < pad_cam2img_inverse.size(); j++) {
      inv_pad_cam2img[j][i] = pad_cam2img_inverse[i][j];
    }
  }

  // Do operation in homogeneous coordinates.
  int num_points = unnormed_xys.size();
  std::vector<std::vector<float>> homo_xys(points.size(),
                                           std::vector<float>(4, 0));
  for (int i = 0; i < points.size(); i++) {
    for (int j = 0; j < 4; j++) {
      if (j < 3)
        homo_xys[i][j] = unnormed_xys[i][j];
      else
        homo_xys[i][j] = 1;
    }
  }

  // torch.mm
  for (int i = 0; i < points.size(); i++) {
    points3D[i].v[0] = homo_xys[i][0] * inv_pad_cam2img[0][0] +
                       homo_xys[i][1] * inv_pad_cam2img[1][0] +
                       homo_xys[i][2] * inv_pad_cam2img[2][0] +
                       homo_xys[i][3] * inv_pad_cam2img[3][0];
    points3D[i].v[1] = homo_xys[i][0] * inv_pad_cam2img[0][1] +
                       homo_xys[i][1] * inv_pad_cam2img[1][1] +
                       homo_xys[i][2] * inv_pad_cam2img[2][1] +
                       homo_xys[i][3] * inv_pad_cam2img[3][1];
    points3D[i].v[2] = homo_xys[i][0] * inv_pad_cam2img[0][2] +
                       homo_xys[i][1] * inv_pad_cam2img[1][2] +
                       homo_xys[i][2] * inv_pad_cam2img[2][2] +
                       homo_xys[i][3] * inv_pad_cam2img[3][2];
  }
}

void points_cam2img(std::vector<Pointscam3D2Img> &points,
                    std::vector<std::vector<float>> &points3D,
                    std::vector<std::vector<float>> &cam2img, bool whe_sub) {
  int N = points3D.size();

  std::vector<std::vector<float>> proj_mat_t(4, std::vector<float>(4, 0));
  std::vector<std::vector<float>> points_4(N, std::vector<float>(4, 0));

  // transpose format
  for (int i = 0; i < 3; i++) {
    // generate eye
    proj_mat_t[i][i] = 1;
    // cover cam2img value
    proj_mat_t[0][i] = cam2img[i][0];
    proj_mat_t[1][i] = cam2img[i][1];
    proj_mat_t[2][i] = cam2img[i][2];
  }
  proj_mat_t[3][3] = 1;

  for (int i = 0; i < N; i++) {
    points_4[i][0] = points3D[i][0];
    points_4[i][1] = points3D[i][1];
    points_4[i][2] = points3D[i][2];
    points_4[i][3] = 1;
  }

  std::vector<std::vector<float>> point_2d(N, std::vector<float>(4, 0));

  for (int i = 0; i < N; i++) {
    // 'point_2d[i][3]' don't take part in the computation of points,
    // so do not compute
    for (int col = 0; col < 3; col++) {
      point_2d[i][col] = points_4[i][0] * proj_mat_t[0][col] +
                         points_4[i][1] * proj_mat_t[1][col] +
                         points_4[i][2] * proj_mat_t[2][col] +
                         points_4[i][3] * proj_mat_t[3][col];
    }

    if (!whe_sub) {
      points[i].v[0] = point_2d[i][0] / point_2d[i][2];
      points[i].v[1] = point_2d[i][1] / point_2d[i][2];
    } else {
      points[i].v[0] = static_cast<int>(point_2d[i][0] / point_2d[i][2] - 1);
      points[i].v[1] = static_cast<int>(point_2d[i][1] / point_2d[i][2] - 1);
    }
  }
}

void cam2imgcorner(std::vector<std::vector<std::vector<float>>> &res,
                   std::vector<std::vector<float>> &points) {
  // Convert the boxes to  in clockwise order, in the form of
  // (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
  //                 front z
  //                     /
  //                     /
  //       (x0, y0, z1) + -----------  + (x1, y0, z1)
  //                   /|            / |
  //                 / |           /  |
  //   (x0, y0, z0) + ----------- +   + (x1, y1, z1)
  //                 |  /      .   |  /
  //                 | / origin    | /
  //   (x0, y1, z0) + ----------- + -------> x right
  //                 |             (x1, y1, z0)
  //                 |
  //                 v
  //           down y

  // use relative origin [0.5, 1, 0.5],if origin != [0.5, 1, 0.5], corners_norm
  // change
  std::vector<std::vector<float>> corners_norm = {
      {-0.5000, -1.0000, -0.5000}, {-0.5000, -1.0000, 0.5000},
      {-0.5000, 0.0000, 0.5000},   {-0.5000, 0.0000, -0.5000},
      {0.5000, -1.0000, -0.5000},  {0.5000, -1.0000, 0.5000},
      {0.5000, 0.0000, 0.5000},    {0.5000, 0.0000, -0.5000}};

  std::vector<std::vector<std::vector<float>>> corners(points.size(),
                                                       corners_norm);

  std::vector<float> angles(points.size(), 0);
  for (int i = 0; i < points.size(); i++) {
    // process corners
    for (int j = 0; j < 8; j++) {
      corners[i][j][0] = points[i][3] * corners_norm[j][0];
      corners[i][j][1] = points[i][4] * corners_norm[j][1];
      corners[i][j][2] = points[i][5] * corners_norm[j][2];
    }
    // process angles
    // angles[i] = points[i][6];
    float rot_sin = std::sin(points[i][6]);
    float rot_cos = std::cos(points[i][6]);
    std::vector<std::vector<float>> rot_mat = {
        {rot_cos, 0, -rot_sin}, {0, 1, 0}, {rot_sin, 0, rot_cos}};
    std::vector<std::vector<float>> tmp_res{corners_norm};

    // matrix_mul
    int height = tmp_res.size();
    int width = tmp_res[0].size();
    int common_length = rot_mat.size();
    for (int ii = 0; ii < height; ++ii) {
      for (int jj = 0; jj < width; ++jj) {
        float sum = 0;
        for (int c = 0; c < common_length; ++c) {
          sum += corners[i][ii][c] * rot_mat[c][jj];
        }
        tmp_res[ii][jj] = sum;
      }
    }

    for (int tmp_res_row = 0; tmp_res_row < tmp_res.size(); tmp_res_row++) {
      tmp_res[tmp_res_row][0] += points[i][0];
      tmp_res[tmp_res_row][1] += points[i][1];
      tmp_res[tmp_res_row][2] += points[i][2];
    }
    res[i] = tmp_res;
  }
}

void box_cxcywh_to_xyxy(std::vector<std::vector<float>> &box_cxcywh) {
  for (int i{0}; i < box_cxcywh.size(); i++) {
    float w = box_cxcywh[i][2];
    float h = box_cxcywh[i][3];
    box_cxcywh[i][0] = box_cxcywh[i][0] - 0.5 * w;
    box_cxcywh[i][1] = box_cxcywh[i][1] - 0.5 * h;
    box_cxcywh[i][2] = box_cxcywh[i][0] + w;
    box_cxcywh[i][3] = box_cxcywh[i][1] + h;
  }
}

void inverse_sigmod(std::vector<float> &res,
                    std::vector<std::vector<float>> &pred_boxes, float eps) {
  int count = 0;
  for (int i = 0; i < pred_boxes.size(); i++) {
    for (int j = 0; j < pred_boxes[0].size(); j++) {
      float x = pred_boxes[i][j];
      if (x < 0) x = 0;
      if (x > 1) x = 1;
      float x1 = x;
      if (x1 < eps) x1 = eps;
      float x2 = (1 - x);
      if (x2 < eps) x2 = eps;
      res[count++] = std::log(x1 / x2);
    }
  }
}

GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
    bool align_corners) {
  if (align_corners) {
    return [](float const x_resized, float, float const length_resized,
              float const length_original) {
      return (std::fabs(length_resized - 1.f) < FLT_EPSILON)
                 ? 0.0f
                 : (x_resized * (length_original - 1.f) /
                    (length_resized - 1.f));
    };
  } else {  // "pytorch_half_pixel"
    return [](float const x_resized, float const x_scale,
              float const length_resized, float) {
      return (length_resized > 1.0f) ? ((x_resized + 0.5f) / x_scale - 0.5f)
                                     : 0.0f;
    };
  }
}

void bilinear_upsample_h(
    uint64_t *const input_width_mul_y1, uint64_t *const input_width_mul_y2,
    float *const dy1, float *const dy2, std::vector<float> &y_original,
    int32_t const output_height, int32_t const output_width,
    int32_t const input_height, int32_t const input_width, float const scale_h,
    GetOriginalCoordinateFunc const &get_original_coordinate) {
  static_cast<void>(output_width);
  for (uint64_t y = 0; y < static_cast<size_t>(output_height); ++y) {
    float in_y{(std::fabs(scale_h - 1.f) < FLT_EPSILON)
                   ? static_cast<float>(y)
                   : get_original_coordinate(static_cast<float>(y), scale_h,
                                             static_cast<float>(output_height),
                                             static_cast<float>(input_height))};
    y_original.emplace_back(in_y);
    in_y =
        std::max(0.0f, std::min(in_y, static_cast<float>(input_height) - 1.f));
    uint64_t const in_y1{std::min(static_cast<uint64_t>(in_y),
                                  static_cast<uint64_t>(input_height) - 1)};
    uint64_t const in_y2{
        std::min(in_y1 + 1, static_cast<uint64_t>(input_height) - 1)};
    dy1[y] = std::fabs(in_y - static_cast<float>(in_y1));
    dy2[y] = std::fabs(in_y - static_cast<float>(in_y2));
    if (in_y1 == in_y2) {
      dy1[y] = 0.5f;
      dy2[y] = 0.5f;
    }
    input_width_mul_y1[y] = static_cast<uint64_t>(input_width) * in_y1;
    input_width_mul_y2[y] = static_cast<uint64_t>(input_width) * in_y2;
  }
}

void bilinear_upsample_w(
    float *const dx1, float *const dx2, uint64_t *const in_x1,
    uint64_t *const in_x2, std::vector<float> &x_original,
    int32_t const input_width, int32_t const output_width, float const scale_w,
    GetOriginalCoordinateFunc const &get_original_coordinate) {
  for (int64_t x{0}; x < output_width; ++x) {
    float in_x{(std::fabs(scale_w - 1.f) < FLT_EPSILON)
                   ? static_cast<float>(x)
                   : get_original_coordinate(static_cast<float>(x), scale_w,
                                             static_cast<float>(output_width),
                                             static_cast<float>(input_width))};
    x_original.emplace_back(in_x);
    in_x =
        std::max(0.0f, std::min(in_x, static_cast<float>(input_width) - 1.f));
    in_x1[x] = std::min(static_cast<uint64_t>(in_x),
                        static_cast<uint64_t>(input_width) - 1);
    in_x2[x] = std::min(in_x1[x] + 1, static_cast<uint64_t>(input_width) - 1);
    dx1[x] = std::abs(in_x - static_cast<float>(in_x1[x]));
    dx2[x] = std::abs(in_x - static_cast<float>(in_x2[x]));
    if (in_x1[x] == in_x2[x]) {
      dx1[x] = 0.5f;
      dx2[x] = 0.5f;
    }
  }
}

int32_t bilinear_upsample(float *output_data, float *input_data,
                          int32_t channels, int32_t output_height,
                          int32_t output_width, int32_t input_height,
                          int32_t input_width, float scale_h, float scale_w,
                          GetOriginalCoordinateFunc get_original_coordinate) {
  std::vector<float> y_original{};
  std::vector<float> x_original{};
  size_t const idx_buffer_size{
      2U * sizeof(uint64_t) *
      (static_cast<size_t>(output_height) + static_cast<size_t>(output_width))};
  size_t const scale_buffer_size{
      2U * sizeof(float) *
      (static_cast<size_t>(output_height) + static_cast<size_t>(output_width))};

  std::vector<uint64_t> inx_scale_data_buffer(
      idx_buffer_size + scale_buffer_size, 0);
  uint64_t *const idx_data{inx_scale_data_buffer.data()};
  uint64_t *const input_width_mul_y1{idx_data};
  uint64_t *const input_width_mul_y2{idx_data + output_height};
  uint64_t *const in_x1{idx_data + 2 * output_height};
  uint64_t *const in_x2{idx_data + 2 * output_height + output_width};
  float *const scale_data{reinterpret_cast<float *>(in_x2 + output_width)};
  float *const dy1{scale_data};
  float *const dy2{scale_data + output_height};
  float *const dx1{scale_data + 2 * output_height};
  float *const dx2{scale_data + 2 * output_height + output_width};

  bilinear_upsample_h(input_width_mul_y1, input_width_mul_y2, dy1, dy2,
                      y_original, output_height, output_width, input_height,
                      input_width, scale_h, get_original_coordinate);

  bilinear_upsample_w(dx1, dx2, in_x1, in_x2, x_original, input_width,
                      output_width, scale_w, get_original_coordinate);

  for (int64_t c{0}; c < channels; ++c) {
    for (int64_t y{0}; y < output_height; ++y) {
      for (int64_t x{0}; x < output_width; ++x) {
        float const x_11{input_data[input_width_mul_y1[y] + in_x1[x]]};
        float const x_21{input_data[input_width_mul_y1[y] + in_x2[x]]};
        float const x_12{input_data[input_width_mul_y2[y] + in_x1[x]]};
        float const x_22{input_data[input_width_mul_y2[y] + in_x2[x]]};
        output_data[output_width * y + x] =
            static_cast<float>(dx2[x] * dy2[y] * x_11 + dx1[x] * dy2[y] * x_21 +
                               dx2[x] * dy1[y] * x_12 + dx1[x] * dy1[y] * x_22);
      }
    }
    input_data += input_height * input_width;
    output_data += output_height * output_width;
  }

  return 0;
}

void LoadFromFile(std::vector<float> &res, std::string file_name) {
  std::ifstream file(file_name, std::ios::binary);

  if (!file) {
    std::cerr << "Failed to open the file " << file_name << "." << std::endl;
    return;
  }

  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::size_t numElements = fileSize / sizeof(float);
  std::size_t remainingBytes = fileSize % sizeof(float);
  if (remainingBytes > 0) {
    std::cerr << "The file size is not an integer multiple of the size of "
                 "'float', which may result in incomplete data read."
              << std::endl;
  }
  res.resize(numElements);

  file.read(reinterpret_cast<char *>(res.data()), fileSize);
  file.close();
}

void im2col_process(float *&data_col, float *data_im, int32_t height,
                    int32_t width, int32_t pad_w, int32_t stride_h,
                    int32_t stride_w, int32_t dilation_w, int32_t kernel_col,
                    int32_t &input_row, int32_t output_w) {
  if (!(static_cast<uint32_t>(input_row) < static_cast<uint32_t>(height))) {
    for (int32_t output_col{output_w}; output_col != 0; output_col--) {
      *data_col = 0.f;
      ++data_col;
    }
  } else {
    int32_t input_col{-pad_w + kernel_col * dilation_w};
    for (int32_t output_col{output_w}; output_col != 0; output_col--) {
      if (static_cast<uint32_t>(input_col) < static_cast<uint32_t>(width)) {
        *data_col = data_im[input_row * width + input_col];
        ++data_col;
      } else {
        *data_col = 0.f;
        ++data_col;
      }
      input_col += stride_w;
    }
  }
  input_row += stride_h;
}

void im2col(float *data_col, float *data_im, int32_t channels, int32_t height,
            int32_t width, int32_t kernel_h, int32_t kernel_w, int32_t pad_h,
            int32_t pad_w, int32_t stride_h, int32_t stride_w,
            int32_t dilation_h, int32_t dilation_w) {
  int32_t const output_h{
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1};
  int32_t const output_w{
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1};
  int32_t const channel_size{height * width};
  for (int32_t channel{channels}; channel != 0; --channel) {
    for (int32_t kernel_row{0}; kernel_row < kernel_h; kernel_row++) {
      for (int32_t kernel_col{0}; kernel_col < kernel_w; kernel_col++) {
        int32_t input_row{-pad_h + kernel_row * dilation_h};
        for (int32_t output_rows{output_h}; output_rows != 0; output_rows--) {
          im2col_process(data_col, data_im, height, width, pad_w, stride_h,
                         stride_w, dilation_w, kernel_col, input_row, output_w);
          static_cast<void>(input_row);
        }
      }
    }
    data_im += channel_size;
  }
}

int32_t add_padding(void *output, const void *input, uint32_t dim_num,
                    const uint32_t *dim, const int64_t *stride,
                    uint32_t element_size) {
  LOGE_AND_RETURN_IF_NULL(output, -1)
  LOGE_AND_RETURN_IF_NULL(input, -1)
  LOGE_AND_RETURN_IF_NULL(dim, -1)
  LOGE_AND_RETURN_IF_NULL(stride, -1)

  add_padding_core(output, input, dim_num, dim, stride, element_size);
  return 0;
}

void add_padding_core(void *output_ptr, const void *input_ptr, uint32_t dim_num,
                      const uint32_t *dim, const int64_t *stride,
                      uint32_t element_size) {
  if (dim_num == 1) {
    memcpy(output_ptr, input_ptr, element_size * dim[0]);
    return;
  }

  char const *in_ptr{reinterpret_cast<char const *>(input_ptr)};
  char *out_ptr{reinterpret_cast<char *>(output_ptr)};
  for (int32_t idx{0U}; idx < dim[0]; idx++) {
    auto size{get_prod_size(dim + 1, dim_num - 1) * element_size};
    char const *input{in_ptr + idx * size};
    void *output{out_ptr + stride[0] * idx};
    add_padding_core(output, input, dim_num - 1, dim + 1, stride + 1,
                     element_size);
  }
}
template <typename T>
int32_t get_prod_size(T const *dim, uint32_t dim_num) {
  int32_t size{1};
  for (uint32_t idx{0}; idx < dim_num; idx++) {
    size *= dim[idx];
  }
  return size;
}
