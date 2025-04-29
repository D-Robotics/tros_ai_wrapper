// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/image_utils.h"

#include <fstream>
#include <iostream>
#include <string>

#include "glog/logging.h"
#include "utils/data_transformer.h"
#include "utils/utils.h"

void short_side_resize(cv::Mat &output_mat, cv::Mat &input_mat,
                       int short_size) {
  int resize_height = short_size;
  int resize_width = short_size;
  int ori_width = input_mat.cols;
  int ori_height = input_mat.rows;
  if (ori_width > ori_height) {
    resize_width = (ori_width / ori_height) * short_size;
    output_mat.create(resize_height, resize_width, input_mat.type());
  } else {
    resize_height = (ori_height / ori_width) * short_size;
    output_mat.create(resize_width, resize_height, input_mat.type());
  }
  cv::resize(input_mat, output_mat, output_mat.size(), 0, 0);
}

void centor_crop(cv::Mat &output_mat, cv::Mat &input_mat, int crop_size) {
  cv::Rect crop_info;
  crop_info.x = input_mat.cols / 2 - crop_size / 2;
  crop_info.y = input_mat.rows / 2 - crop_size / 2;
  crop_info.width = crop_size;
  crop_info.height = crop_size;
  cv::Mat roi(input_mat, crop_info);
  roi.copyTo(output_mat);
}

void padding_resize(ImageTensor *image_tensor, cv::Mat &output_mat,
                    cv::Mat &input_mat) {
  int input_height = input_mat.rows;
  int input_width = input_mat.cols;
  int target_height = output_mat.rows;
  int target_width = output_mat.cols;
  float scale = std::min(target_width * 1.0 / input_width,
                         target_height * 1.0 / input_height);

  int resize_height = scale * input_height;
  int resize_width = scale * input_width;
  resize_height = resize_height % 2 == 0 ? resize_height : resize_height + 1;
  resize_width = resize_width % 2 == 0 ? resize_width : resize_width + 1;

  cv::Mat resize_mat;
  resize_mat.create(resize_height, resize_width, input_mat.type());
  cv::resize(input_mat, resize_mat, resize_mat.size(), 0, 0);

  cv::copyMakeBorder(
      resize_mat, output_mat, (target_height - resize_height) / 2,
      (target_height - resize_height) / 2, (target_width - resize_width) / 2,
      (target_width - resize_width) / 2, cv::BORDER_CONSTANT,
      cv::Scalar(127, 127, 127));
  image_tensor->is_pad_resize = true;
}

void padded_center_crop(cv::Mat &output_mat, cv::Mat &input_mat,
                        float target_size, int crop_pad) {
  int input_height = input_mat.rows;
  int input_width = input_mat.cols;
  int short_size = std::min(input_height, input_width);
  float scale = target_size / (target_size + crop_pad);
  float padded_center_crop_size = scale * short_size;
  int offset_height = ((input_height - padded_center_crop_size) + 1) / 2;
  int offset_width = ((input_width - padded_center_crop_size) + 1) / 2;

  cv::Rect crop_info;
  crop_info.x = offset_width;
  crop_info.y = offset_height;
  crop_info.width = input_height - 2 * offset_width;
  crop_info.height = input_height - 2 * offset_height;
  cv::Mat roi(input_mat, crop_info);
  roi.copyTo(output_mat);
}

void bgr_to_nv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

  uint8_t *yuv = yuv_mat.ptr<uint8_t>();
  img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  uint8_t *ynv12 = img_nv12.ptr<uint8_t>();

  int uv_height = height / 2;
  int uv_width = width / 2;

  // copy y data
  int y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
}

// (i, j) = (i, a) x (a, j)
static int matrix_mul(std::vector<std::vector<float>> &A,
                      std::vector<std::vector<float>> &B,
                      std::vector<std::vector<float>> &C) {
  if (A[0].size() != B.size()) {
    VLOG(EXAMPLE_SYSTEM) << "matrix_mul shapes mismatch...";
    return -1;
  }

  int height = C.size();
  int width = C[0].size();
  int common_length = B.size();

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int c = 0; c < common_length; ++c) {
        C[i][j] += A[i][c] * B[c][j];
      }
    }
  }
  return 0;
}

static std::vector<std::vector<float>> rotation_box_3d(
    std::vector<std::vector<float>> &corner,
    std::vector<std::vector<float>> &rot_mat_T, float &x_mean, float &y_mean) {
  std::vector<std::vector<float>> rot_corner(1, std::vector<float>(3, 0.0));
  matrix_mul(corner, rot_mat_T, rot_corner);
  rot_corner[0][0] += x_mean;
  rot_corner[0][1] += y_mean;
  return rot_corner;
}

void draw_bev_detection_mul_imgs(
    ImageTensor *frame, std::vector<BevDetection3D> &bev_dets,
    std::vector<cv::Mat> &mat,
    std::vector<std::vector<std::vector<float>>> &ego2img) {
  auto num_img = frame->num_img;
  auto height = frame->input_height;
  auto stride = frame->input_width;

  for (int i = 0; i < num_img; i++) {
    std::string image_path = frame->ori_image_path_list[i];
    int32_t data_length = 0;
    char *data_buffer = nullptr;
    auto ret = read_binary_file(image_path, &data_buffer, &data_length);
    cv::Mat out_image;
    out_image.create(height * 3 / 2, stride, CV_8UC1);
    auto y_size = height * stride;
    auto uv_size = y_size / 2;

    auto y_addr =
        reinterpret_cast<uint8_t *>(data_buffer);  // + (y_size + uv_size);
    auto uv_addr = y_addr + y_size;
    VLOG(EXAMPLE_DETAIL) << "draw_bev_detection:" << height << "x" << stride
                         << ", length: " << data_length;
    auto dst_addr = out_image.data;
    memcpy(dst_addr, y_addr, y_size);
    memcpy(dst_addr + y_size, uv_addr, uv_size);
    cv::cvtColor(out_image, mat[i], CV_YUV2BGR_NV12);

    std::vector<std::vector<std::vector<float>>> img_bbox;
    bbox_ego2img(img_bbox, ego2img[i], bev_dets, height, stride, 0.4);

    draw_bev_bbox(mat[i], img_bbox, 2);
    delete[] data_buffer;
  }
}

void draw_map_detection_mul_imgs(ImageTensor *frame,
                                 std::vector<MapDetection> &map_dets,
                                 std::vector<cv::Mat> &mat,
                                 const std::vector<float> &pc_range,
                                 cv::Mat &image_lines) {
  auto num_img = frame->num_img;
  auto height = frame->input_height;
  auto stride = frame->input_width;

  for (int i = 0; i < num_img; i++) {
    std::string image_path = frame->ori_image_path_list[i];
    int32_t data_length = 0;
    char *data_buffer = nullptr;
    auto ret = read_binary_file(image_path, &data_buffer, &data_length);
    cv::Mat out_image;
    out_image.create(height * 3 / 2, stride, CV_8UC1);
    auto y_size = height * stride;
    auto uv_size = y_size / 2;

    auto y_addr =
        reinterpret_cast<uint8_t *>(data_buffer);  // + (y_size + uv_size);
    auto uv_addr = y_addr + y_size;
    VLOG(EXAMPLE_REPORT) << "draw_map_detection:" << height << " x" << stride
                         << ", length: " << data_length;
    auto dst_addr = out_image.data;
    memcpy(dst_addr, y_addr, y_size);
    memcpy(dst_addr + y_size, uv_addr, uv_size);
    cv::cvtColor(out_image, mat[i], CV_YUV2BGR_NV12);
    delete[] data_buffer;
  }

  draw_map_lines(map_dets, pc_range, image_lines);
}

void draw_map_lines(std::vector<MapDetection> &mapDets,
                    const std::vector<float> &pc_range, cv::Mat &image) {
  // scale
  float x_scale = image.cols / (pc_range[3] - pc_range[0]);
  float y_scale = image.rows / (pc_range[4] - pc_range[1]);

  // draw lines and points
  for (size_t i = 0; i < mapDets.size(); ++i) {
    const auto &points = mapDets[i].pts;
    int label = mapDets[i].label;

    std::vector<cv::Point> scaled_points;
    for (const auto &pt : points) {
      int x = static_cast<int>((pt.first - pc_range[0]) * x_scale);
      int y = image.rows - static_cast<int>((pt.second - pc_range[1]) *
                                            y_scale);  // reverse y
      scaled_points.push_back(cv::Point(x, y));
    }

    // draw lines
    for (size_t j = 0; j < scaled_points.size() - 1; ++j) {
      cv::line(image, scaled_points[j], scaled_points[j + 1], colors[label], 1,
               cv::LINE_AA, 0);
    }

    // draw points
    for (const auto &pt : scaled_points) {
      cv::circle(image, pt, 2, colors[label], cv::FILLED, cv::LINE_AA, 0);
    }
  }
}

void draw_bev_detection(ImageTensor *frame,
                        std::vector<BevDetection3D> &bev_dets,
                        std::vector<cv::Mat> &mat,
                        std::vector<std::vector<std::vector<float>>> &ego2img) {
  std::string image_path = frame->ori_image_path;
  auto height = frame->input_height;
  auto stride = frame->input_width;

  int32_t data_length = 0;
  char *data_buffer = nullptr;
  auto ret = read_binary_file(image_path, &data_buffer, &data_length);

  for (int i = 0; i < 6; i++) {
    cv::Mat out_image;
    out_image.create(height * 3 / 2, stride, CV_8UC1);
    auto y_size = height * stride;
    auto uv_size = y_size / 2;

    auto y_addr =
        reinterpret_cast<uint8_t *>(data_buffer) + i * (y_size + uv_size);
    auto uv_addr = y_addr + y_size;

    auto dst_addr = out_image.data;
    memcpy(dst_addr, y_addr, y_size);
    memcpy(dst_addr + y_size, uv_addr, uv_size);
    cv::cvtColor(out_image, mat[i], CV_YUV2BGR_NV12);

    std::vector<std::vector<std::vector<float>>> img_bbox;
    bbox_ego2img(img_bbox, ego2img[i], bev_dets, height, stride, 0.4);

    draw_bev_bbox(mat[i], img_bbox, 2);
  }

  delete[] data_buffer;
}

void draw_lidar3d(ImageTensor *frame, std::vector<LidarDetection3D> &dets,
                  cv::Mat &mat) {
  std::vector<float> corners;
  std::vector<float> angles;
  std::vector<float> scores;
  for (int i = 0; i < dets.size(); ++i) {
    Lidar3D center = dets[i].bbox;
    float x = dets[i].bbox.xs;
    float y = dets[i].bbox.ys;
    float z = dets[i].bbox.height;
    float w = dets[i].bbox.dim_0;
    float h = dets[i].bbox.dim_1;
    float p = dets[i].bbox.dim_2;
    float angle = dets[i].bbox.rot;

    float rot_sin = std::sin(angle);
    float rot_cos = std::cos(angle);
    std::vector<std::vector<float>> rot_mat_T(3, std::vector<float>(3));
    rot_mat_T[0][0] = rot_cos;
    rot_mat_T[0][1] = -rot_sin;
    rot_mat_T[0][2] = 0;
    rot_mat_T[1][0] = rot_sin;
    rot_mat_T[1][1] = rot_cos;
    rot_mat_T[1][2] = 0;
    rot_mat_T[2][0] = 0;
    rot_mat_T[2][1] = 0;
    rot_mat_T[2][2] = 1;

    std::vector<std::vector<float>> corner(1, std::vector<float>(3));
    corner[0][0] = w / 2.0;
    corner[0][1] = -h / 2.0;
    corner[0][2] = z - p / 2.0;
    std::vector<std::vector<float>> rot_corner =
        rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = w / 2.0;
    corner[0][1] = h / 2.0;
    corner[0][2] = z - p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = -w / 2.0;
    corner[0][1] = h / 2.0;
    corner[0][2] = z - p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = -w / 2.0;
    corner[0][1] = -h / 2.0;
    corner[0][2] = z - p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = w / 2.0;
    corner[0][1] = -h / 2.0;
    corner[0][2] = z + p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = w / 2.0;
    corner[0][1] = h / 2.0;
    corner[0][2] = z + p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = -w / 2.0;
    corner[0][1] = h / 2.0;
    corner[0][2] = z + p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    corner[0][0] = -w / 2.0;
    corner[0][1] = -h / 2.0;
    corner[0][2] = z + p / 2.0;
    rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
    corners.push_back(rot_corner[0][0]);
    corners.push_back(rot_corner[0][1]);
    corners.push_back(rot_corner[0][2]);

    angles.push_back(angle);
    scores.push_back(dets[i].score);
  }

  // 1. read point
  int32_t data_length = 0;
  char *data_buffer = nullptr;
  auto ret =
      read_binary_file(frame->ori_image_path, &data_buffer, &data_length);
  int element_size = data_length / 4;
  std::vector<float> padding_points(element_size);
  memcpy(padding_points.data(), data_buffer, data_length);

  // 2. get point x,y
  int point_num = element_size / 5;
  std::vector<float> points_y;
  std::vector<float> points_x;
  for (int i = 0; i < point_num; ++i) {
    points_y.push_back(padding_points[i * 5 + 1]);
    points_x.push_back(-1 * padding_points[i * 5 + 0]);
  }

  // 3. min_width, max_width
  std::vector<float>::iterator smallest_y =
      std::min_element(std::begin(points_y), std::end(points_y));
  std::vector<float>::iterator biggest_y =
      std::max_element(std::begin(points_y), std::end(points_y));
  VLOG(EXAMPLE_DEBUG) << "(min_width, max_width): " << *(smallest_y) << ", "
                      << *(biggest_y);

  // 4. min_height, max_height
  std::vector<float>::iterator smallest_x =
      std::min_element(std::begin(points_x), std::end(points_x));
  std::vector<float>::iterator biggest_x =
      std::max_element(std::begin(points_x), std::end(points_x));
  VLOG(EXAMPLE_DEBUG) << "(min_height, max_height): " << *(smallest_x) << ", "
                      << *(biggest_x);

  float width_offset =
      *(smallest_y) < 0.0f ? std::ceil(std::abs(*(smallest_y))) + 10.f : 10.f;
  float height_offset =
      *(smallest_x) < 0.0f ? std::ceil(std::abs(*(smallest_x))) + 10.f : 10.f;

  // 5. +offset
  for (int i = 0; i < points_y.size(); ++i) {
    points_y[i] += width_offset;
    points_x[i] += height_offset;
  }

  int width = std::ceil((*(biggest_y) - *(smallest_y)) + 1.f) + 20;
  int width_resize = 10;
  int height = std::ceil((*(biggest_x) - *(smallest_x)) + 1.f) + 20;
  int height_resize = 10;

  cv::Mat image(height * height_resize, width * width_resize, CV_8UC3,
                cv::Scalar(255, 255, 255));

  for (int i = 0; i < points_x.size(); ++i) {
    image.at<cv::Vec3b>(
        cv::Point2f(points_y[i] * width_resize,
                    (height - points_x[i]) * height_resize))[0] = 255;
    image.at<cv::Vec3b>(
        cv::Point2f(points_y[i] * width_resize,
                    (height - points_x[i]) * height_resize))[1] = 0;
    image.at<cv::Vec3b>(
        cv::Point2f(points_y[i] * width_resize,
                    (height - points_x[i]) * height_resize))[2] = 0;
  }

  float offset1 = 10;
  float score_thresh = 0.4;
  std::vector<float> box(16, 0);

  for (int idx = 0; idx < angles.size(); ++idx) {
    for (int p = 0; p < 8; ++p) {
      box[p * 2 + 0] = corners[idx * 8 * 3 + p * 3 + 0];
      box[p * 2 + 1] = corners[idx * 8 * 3 + p * 3 + 1];
    }
    if (scores[idx] < score_thresh) {
      continue;
    }

    // box < 0
    for (int k = 0; k < 4; ++k) {
      int i = k;
      int j = (k + 1) % 4;
      float pointi_y = box[i * 2 + 1];
      float pointj_y = box[j * 2 + 1];

      float pointi_x = -box[i * 2 + 0];
      float pointj_x = -box[j * 2 + 0];

      VLOG(EXAMPLE_DEBUG) << "(" << i << ", " << j << "): "
                          << "[" << pointi_y << ", " << pointi_x << "], "
                          << "[" << pointj_y << ", " << pointj_x << "]";

      cv::line(image,
               cv::Point2f((pointi_y + width_offset) * width_resize,
                           (height - pointi_x - height_offset) * height_resize),
               cv::Point2f((pointj_y + width_offset) * width_resize,
                           (height - pointj_x - height_offset) * height_resize),
               cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    // direction
    float length = 4;
    float axis_rot = 0.5 * 3.141592653589793;
    std::vector<float> box_xy(box.begin(), box.begin() + 8);
    float x0 = -(box_xy[0] + box_xy[2] + box_xy[4] + box_xy[6]) / 4.0;
    float y0 = (box_xy[1] + box_xy[3] + box_xy[5] + box_xy[7]) / 4.0;
    float dx = -std::cos(angles[idx] + axis_rot) * length;
    float dy = -std::sin(angles[idx] + axis_rot) * length;
    float x1 = x0 + dx;
    float y1 = y0 + dy;

    VLOG(EXAMPLE_DEBUG) << "[x0, yo, dx, dy]: " << x0 << ", " << y0 << ", "
                        << dx << ", " << dy;

    cv::arrowedLine(image,
                    cv::Point2f((y0 + width_offset) * width_resize,
                                (height - x0 - height_offset) * height_resize),
                    cv::Point2f((y1 + width_offset) * width_resize,
                                (height - x1 - height_offset) * height_resize),
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    // score
    std::stringstream text_score;
    text_score << std::fixed << std::setprecision(4) << scores[idx];
    cv::putText(image, text_score.str(),
                cv::Point2f((y0 + width_offset - 2) * width_resize,
                            (height - x0 - height_offset - 2) * height_resize),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1,
                cv::LINE_AA);
  }
  delete[] data_buffer;
  // mat = std::move(image);
  
  // affine处理，固定画布大小，并将自车移动到画布最中间位置
  static int PUBIMAGE_WIDTH = 1920;
  static int PUBIMAGE_HEIGHT = 1440;
  mat = cv::Mat(PUBIMAGE_HEIGHT, PUBIMAGE_WIDTH, mat.type());
  // 获取自车坐标
  int selfCar_point_x = width_offset * width_resize;
  int selfCar_point_y = (height - height_offset) * height_resize;
  
  // 画布大小默认为1920 * 1440 ，需要将自车移动到画布最中间位置
  int centerpoint_x = PUBIMAGE_WIDTH/2;
  int centerpoint_y = PUBIMAGE_HEIGHT/2;
  int offset_x = centerpoint_x - selfCar_point_x;
  int offset_y = centerpoint_y - selfCar_point_y;
  cv::Scalar colorborder(255, 255, 255); // 白色填充
  cv::Mat warp_matrix = (cv::Mat_<float>(2, 3) <<
        cos(0), -sin(0), offset_x,
        sin(0), cos(0), offset_y);
  cv::warpAffine(image, mat, warp_matrix, mat.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, colorborder);

  // 输出图像尺寸
  VLOG(EXAMPLE_DEBUG) << "mat.size(): " << mat.size();
}

int draw_perception(ImageTensor *frame, Perception *perception, cv::Mat &mat) {
  if (frame->tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_S8 &&
      frame->tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    // ori_image_path is non-empty only in image_list input mode
    if (frame->ori_image_path.empty()) {
      return -1;
    } else {
      mat = cv::imread(frame->ori_image_path);
    }
  }

  VLOG(EXAMPLE_DEBUG) << "perception type: " << perception->type
    << ", DET3D: " << Perception::DET3D
    << ", LIDAR3D: " << Perception::LIDAR3D;

  if (perception->type == Perception::DET) {
    auto &det = perception->det;
    for (int i = 0; i < det.size(); i++) {
      auto &color = colors[det[i].id % 7];
      Bbox &bbox = det[i].bbox;
      auto w_base = perception->w_base;
      auto h_base = perception->h_base;
      cv::rectangle(mat, cv::Point(bbox.xmin * w_base, bbox.ymin * h_base),
                    cv::Point(bbox.xmax * w_base, bbox.ymax * h_base), color);
      std::stringstream text_ss;
      std::string class_name =
          det[i].class_name == nullptr ? "empty" : det[i].class_name;
      text_ss << det[i].id << " " << class_name << ":" << std::fixed
              << std::setprecision(4) << det[i].score;
      cv::putText(
          mat, text_ss.str(),
          cv::Point(bbox.xmin * w_base, std::abs(bbox.ymin * h_base - 5)),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
  } else if (perception->type == Perception::MASK) {
    auto &det = perception->mask.det_info;
    for (int i = 0; i < det.size(); i++) {
      auto &color = colors[det[i].id % 7];
      Bbox &bbox = det[i].bbox;
      auto w_base = perception->mask.w_base;
      auto h_base = perception->mask.h_base;
      cv::rectangle(mat, cv::Point(bbox.xmin * w_base, bbox.ymin * h_base),
                    cv::Point(bbox.xmax * w_base, bbox.ymax * h_base), color);
      std::stringstream text_ss;
      std::string class_name =
          det[i].class_name == nullptr ? "empty" : det[i].class_name;
      text_ss << det[i].id << " " << class_name << ":" << std::fixed
              << std::setprecision(4) << det[i].score;
      cv::putText(
          mat, text_ss.str(),
          cv::Point(bbox.xmin * w_base, std::abs(bbox.ymin * h_base - 5)),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
  } else if (perception->type == Perception::SEG) {
    auto result_ptr = perception->seg.seg.data();
    int parsing_width = perception->seg.width;
    int parsing_height = perception->seg.height;

    cv::Mat parsing_img(parsing_height, parsing_width, CV_8UC3);
    uint8_t *parsing_img_ptr = parsing_img.ptr<uint8_t>();
    auto w_base = perception->w_base;
    auto h_base = perception->h_base;
    // set parsing bgr
    for (int i = 0; i < parsing_height; ++i) {
      for (int j = 0; j < parsing_width; ++j) {
        int src_h = i * h_base;
        int src_w = j * w_base;
        int8_t id = result_ptr[src_h * parsing_width + src_w];
        if (id >= 19) continue;
        *parsing_img_ptr++ = bgr_putpalette[id * 3];
        *parsing_img_ptr++ = bgr_putpalette[id * 3 + 1];
        *parsing_img_ptr++ = bgr_putpalette[id * 3 + 2];
      }
    }

    // resize parsing image
    cv::resize(parsing_img, parsing_img, mat.size(), 0, 0, cv::INTER_NEAREST);

    // alpha blending
    float alpha_f = 0.5;
    cv::Mat dst;

    addWeighted(mat, alpha_f, parsing_img, 1 - alpha_f, 0.0, dst);
    mat = std::move(dst);
  } else if (perception->type == Perception::DET3D) {
    // get boxes
    std::vector<Detection3D> dets = perception->det3d;

    std::vector<float> corners;
    std::vector<float> angles;
    std::vector<float> scores;
    for (int i = 0; i < dets.size(); ++i) {
      Bbox3D center = dets[i].bbox;
      float x = dets[i].bbox.x;
      float y = dets[i].bbox.y;
      float z = dets[i].bbox.z;
      float w = dets[i].bbox.w;
      float h = dets[i].bbox.l;
      float p = dets[i].bbox.h;
      float angle = dets[i].bbox.r;

      float rot_sin = std::sin(angle);
      float rot_cos = std::cos(angle);
      std::vector<std::vector<float>> rot_mat_T(3, std::vector<float>(3));
      rot_mat_T[0][0] = rot_cos;
      rot_mat_T[0][1] = -rot_sin;
      rot_mat_T[0][2] = 0;
      rot_mat_T[1][0] = rot_sin;
      rot_mat_T[1][1] = rot_cos;
      rot_mat_T[1][2] = 0;
      rot_mat_T[2][0] = 0;
      rot_mat_T[2][1] = 0;
      rot_mat_T[2][2] = 1;

      std::vector<std::vector<float>> corner(1, std::vector<float>(3));
      corner[0][0] = w / 2.0;
      corner[0][1] = -h / 2.0;
      corner[0][2] = z - p / 2.0;
      std::vector<std::vector<float>> rot_corner =
          rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = w / 2.0;
      corner[0][1] = h / 2.0;
      corner[0][2] = z - p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = -w / 2.0;
      corner[0][1] = h / 2.0;
      corner[0][2] = z - p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = -w / 2.0;
      corner[0][1] = -h / 2.0;
      corner[0][2] = z - p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = w / 2.0;
      corner[0][1] = -h / 2.0;
      corner[0][2] = z + p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = w / 2.0;
      corner[0][1] = h / 2.0;
      corner[0][2] = z + p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = -w / 2.0;
      corner[0][1] = h / 2.0;
      corner[0][2] = z + p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      corner[0][0] = -w / 2.0;
      corner[0][1] = -h / 2.0;
      corner[0][2] = z + p / 2.0;
      rot_corner = rotation_box_3d(corner, rot_mat_T, x, y);
      corners.push_back(rot_corner[0][0]);
      corners.push_back(rot_corner[0][1]);
      corners.push_back(rot_corner[0][2]);

      angles.push_back(angle);
      scores.push_back(dets[i].score);
    }

    // 1. read point
    int32_t data_length = 0;
    char *data_buffer = nullptr;
    auto ret =
        read_binary_file(frame->ori_image_path, &data_buffer, &data_length);
    int element_size = data_length / 4;
    std::vector<float> padding_points(element_size);
    memcpy(padding_points.data(), data_buffer, data_length);

    // 2. remove padding
    int point_num = element_size / 4;
    std::vector<float> points_y;
    std::vector<float> points_x;
    for (int i = 0; i < point_num; ++i) {
      // padding data: [-100, -100, -100, -100]
      if (padding_points[i * 4] == -100.f) {
        break;
      }
      points_y.push_back(-1 * padding_points[i * 4 + 1]);
      points_x.push_back(padding_points[i * 4 + 0]);
    }

    // 3. min_width, max_width
    std::vector<float>::iterator smallest_y =
        std::min_element(std::begin(points_y), std::end(points_y));
    std::vector<float>::iterator biggest_y =
        std::max_element(std::begin(points_y), std::end(points_y));
    VLOG(EXAMPLE_DEBUG) << "(min_width, max_width): " << *(smallest_y) << ", "
                        << *(biggest_y);

    // 4. min_height, max_height
    std::vector<float>::iterator smallest_x =
        std::min_element(std::begin(points_x), std::end(points_x));
    std::vector<float>::iterator biggest_x =
        std::max_element(std::begin(points_x), std::end(points_x));
    VLOG(EXAMPLE_DEBUG) << "(min_height, max_height): " << *(smallest_x) << ", "
                        << *(biggest_x);

    float width_offset =
        *(smallest_y) < 0.0f ? std::ceil(std::abs(*(smallest_y))) + 10.f : 10.f;
    float height_offset =
        *(smallest_x) < 0.0f ? std::ceil(std::abs(*(smallest_x))) + 10.f : 10.f;

    // 5. +offset
    for (int i = 0; i < points_y.size(); ++i) {
      points_y[i] += width_offset;
      points_x[i] += height_offset;
    }

    int width = std::ceil((*(biggest_y) - *(smallest_y)) + 1.f) + 20;
    int width_resize = 10;
    int height = std::ceil((*(biggest_x) - *(smallest_x)) + 1.f) + 20;
    int height_resize = 10;

    cv::Mat image(height * height_resize, width * width_resize, CV_8UC3,
                  cv::Scalar(255, 255, 255));

    for (int i = 0; i < points_x.size(); ++i) {
      image.at<cv::Vec3b>(
          cv::Point2f(points_y[i] * width_resize,
                      (height - points_x[i]) * height_resize))[0] = 255;
      image.at<cv::Vec3b>(
          cv::Point2f(points_y[i] * width_resize,
                      (height - points_x[i]) * height_resize))[1] = 0;
      image.at<cv::Vec3b>(
          cv::Point2f(points_y[i] * width_resize,
                      (height - points_x[i]) * height_resize))[2] = 0;
    }

    float offset1 = 10;
    float score_thresh = 0.4;
    std::vector<float> box(16, 0);

    for (int idx = 0; idx < angles.size(); ++idx) {
      for (int p = 0; p < 8; ++p) {
        box[p * 2 + 0] = corners[idx * 8 * 3 + p * 3 + 0];
        box[p * 2 + 1] = corners[idx * 8 * 3 + p * 3 + 1];
      }
      if (scores[idx] < score_thresh) {
        continue;
      }

      // box < 0
      for (int k = 0; k < 4; ++k) {
        int i = k;
        int j = (k + 1) % 4;
        float pointi_y = -box[i * 2 + 1];
        float pointj_y = -box[j * 2 + 1];

        float pointi_x = box[i * 2 + 0];
        float pointj_x = box[j * 2 + 0];

        VLOG(EXAMPLE_DEBUG) << "(" << i << ", " << j << "): "
                            << "[" << pointi_y << ", " << pointi_x << "], "
                            << "[" << pointj_y << ", " << pointj_x << "]";

        cv::line(
            image,
            cv::Point2f((pointi_y + width_offset) * width_resize,
                        (height - pointi_x - height_offset) * height_resize),
            cv::Point2f((pointj_y + width_offset) * width_resize,
                        (height - pointj_x - height_offset) * height_resize),
            cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      }

      // direction
      float length = 4;
      float axis_rot = -0.5 * 3.141592653589793;
      std::vector<float> box_xy(box.begin(), box.begin() + 8);
      float x0 = (box_xy[0] + box_xy[2] + box_xy[4] + box_xy[6]) / 4.0;
      float y0 = -(box_xy[1] + box_xy[3] + box_xy[5] + box_xy[7]) / 4.0;
      float dx = -std::cos(angles[idx] + axis_rot) * length;
      float dy = -std::sin(angles[idx] + axis_rot) * length;
      float x1 = x0 + dx;
      float y1 = y0 + dy;

      VLOG(EXAMPLE_DEBUG) << "[x0, yo, dx, dy]: " << x0 << ", " << y0 << ", "
                          << dx << ", " << dy;

      cv::arrowedLine(
          image,
          cv::Point2f((y0 + width_offset) * width_resize,
                      (height - x0 - height_offset) * height_resize),
          cv::Point2f((y1 + width_offset) * width_resize,
                      (height - x1 - height_offset) * height_resize),
          cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
      // score
      std::stringstream text_score;
      text_score << std::fixed << std::setprecision(4) << scores[idx];
      cv::putText(
          image, text_score.str(),
          cv::Point2f((y0 + width_offset - 2) * width_resize,
                      (height - x0 - height_offset - 2) * height_resize),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    delete[] data_buffer;
    mat = std::move(image);
  } else if (perception->type == Perception::POINT) {
    int height = perception->pt.height;
    int width = perception->pt.width;
    int channel = 2;
    int c_stride = height * width;

    float *flow_chw = perception->pt.point.data();
    std::vector<float> flow_c1;
    flow_c1.reserve(c_stride);
    std::vector<float> flow_c2;
    flow_c2.reserve(c_stride);
    for (int i = 0; i < c_stride; ++i) {
      flow_c1.emplace_back(flow_chw[i] * 4.0);
      flow_c2.emplace_back(flow_chw[c_stride + i] * 4.0);
    }

    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_c1, flow_c2, magnitude, angle);
    float *magnitude_data = reinterpret_cast<float *>(magnitude.data);
    if (magnitude_data == nullptr) {
      VLOG(EXAMPLE_SYSTEM) << "magnitude_data is null pointer";
      return -1;
    }
    float *angle_data = reinterpret_cast<float *>(angle.data);
    // 1. magnitude: nan check
    for (int i = 0; i < c_stride; ++i) {
      magnitude_data[i] =
          std::isnan(magnitude_data[i]) ? 0.0f : magnitude_data[i];
    }
    // 2. magnitude: normalize
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    // 3. flow2img
    cv::Mat flow_img(height, width, CV_8UC3);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int stride = h * width + w;
        flow_img.at<cv::Vec3b>(h, w)[0] = angle_data[stride] * 180.0 / PI / 2.0;
        flow_img.at<cv::Vec3b>(h, w)[1] = magnitude_data[stride];
        flow_img.at<cv::Vec3b>(h, w)[2] = 255;
      }
    }
    // 4. hsv2rgb
    cv::cvtColor(flow_img, flow_img, cv::COLOR_HSV2BGR);
    // 5. upx4
    cv::resize(flow_img, mat, cv::Size(width * 4, height * 4));
  } else if (perception->type == Perception::KEYPOINT) {
    auto &kpt = perception->kpt;
    float scale_w = static_cast<float>(frame->ori_image_width) /
                    static_cast<float>(kpt.width);
    float scale_h = static_cast<float>(frame->ori_image_height) /
                    static_cast<float>(kpt.height);
    if (!kpt.is_keypoints) {
      for (int i = 0; i < kpt.groups_size; i++) {
        for (int j = 0; j < kpt.points[i].size() - 1; j++) {
          cv::line(mat,
                   cv::Point(kpt.points[i][j].first * scale_w,
                             (kpt.points[i][j].second - 270) * scale_h),
                   cv::Point(kpt.points[i][j + 1].first * scale_w,
                             (kpt.points[i][j + 1].second - 270) * scale_h),
                   colors[i % 7], 2);
        }
      }
    } else {
      for (int i = 0; i < kpt.groups_size; i++) {
        for (int j = 0; j < kpt.points[i].size(); j++) {
          cv::circle(mat,
                     cv::Point(kpt.points[i][j].first * scale_w,
                               (kpt.points[i][j].second) * scale_h),
                     2, colors[i % 7], 2);
        }
      }
    }
  } else if (perception->type == Perception::DETCAM3D) {
    auto &detcam3d = perception->detcam3d;
    std::vector<std::vector<float>> edges = {{0, 1}, {0, 3}, {0, 4}, {1, 2},
                                             {1, 5}, {3, 2}, {3, 7}, {4, 5},
                                             {4, 7}, {2, 6}, {5, 6}, {6, 7}};
    if (detcam3d[0].bboxes_.bboxes_empty) {
      return 0;
    }
    std::vector<std::vector<std::vector<float>>> corners{
        detcam3d[0].bboxes_.bboxes.size()};
    cam2imgcorner(corners, detcam3d[0].bboxes_.bboxes);

    for (int i = 0; i < corners.size(); i++) {
      std::vector<Pointscam3D2Img> points(8);
      points_cam2img(points, corners[i], detcam3d[0].bboxes_.ca2img_v, true);

      for (int edges_i = 0; edges_i < edges.size(); edges_i++) {
        cv::line(mat,
                 cv::Point(points[edges[edges_i][0]].v[0],
                           points[edges[edges_i][0]].v[1]),
                 cv::Point(points[edges[edges_i][1]].v[0],
                           points[edges[edges_i][1]].v[1]),
                 colors[i % 7], 2);
      }
    }

  } else if (perception->type == Perception::LIDAR3D) {
    std::vector<LidarDetection3D> dets = perception->lidar3d;
    draw_lidar3d(frame, dets, mat);
  } else {
    auto &cls = perception->cls;
    for (int i = 0; i < cls.size(); i++) {
      auto &c = cls[i];
      auto &color = colors[c.id % 7];
      std::stringstream text_ss;
      text_ss << c.id << ":" << std::fixed << std::setprecision(5) << c.score;
      cv::putText(mat, text_ss.str(), cv::Point(5, 20 + 10 * i),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }
  }
  return 0;
}

void bbox_to_corner(std::vector<std::vector<std::vector<float>>> &corner_bbox,
                    std::vector<float> scores, std::vector<int> labels,
                    std::vector<std::vector<float>> &bev_bbox,
                    float score_thresh) {
  int32_t bbox_num = bev_bbox.size();
  float score{0.f};
  int label{0};
  float x{0.f};
  float y{0.f};
  float z{0.f};
  float w{0.f};
  float l{0.f};
  float h{0.f};
  float rot{0.f};
  for (int32_t i = 0; i < bbox_num; i++) {
    score = bev_bbox[i][9];
    if (score < score_thresh) {
      continue;
    }
    x = bev_bbox[i][0];
    y = bev_bbox[i][1];
    z = bev_bbox[i][2];
    w = bev_bbox[i][3];
    l = bev_bbox[i][4];
    h = bev_bbox[i][5];
    rot = bev_bbox[i][6];
    label = bev_bbox[i][10];

    std::vector<std::vector<float>> corner_tmp(3, std::vector<float>(8, 0.f));
    for (int32_t idx = 0; idx < 8; idx++) {
      corner_tmp[0][idx] = l / 2.f * corner_arr[0][idx];
      corner_tmp[1][idx] = w / 2.f * corner_arr[1][idx];
      corner_tmp[2][idx] = h / 2.f * corner_arr[2][idx];
    }

    Quaternion quaternion(rot, 0, 0, 1);
    auto rotation = quaternion.RotationMatrix();
    std::vector<std::vector<float>> corner(3, std::vector<float>(8, 0.f));
    matrix_mul(rotation, corner_tmp, corner);

    for (int32_t idx = 0; idx < 8; idx++) {
      corner[0][idx] += x;
      corner[1][idx] += y;
      corner[2][idx] += z;
    }

    corner_bbox.emplace_back(corner);
    scores.emplace_back(score);
    labels.emplace_back(label);
  }
}

void bbox_ego2bev(std::vector<std::vector<float>> &bev_bbox,
                  std::vector<BevDetection3D> &ego_bbox) {
  float bev_size[3] = {51.2, 51.2, 0.8};
  float max_x = bev_size[1] - bev_size[2] / 2;
  float max_y = bev_size[0] - bev_size[2] / 2;
  bev_bbox.resize(ego_bbox.size(), std::vector<float>(11, 0.f));
  for (int32_t i = 0; i < ego_bbox.size(); i++) {
    bev_bbox[i][0] = (ego_bbox[i].bbox.xs + max_x) / bev_size[2];
    bev_bbox[i][1] = (ego_bbox[i].bbox.ys + max_y) / bev_size[2];
    bev_bbox[i][2] = ego_bbox[i].bbox.height / bev_size[2];
    bev_bbox[i][3] = ego_bbox[i].bbox.dim_0 / bev_size[2];
    bev_bbox[i][4] = ego_bbox[i].bbox.dim_1 / bev_size[2];
    bev_bbox[i][5] = ego_bbox[i].bbox.dim_2 / bev_size[2];
    bev_bbox[i][6] = ego_bbox[i].bbox.rot;
    bev_bbox[i][7] = ego_bbox[i].bbox.vel_0 / bev_size[2];
    bev_bbox[i][8] = ego_bbox[i].bbox.vel_1 / bev_size[2];

    bev_bbox[i][9] = ego_bbox[i].score;
    bev_bbox[i][10] = ego_bbox[i].label;
  }
}

void bbox_ego2img(std::vector<std::vector<std::vector<float>>> &img_bbox,
                  std::vector<std::vector<float>> &ego2img,
                  std::vector<BevDetection3D> &ego_bbox, int32_t height,
                  int32_t width, float score_thresh) {
  int32_t bbox_num = ego_bbox.size();
  for (int32_t i = 0; i < bbox_num; i++) {
    if (ego_bbox[i].score < score_thresh) {
      continue;
    }

    std::vector<std::vector<float>> corner_tmp(3, std::vector<float>(8, 0.f));
    for (int32_t idx = 0; idx < 8; idx++) {
      corner_tmp[0][idx] = ego_bbox[i].bbox.dim_1 / 2.f * corner_arr[0][idx];
      corner_tmp[1][idx] = ego_bbox[i].bbox.dim_0 / 2.f * corner_arr[1][idx];
      corner_tmp[2][idx] = ego_bbox[i].bbox.dim_2 / 2.f * corner_arr[2][idx];
    }

    Quaternion quaternion(ego_bbox[i].bbox.rot, 0, 0, 1);
    auto rotation = quaternion.RotationMatrix();
    std::vector<std::vector<float>> corner_3d_tmp(3,
                                                  std::vector<float>(8, 0.f));

    matrix_mul(rotation, corner_tmp, corner_3d_tmp);

    for (int32_t idx = 0; idx < 8; idx++) {
      corner_3d_tmp[0][idx] += ego_bbox[i].bbox.xs;
      corner_3d_tmp[1][idx] += ego_bbox[i].bbox.ys;
      corner_3d_tmp[2][idx] += ego_bbox[i].bbox.height;
    }

    corner_3d_tmp.emplace_back(std::vector<float>(8, 1.f));
    std::vector<std::vector<float>> corner_3d(4, std::vector<float>(8, 0.f));
    matrix_mul(ego2img, corner_3d_tmp, corner_3d);
    std::vector<std::vector<float>> corner_2d(3, std::vector<float>(8, 1.f));

    std::vector<bool> visible(8, false);
    for (int32_t idx = 0; idx < 8; idx++) {
      corner_2d[0][idx] = corner_3d[0][idx] / corner_3d[2][idx];
      corner_2d[1][idx] = corner_3d[1][idx] / corner_3d[2][idx];
      visible[idx] = (corner_2d[0][idx] > 0.f) &&
                     (corner_2d[0][idx] < static_cast<float>(width)) &&
                     (corner_2d[1][idx] > 0.f) &&
                     (corner_2d[1][idx] < static_cast<float>(height)) &&
                     (corner_3d[2][idx] > 1.f);
      corner_2d[0][idx] = std::min(std::max(0.f, corner_2d[0][idx]),
                                   static_cast<float>(width - 1));
      corner_2d[1][idx] = std::min(std::max(0.f, corner_2d[1][idx]),
                                   static_cast<float>(height - 1));
    }

    for (int idx = 0; idx < 8; idx++) {
      if (visible[idx]) {
        img_bbox.emplace_back(corner_2d);
        break;
      }
    }
  }
}

int32_t read_bev_homo(std::vector<std::vector<std::vector<float>>> &homo,
                      std::string &file_path) {
  homo.resize(6,
              std::vector<std::vector<float>>(4, std::vector<float>(4, 0.f)));
  std::ifstream ofs(file_path, std::ios::binary);
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "read bev homography file error, path: "
                         << file_path;
    return -1;
  }
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        float value;
        ofs.read(reinterpret_cast<char *>(&value), sizeof(float));
        homo[i][j][k] = value;
      }
    }
  }
  ofs.close();
  return 0;
}

void get_ego2img(std::vector<std::vector<std::vector<float>>> &ego2img,
                 std::vector<std::vector<std::vector<float>>> &homo,
                 int ori_height, int ori_width, int resize_height,
                 int resize_width, int height, int width, bool is_pad_resize) {
  ego2img.resize(
      6, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.f)));
  std::vector<std::vector<float>> view1(4, std::vector<float>(4, 0.f));
  std::vector<std::vector<float>> view2(4, std::vector<float>(4, 0.f));
  view1[0][0] =
      static_cast<float>(resize_height) / static_cast<float>(ori_height);
  view1[1][1] =
      static_cast<float>(resize_width) / static_cast<float>(ori_width);
  view1[2][2] = 1.f;
  view1[3][3] = 1.f;

  view2[0][0] = 1.f;
  view2[1][1] = 1.f;
  view2[2][2] = 1.f;
  view2[3][3] = 1.f;
  if (!is_pad_resize) {
    view2[0][2] -= static_cast<float>((resize_width - width) / 2);
    view2[1][2] -= static_cast<float>(resize_height - height);
  }

  for (int i = 0; i < 6; i++) {
    std::vector<std::vector<float>> tmp(4, std::vector<float>(4, 0.f));
    matrix_mul(view1, homo[i], tmp);
    matrix_mul(view2, tmp, ego2img[i]);
  }
}

void create_pascal_label_colormap(std::vector<std::vector<int32_t>> &colormap) {
  colormap.resize(256, std::vector<int32_t>(3, 0));
  std::vector<int32_t> ind(256);

  for (int32_t i = 0; i < 256; ++i) {
    ind[i] = i;
  }

  for (int32_t shift = 7; shift >= 0; --shift) {
    for (int32_t channel = 0; channel < 3; ++channel) {
      for (int32_t i = 0; i < 256; ++i) {
        colormap[i][channel] |= ((ind[i] >> channel) & 1) << shift;
      }
    }
    for (int32_t i = 0; i < 256; ++i) {
      ind[i] >>= 3;
    }
  }
}

void draw_bev_segment(ImageTensor *frame, Parsing<uint32_t> &segs,
                      std::vector<std::vector<int32_t>> &colormap,
                      cv::Mat &mat) {
  auto result_ptr = segs.seg.data();
  int width = segs.width;
  int height = segs.height;

  cv::Mat seg_img(height, width, CV_8UC3);
  uint8_t *seg_img_ptr = seg_img.ptr<uint8_t>();

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      uint8_t id = static_cast<uint8_t>(result_ptr[h * width + w]);
      if (id >= 256) continue;
      *seg_img_ptr++ = colormap[id][2];
      *seg_img_ptr++ = colormap[id][1];
      *seg_img_ptr++ = colormap[id][0];
    }
  }

  mat = std::move(seg_img);
}

int draw_rect(cv::Mat &mat, std::vector<std::vector<float>> corner,
              cv::Scalar color, int thickness) {
  if (corner.size() != 2 && corner[0].size() != 4) {
    VLOG(EXAMPLE_SYSTEM) << "corner size error";
    return -1;
  }

  cv::Point point0{static_cast<int>(corner[0][0]),
                   static_cast<int>(corner[1][0])};
  cv::Point point1{static_cast<int>(corner[0][1]),
                   static_cast<int>(corner[1][1])};
  cv::Point point2{static_cast<int>(corner[0][2]),
                   static_cast<int>(corner[1][2])};
  cv::Point point3{static_cast<int>(corner[0][3]),
                   static_cast<int>(corner[1][3])};
  cv::line(mat, point0, point1, color, thickness);
  cv::line(mat, point1, point2, color, thickness);
  cv::line(mat, point2, point3, color, thickness);
  cv::line(mat, point3, point0, color, thickness);

  return 0;
}

int draw_bev_bbox(cv::Mat &mat,
                  std::vector<std::vector<std::vector<float>>> &corner_bbox,
                  int thickness, float resize_ratio) {
  for (auto &bbox : corner_bbox) {
    if (bbox.size() != 3 && bbox[0].size() != 8) {
      VLOG(EXAMPLE_SYSTEM) << "corner size error";
      return -1;
    }

    std::vector<std::vector<float>> rect1(2, std::vector<float>(4, 0.f));
    std::vector<std::vector<float>> rect2(2, std::vector<float>(4, 0.f));

    for (int i = 0; i < 4; i++) {
      cv::Point point0{static_cast<int>(bbox[0][i]) * resize_ratio,
                       static_cast<int>(bbox[1][i]) * resize_ratio};
      cv::Point point1{static_cast<int>(bbox[0][i + 4] * resize_ratio),
                       static_cast<int>(bbox[1][i + 4] * resize_ratio)};
      cv::line(mat, point0, point1, cv::Scalar(0, 0, 0), thickness);

      rect1[0][i] = bbox[0][i] * resize_ratio;
      rect1[1][i] = bbox[1][i] * resize_ratio;
      rect2[0][i] = bbox[0][i + 4] * resize_ratio;
      rect2[1][i] = bbox[1][i + 4] * resize_ratio;
    }
    draw_rect(mat, rect1, cv::Scalar(255, 0, 0), thickness);
    draw_rect(mat, rect2, cv::Scalar(0, 0, 255), thickness);

    int center_bottom_forward_x =
        static_cast<int>((bbox[0][2] + bbox[0][3]) / 2 * resize_ratio);
    int center_bottom_forward_y =
        static_cast<int>((bbox[1][2] + bbox[1][3]) / 2 * resize_ratio);
    int center_bottom_x = static_cast<int>(
        (bbox[0][2] + bbox[0][3] + bbox[0][6] + bbox[0][7]) / 4 * resize_ratio);
    int center_bottom_y = static_cast<int>(
        (bbox[1][2] + bbox[1][3] + bbox[1][6] + bbox[1][7]) / 4 * resize_ratio);
    cv::Point point0{center_bottom_forward_x, center_bottom_forward_y};
    cv::Point point1{center_bottom_x, center_bottom_y};
    cv::line(mat, point0, point1, cv::Scalar(255, 0, 0), thickness);
  }
  return 0;
}

const std::vector<cv::Vec4b> colors_map = {
    cv::Vec4b(0, 0, 0, 255),        // 0 undefined
    cv::Vec4b(112, 128, 144, 255),  // 1 car orange
    cv::Vec4b(220, 20, 60, 255),    // 2 pedestrian Blue
    cv::Vec4b(255, 127, 80, 255),   // 3 sign Darkslategrey
    cv::Vec4b(255, 158, 0, 255),    // 4 CYCLIST Crimson
    cv::Vec4b(233, 150, 70, 255),   // 5 traffic_light Orangered
    cv::Vec4b(255, 61, 99, 255),    // 6 pole Darkorange
    cv::Vec4b(0, 0, 230, 255),      // 7 construction_cone Darksalmon
    cv::Vec4b(47, 79, 79, 255),     // 8 bicycle Red
    cv::Vec4b(255, 140, 0, 255),    // 9 motorcycle Slategrey
    cv::Vec4b(255, 99, 71, 255),    // 10 building Burlywood
    cv::Vec4b(0, 207, 191, 255),    // 11 vegetation Green
    cv::Vec4b(175, 0, 75, 255),     // 12 trunk nuTonomy green
    cv::Vec4b(75, 0, 75, 255),      // 13 curb, road, lane_marker, other_ground
    cv::Vec4b(112, 180, 60, 255),   // 14 walkable, sidewalk
    cv::Vec4b(222, 184, 135, 255),  // 15 unobserved
    cv::Vec4b(0, 175, 0, 255),      // 16 undefined
    cv::Vec4b(0, 0, 0, 255)         // 17 undefined
};

int draw_2d_occ(Parsing3d<uint32_t> &seg3d, cv::Mat &occ_bev_resized,
                bool reverse_rgb) {
  int32_t height = seg3d.h;
  int32_t width = seg3d.w;
  int32_t channels = seg3d.z;
  std::vector<uint32_t> flat_semantics = seg3d.seg;
  cv::Mat occ_bev(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base_idx = (h * width + w) * channels;

      bool has_valid = false;
      int max_idx = 0;
      float max_val = -1.0f;

      for (int c = 0; c < channels; c++) {
        float val = flat_semantics[base_idx + c];
        if (val > max_val && val != 17) {
          max_val = val;
          max_idx = c;
          has_valid = true;
        }
      }

      if (has_valid) {
        int color_idx = static_cast<int>(flat_semantics[base_idx + max_idx]);
        if (color_idx >= 0 && color_idx < colors_map.size()) {
          cv::Vec4b color = colors_map[color_idx];
          occ_bev.at<cv::Vec3b>(h, w) = cv::Vec3b(color[0], color[1], color[2]);
        }
      }
    }
  }
  cv::resize(occ_bev, occ_bev_resized, cv::Size(400, 400));

  if (reverse_rgb) {
    cv::Mat temp;
    cv::cvtColor(occ_bev_resized, temp, cv::COLOR_RGB2BGR);
    occ_bev_resized = temp;
  }
  return 0;
}
