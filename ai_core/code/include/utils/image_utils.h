// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_IMAGE_UTILS_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_IMAGE_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "input/input_data.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

#include "utils.h"

double const PI = 3.14159265;

static float corner_arr[3][8] = {{1.f, 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f},
                                 {1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f},
                                 {1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f}};

static std::vector<std::string> cam_names = {
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"};

static cv::Scalar colors[] = {
    cv::Scalar(255, 0, 0),     // red
    cv::Scalar(255, 165, 0),   // orange
    cv::Scalar(255, 255, 0),   // yellow
    cv::Scalar(0, 255, 0),     // green
    cv::Scalar(0, 0, 255),     // blue
    cv::Scalar(75, 0, 130),    // indigo
    cv::Scalar(238, 130, 238)  // violet
};
static uint8_t bgr_putpalette[] = {
    128, 64,  128, 244, 35,  232, 70,  70,  70,  102, 102, 156, 190, 153, 153,
    153, 153, 153, 250, 170, 30,  220, 220, 0,   107, 142, 35,  152, 251, 152,
    0,   130, 180, 220, 20,  60,  255, 0,   0,   0,   0,   142, 0,   0,   70,
    0,   60,  100, 0,   80,  100, 0,   0,   230, 119, 11,  32};

/**
 * Convert BGR to NV12
 * @param[in] bgr
 * @param[out] img_nv12
 */
void bgr_to_nv12(cv::Mat &bgr, cv::Mat &img_nv12);

/**
 * short size resize
 * @param[out] output_mat
 * @param[in] input_mat
 * @param[in] short_size
 */
void short_side_resize(cv::Mat &output_mat, cv::Mat &input_mat, int short_size);

/**
 * centor crop
 * @param[out] output_mat
 * @param[in] input_mat
 * @param[in] crop_size
 */
void centor_crop(cv::Mat &output_mat, cv::Mat &input_mat, int crop_size);

/**
 * padding and resize
 * @param[out] output_mat
 * @param[in] input_mat
 */
void padding_resize(ImageTensor *image_tensor, cv::Mat &output_mat,
                    cv::Mat &input_mat);

/**
 * padding centor crop
 * @param[out] output_mat
 * @param[in] input_mat
 * @param[in] target_size
 * @param[in] crop_pad
 */
void padded_center_crop(cv::Mat &output_mat, cv::Mat &input_mat,
                        float target_size, int crop_pad);

/**
 * draw rect
 * @param[out] mat
 * @param[in] corner (2,4)
 * @param[in] color
 * @param[in] thickness
 * @return 0 if success
 */
int draw_rect(cv::Mat &mat, std::vector<std::vector<float>> corner,
              cv::Scalar color, int thickness = 1);

/**
 * draw bev bbox
 * @param[out] mat
 * @param[in] corner_bbox (n,3,8)
 * @param[in] thickness
 * @return 0 if success
 */
int draw_bev_bbox(cv::Mat &mat,
                  std::vector<std::vector<std::vector<float>>> &corner_bbox,
                  int thickness = 1, float resize_ratio = 1.0);

/**
 * Get 3d bbox corner.
 * @param[out] corner_bbox meta info for corner bbox (n,3,8)
 * @param[out] scores (n,1)
 * @param[out] labels (n,1)
 * @param[in] bev_bbox: meta info for bbox (x,y,z,w,l,h,r,v0,v1,score,label)
 * @param[in] score_thresh: theshold for filtering bbox with low score.
 */
void bbox_to_corner(std::vector<std::vector<std::vector<float>>> &corner_bbox,
                    std::vector<float> scores, std::vector<int> labels,
                    std::vector<std::vector<float>> &bev_bbox,
                    float score_thresh = 0.f);

/**
 * Convert ego coordinate to bev coordinate.
 * @param[out] bev_bbox: meta info for bbox (x,y,z,w,l,h,r,v0,v1,score,label)
 * @param[in] ego_bbox: bev post process result
 */
void bbox_ego2bev(std::vector<std::vector<float>> &bev_bbox,
                  std::vector<BevDetection3D> &ego_bbox);

/**
 * Convert ego coordinate to img coordinate
 * @param[out] img_bbox:
 * @param[in] ego_bbox: bev post process result
 * @param[in] ego2img: homography for ego to image coordinate.
 * @param[in] height:
 * @param[in] width:
 * @param[in] score_thresh: theshold for filtering bbox with low score.
 */
void bbox_ego2img(std::vector<std::vector<std::vector<float>>> &img_bbox,
                  std::vector<std::vector<float>> &ego2img,
                  std::vector<BevDetection3D> &ego_bbox, int32_t height,
                  int32_t width, float score_thresh = 0.f);

/**
 * read homography for ego to image coordinate
 * @param[out] homo: homography for ego to image coordinate
 * @param[in] file_path
 * @return 0 if success
 */
int read_bev_homo(std::vector<std::vector<std::vector<float>>> &homo,
                  std::string &file_path);

/**
 * get ego2img for ego to image coordinate
 * @param[out] ego2img:
 * @param[in] homo: homography for ego to image coordinate
 * @param[in] ori_height
 * @param[in] ori_width
 * @param[in] resize_height
 * @param[in] resize_width
 * @param[in] height
 * @param[in] width
 * @param[in] is_pad_resize
 */
void get_ego2img(std::vector<std::vector<std::vector<float>>> &ego2img,
                 std::vector<std::vector<std::vector<float>>> &homo,
                 int ori_height, int ori_width, int resize_height,
                 int resize_width, int height, int width,
                 bool is_pad_resize = false);

/**
 * Draw bev result to frame
 * @param[in] image_tensor: ImageTensor data
 * @param[in] bev_dets: detection result
 * @param[out] mat: (bgr or gray)
 * @param[in] ego2img: homography for ego to image coordinate
 * @return 0 if success
 */
void draw_bev_detection(ImageTensor *frame,
                        std::vector<BevDetection3D> &bev_dets,
                        std::vector<cv::Mat> &mat,
                        std::vector<std::vector<std::vector<float>>> &ego2img);

/**
 * Draw bev result to multi frame
 * @param[in] image_tensor: ImageTensor data
 * @param[in] bev_dets: detection result
 * @param[out] mat: (bgr or gray)
 * @param[in] ego2img: homography for ego to image coordinate
 * @return 0 if success
 */
void draw_bev_detection_mul_imgs(
    ImageTensor *frame, std::vector<BevDetection3D> &bev_dets,
    std::vector<cv::Mat> &mat,
    std::vector<std::vector<std::vector<float>>> &ego2img);

/**
 * Draw bev result to multi frame
 * @param[in] image_tensor: ImageTensor data
 * @param[in] map_dets: detection result
 * @param[out] mat: (bgr or gray)
 * * @param[out] image_lines: (bgr or gray)
 */
void draw_map_detection_mul_imgs(ImageTensor *frame,
                                 std::vector<MapDetection> &map_dets,
                                 std::vector<cv::Mat> &mat,
                                 const std::vector<float> &pc_range,
                                 cv::Mat &image_lines);

void draw_map_lines(std::vector<MapDetection> &mapDets,
                    const std::vector<float> &pc_range, cv::Mat &image);

/**
 * Create a colormap of Pascal labels
 * @param[out] colormap:
 */
void create_pascal_label_colormap(std::vector<std::vector<int32_t>> &colormap);

/**
 * draw bev segment
 * @param[in] frame: ImageTensor data
 * @param[in] segs: segment result
 * @param[in] colormap: color map
 * @param[out] mat (bgr or gray)
 */
void draw_bev_segment(ImageTensor *frame, Parsing<uint32_t> &segs,
                      std::vector<std::vector<int32_t>> &colormap,
                      cv::Mat &mat);

/**
 * Draw perception result to frame
 * @param[in] image_tensor: ImageTensor data
 * @param[in] perception: Perception result
 * @param[out] mat: (bgr or gray)
 * @return 0 if success
 */
int draw_perception(ImageTensor *image_tensor, Perception *perception,
                    cv::Mat &mat);

/**
 * draw lidar3d point cloud
 * @param[in] frame: ImageTensor data
 * @param[in] dets: lidar3d detection bbox
 * @param[out] mat (bgr or gray)
 */
void draw_lidar3d(ImageTensor *frame, std::vector<LidarDetection3D> &dets,
                  cv::Mat &mat);

/**
 * draw segment
 * @param[in] frame: ImageTensor data
 * @param[in] segs: segment result
 * @param[out] mat (bgr or gray)
 */
template <typename T>
void draw_segment(ImageTensor *frame, Parsing<T> &segs, cv::Mat &mat) {
  auto result_ptr = segs.seg.data();
  int width = segs.width;
  int height = segs.height;

  cv::Mat seg_img(height, width, CV_8UC3);
  uint8_t *seg_img_ptr = seg_img.ptr<uint8_t>();

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      int8_t id = static_cast<int8_t>(result_ptr[h * width + w]);
      if (id >= 19) continue;
      *seg_img_ptr++ = bgr_putpalette[id * 3];
      *seg_img_ptr++ = bgr_putpalette[id * 3 + 1];
      *seg_img_ptr++ = bgr_putpalette[id * 3 + 2];
    }
  }

  mat = std::move(seg_img);
}

/**
 * draw_lane
 * @param[out] image
 * @param[in] lane
 * @param[in] color
 * @param[in] markerSize
 * @param[in] lineWidth
 */
template <typename T>
void draw_lane(cv::Mat &image, const std::vector<std::vector<T>> &lane,
               const cv::Scalar &color, int markerSize = 1, int lineWidth = 1) {
  int numPoints = lane.size();
  for (int i = 0; i < numPoints - 1; ++i) {
    cv::Point pt1(lane[i][0] * 100 + 12800 / 2, 12800 / 2 - lane[i][1] * 100);
    cv::Point pt2(lane[i + 1][0] * 100 + 12800 / 2,
                  12800 / 2 - lane[i + 1][1] * 100);
    cv::line(image, pt1, pt2, color, lineWidth, cv::LINE_AA);
    cv::circle(image, pt1, markerSize, color, cv::FILLED);
  }
}

/**
 * draw_traj
 * @param[out] image
 * @param[in] traj
 * @param[in] color
 * @param[in] markerSize
 * @param[in] lineWidth
 */
template <typename T>
void draw_traj(cv::Mat &image, const std::vector<std::vector<T>> &traj,
               const cv::Scalar &color, int markerSize = 1, int lineWidth = 1) {
  int numPoints = traj.size();
  for (int i = 0; i < numPoints - 1; ++i) {
    cv::Point pt1(traj[i][0] * 100 + 12800 / 2, 12800 / 2 - traj[i][1] * 100);
    cv::Point pt2(traj[i + 1][0] * 100 + 12800 / 2,
                  12800 / 2 - traj[i + 1][1] * 100);
    cv::line(image, pt1, pt2, color, lineWidth, cv::LINE_AA);
    cv::drawMarker(image, pt1, color, cv::MARKER_TRIANGLE_UP, markerSize);
  }
}

/**
 * draw_pred
 * @param[out] image
 * @param[in] pred
 * @param[in] color
 * @param[in] markerSize
 * @param[in] lineWidth
 */
template <typename T>
void draw_pred(cv::Mat &image, const std::vector<std::vector<T>> &pred,
               const cv::Scalar &color, int markerSize = 1, int lineWidth = 1) {
  int numPoints = pred.size();
  for (int i = 0; i < numPoints - 1; ++i) {
    cv::Point pt1(pred[i][0] * 100 + 12800 / 2, 12800 / 2 - pred[i][1] * 100);
    cv::Point pt2(pred[i + 1][0] * 100 + 12800 / 2,
                  12800 / 2 - pred[i + 1][1] * 100);
    cv::line(image, pt1, pt2, color, lineWidth, cv::LINE_AA);
    cv::circle(image, pt1, markerSize, color, cv::FILLED);
  }
  // Draw the last marker for the last point
  cv::Point lastPt(pred[numPoints - 1][0], pred[numPoints - 1][1]);
  cv::circle(image, lastPt, markerSize, color, cv::FILLED);
}

/**
 * Visualize occ prediction
 * @param[out] seg3d: occ prediction result
 * @param[in] occ_bev_resized: occ visualization process result
 * @param[in] reverse_rgb: Reverse RGB<->BGR orders if `True`
 */
int draw_2d_occ(Parsing3d<uint32_t> &seg3d, cv::Mat &occ_bev_resized,
                bool reverse_rgb = false);

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_UTILS_IMAGE_UTILS_H_
