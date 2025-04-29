// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/image_list_output.h"

#include <cstdlib>
#include <fstream>

#include "glog/logging.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "utils/image_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_OUTPUT(image, ImageListOutputModule)

int ImageListOutputModule::Init(std::string config_file,
                                std::string config_string) {
  int ret_code = OutputModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  return 0;
}

int ImageListOutputModule::Init(rapidjson::Document &document) { return 0; }

void ImageListOutputModule::Write(ImageTensor *frame, Perception *perception) {
  if (output_flag_) {
    std::string cmd = "mkdir -p " + image_output_dir_;
    int result = system(cmd.c_str());
    if (result != 0) {
      VLOG(EXAMPLE_SYSTEM) << image_output_dir_
                           << " is error, please check image_output_dir "
                              "in workflow_latency.json";
      return;
    }

    std::stringstream ss;
    ss << image_output_dir_ << "/" << frame->frame_id;
    if (frame->image_name.empty()) {
      ss << ".jpg";
    } else {
      ss << '_' << frame->image_name;
    }

    if (perception->type == Perception::LIDARMULTITASK) {
      cv::Mat det_mat;
      std::vector<LidarDetection3D> dets = perception->lidar3d;
      std::string det_name = ss.str() + "_det.png";
      draw_lidar3d(frame, dets, det_mat);
      cv::imwrite(det_name, det_mat);

      cv::Mat seg_mat;
      Parsing<uint8_t> segs = perception->lidarSeg;
      std::string seg_name = ss.str() + "_seg.png";
      draw_segment(frame, segs, seg_mat);
      cv::imwrite(seg_name, seg_mat);

    } else if (perception->type == Perception::BEV) {
      std::vector<std::vector<std::vector<float>>> boston;
      std::vector<std::vector<std::vector<float>>> singapore;
      read_bev_homo(boston, bev_homo_boston_dir_);
      read_bev_homo(singapore, bev_homo_singapore_dir_);
      size_t pos = frame->image_name.find(".bin");
      std::string token = frame->image_name.substr(0, pos);
      if (bev_scenes_info_.HasMember(token.c_str())) {
        std::string scene = bev_scenes_info_[token.c_str()].GetString();
        VLOG(EXAMPLE_DETAIL) << "get scene [" << scene << "] with token ["
          << token << "] in image_name [" << frame->image_name << "]";
        std::vector<std::vector<std::vector<float>>> ego2img;
        if (scene == "boston-seaport") {
          get_ego2img(ego2img, boston, 900, 1600, frame->resize_height,
                      frame->resize_width, frame->input_height,
                      frame->input_width, frame->is_pad_resize);
        } else {
          get_ego2img(ego2img, singapore, 900, 1600, frame->resize_height,
                      frame->resize_width, frame->input_height,
                      frame->input_width, frame->is_pad_resize);
        }
        std::vector<cv::Mat> img(6);
        if (frame->num_img > 1) {
          draw_bev_detection_mul_imgs(frame, perception->bevDet3d, img,
                                      ego2img);
        } else {
          draw_bev_detection(frame, perception->bevDet3d, img, ego2img);
        }
        if (save_flag_) {
          for (int i = 0; i < 6; i++) {
            cv::imwrite(ss.str() + cam_names[i] + "_det.png", img[i]);
            
            std::string cmd = "mkdir -p raw_render/" + cam_names[i];
            int result = system(cmd.c_str());
            if (result != 0) {
              VLOG(EXAMPLE_SYSTEM) << "Do cmd: " << cmd << " failed";
            } else {
              cv::imwrite("raw_render/" + cam_names[i] + "/" + std::to_string(frame->frame_id) + ".png", img[i]);
            }
          }
        }

        if (!perception->bevSeg.seg.empty()) {
          std::vector<std::vector<float>> bev_bbox;
          bbox_ego2bev(bev_bbox, perception->bevDet3d);
          std::vector<std::vector<std::vector<float>>> corner_bbox;
          std::vector<float> scores;
          std::vector<int> labels;
          bbox_to_corner(corner_bbox, scores, labels, bev_bbox);
          float bev_size[3] = {51.2, 51.2, 0.8};
          float resize_ratio = 7.5;
          int height = static_cast<int>(bev_size[0] * 2 / bev_size[2]) * resize_ratio;
          int width = static_cast<int>(bev_size[1] * 2 / bev_size[2]) * resize_ratio;
          cv::Mat bev_mat(height, width, CV_8UC3);
          bev_mat.setTo(cv::Scalar(255, 255, 255));
          draw_bev_bbox(bev_mat, corner_bbox, 4, resize_ratio);
          if (save_flag_) {
            std::string bev_name = ss.str() + "_bev.png";
            cv::imwrite(bev_name, bev_mat);
            
            std::string cmd = "mkdir -p raw_render/BEV";
            int result = system(cmd.c_str());
            if (result != 0) {
              VLOG(EXAMPLE_SYSTEM) << "Do cmd: " << cmd << " failed";
            } else {
              cv::imwrite("raw_render/BEV/" + std::to_string(frame->frame_id) + ".png", bev_mat);
            }
          }

          cv::Mat seg_mat;
          Parsing<uint32_t> segs = perception->bevSeg;
          std::string seg_name = ss.str() + "_seg.png";
          draw_bev_segment(frame, segs, color_map_, seg_mat);
          if (save_flag_) {
            cv::imwrite(seg_name, seg_mat);
            std::string cmd = "mkdir -p raw_render/SEG";
            int result = system(cmd.c_str());
            if (result != 0) {
              VLOG(EXAMPLE_SYSTEM) << "Do cmd: " << cmd << " failed";
            } else {
              cv::imwrite("raw_render/SEG/" + std::to_string(frame->frame_id) + ".png", seg_mat);
            }
          }

          if (get_render_imgs_func_) {
            img.push_back(bev_mat);
            img.push_back(seg_mat);
            get_render_imgs_func_(img, frame, perception);
          }
        }

      } else {
        VLOG(EXAMPLE_SYSTEM) << "input file error, given file: " << token;
        VLOG(EXAMPLE_SYSTEM) << "get scene failed with token [" << token << "] in image_name [" << frame->image_name << "]";
      }

    } else if (perception->type == Perception::TRAJPRED) {
      auto &tarj_pred = perception->trajPred;
      std::vector<float> lane_feat_origin;
      LoadFromFile(lane_feat_origin, "../../../mini_data/argoverse1/" +
                                         frame->image_name + "-input1.bin");
      // only support draw one batch
      for (int i = 0; i < 1; i++) {
        cv::Mat img_flipped(12800, 12800, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat image;
        cv::flip(img_flipped, image, 0);
        std::cout << tarj_pred[i][0].single_bacth_name << std::endl;
        std::vector<float> traj_mask;
        LoadFromFile(traj_mask, "../../../mini_data/argoverse1/visualization/" +
                                    tarj_pred[i][0].single_bacth_name +
                                    "_traj_mask_vis.bin");
        std::vector<float> lane_mask;
        LoadFromFile(lane_mask, "../../../mini_data/argoverse1/visualization/" +
                                    tarj_pred[i][0].single_bacth_name +
                                    "_lane_mask_vis.bin");
        std::vector<float> feat_mask;  // (13,20)
        LoadFromFile(feat_mask, "../../../mini_data/argoverse1/visualization/" +
                                    tarj_pred[i][0].single_bacth_name +
                                    "_feat_mask_vis.bin");
        // lane_feat vaild
        std::vector<float> lane_feat_vaild;
        int lane_feat_c = frame->tensors[1].properties.stride[0] /
                          frame->tensors[1].properties.stride[1];
        int lane_feat_h = frame->tensors[1].properties.stride[1] /
                          frame->tensors[1].properties.stride[2];
        int lane_feat_w = frame->tensors[1].properties.stride[2] /
                          frame->tensors[1].properties.stride[3];
        std::vector<float> lane_feat_transpose(lane_feat_c * lane_feat_h *
                                               lane_feat_w);
        transposeArray(lane_feat_transpose.data(), lane_feat_origin.data(),
                       lane_feat_c, lane_feat_h, lane_feat_w);
        // batch offset
        int lane_feat_batch_offset =
            i * lane_feat_c * lane_feat_h * lane_feat_w;
        for (int lane_mask_indx = 0; lane_mask_indx < lane_mask.size();
             lane_mask_indx++) {
          if (lane_mask[lane_mask_indx] == 0) {
            // has transpose,so should use h and c
            int lane_feat_offset = lane_feat_batch_offset +
                                   lane_mask_indx * lane_feat_c * lane_feat_h;
            // has transpose,so should use h and c
            for (int lane_feat_transpose_offset = 0;
                 lane_feat_transpose_offset < lane_feat_h * lane_feat_c;
                 lane_feat_transpose_offset++) {
              lane_feat_vaild.push_back(
                  lane_feat_transpose[lane_feat_offset +
                                      lane_feat_transpose_offset]);
            }
          }
        }

        // traj_feat vaild
        std::vector<float> traj_feat_vaild;
        std::vector<float> traj_feat_origin;
        LoadFromFile(traj_feat_origin, "../../../mini_data/argoverse1/" +
                                           frame->image_name + "-input0.bin");
        int traj_feat_c = frame->tensors[0].properties.stride[0] /
                          frame->tensors[0].properties.stride[1];
        int traj_feat_h = frame->tensors[0].properties.stride[1] /
                          frame->tensors[0].properties.stride[2];
        int traj_feat_w = frame->tensors[0].properties.stride[2] /
                          frame->tensors[0].properties.stride[3];
        std::vector<float> traj_feat_transpose(traj_feat_c * traj_feat_h *
                                               traj_feat_w);
        transposeArray(traj_feat_transpose.data(), traj_feat_origin.data(),
                       traj_feat_c, traj_feat_h, traj_feat_w);

        // batch offset
        int traj_feat_batch_offset =
            i * traj_feat_c * traj_feat_h * traj_feat_w;

        for (int traj_feat_index = 0; traj_feat_index < traj_mask.size();
             traj_feat_index++) {
          if (traj_mask[traj_feat_index] == 0) {
            // has transpose,so should use h and c
            int traj_feat_offset = traj_feat_batch_offset +
                                   traj_feat_index * traj_feat_c * traj_feat_h;
            // has transpose,so should use h and c
            for (int traj_feat_transpose_offset = 0;
                 traj_feat_transpose_offset < traj_feat_h * traj_feat_c;
                 traj_feat_transpose_offset++) {
              traj_feat_vaild.push_back(
                  traj_feat_transpose[traj_feat_offset +
                                      traj_feat_transpose_offset]);
            }
          }
        }

        // process lane feat
        {
          cv::Scalar brownColor(42, 42, 165);
          int lanefeat_aligned_h = frame->tensors[1].properties.stride[1] /
                                   frame->tensors[1].properties.stride[2];
          int lanefeat_aligned_w = frame->tensors[1].properties.stride[0] /
                                   frame->tensors[1].properties.stride[1];
          int lanefeat_aligned_c = lane_feat_vaild.size() /
                                   (lanefeat_aligned_h * lanefeat_aligned_w);
          for (int c = 0; c < lanefeat_aligned_c; c++) {
            int offset_hw = c * lanefeat_aligned_h * lanefeat_aligned_w;
            std::vector<std::vector<float>> res_draw;
            std::vector<float> tmp_con;
            for (int h = 0; h < lanefeat_aligned_h; h++) {
              int offset_w = offset_hw + h * lanefeat_aligned_w;
              std::vector<float> tmp;
              for (int w = 0; w < lanefeat_aligned_w; w++) {
                if (w < 2) {
                  tmp.push_back(lane_feat_vaild[offset_w + w]);
                }
                if ((h == lanefeat_aligned_h - 2) && (w == 2 || w == 3)) {
                  tmp_con.push_back(lane_feat_vaild[offset_w + w]);
                }
              }
              res_draw.push_back(tmp);
            }
            res_draw.push_back(tmp_con);
            draw_lane(image, res_draw, brownColor, 14, 15);
          }
        }
        // process traj_feat
        {
          int traj_scale = 50;
          int trajfeat_aligned_h = frame->tensors[0].properties.stride[1] /
                                   frame->tensors[0].properties.stride[2];
          int trajfeat_aligned_w = frame->tensors[0].properties.stride[0] /
                                   frame->tensors[0].properties.stride[1];
          int trajfeat_aligned_c = traj_feat_vaild.size() /
                                   (trajfeat_aligned_h * trajfeat_aligned_w);
          std::vector<std::vector<std::vector<float>>> res_traj;
          for (int c = 0; c < trajfeat_aligned_c; c++) {
            int offset_hw = c * trajfeat_aligned_h * trajfeat_aligned_w;
            std::vector<std::vector<float>> res_draw;
            std::vector<float> tmp_con;
            for (int h = 0; h < trajfeat_aligned_h; h++) {
              int offset_w = offset_hw + h * trajfeat_aligned_w;
              std::vector<float> tmp;
              for (int w = 0; w < trajfeat_aligned_w; w++) {
                if (w < 2) {
                  tmp.push_back(traj_feat_vaild[offset_w + w] * traj_scale);
                }
                if ((h == trajfeat_aligned_h - 2) && (w == 2 || w == 3)) {
                  tmp_con.push_back(traj_feat_vaild[offset_w + w] * traj_scale);
                }
              }
              res_draw.push_back(tmp);
            }
            res_draw.push_back(tmp_con);
            res_traj.push_back(res_draw);
          }

          // other traj filter 40~
          cv::Scalar yellowColor(0, 255, 255);
          for (int other_indx = 0; other_indx < res_traj.size() - 2;
               other_indx++) {
            std::vector<std::vector<float>> other_traj;
            for (int mask_count = 20 * other_indx + 40;
                 mask_count < (20 * other_indx + 20 + 40); mask_count++) {
              if (feat_mask[mask_count] == 1)
                other_traj.push_back(
                    res_traj[other_indx]
                            [(mask_count - 20 * other_indx - 40) / 2]);
            }
            draw_traj(image, other_traj, yellowColor, 24, 64);
          }

          // agent traj filter 0~20
          cv::Scalar blueColor(255, 0, 0);
          std::vector<std::vector<float>> agent_traj;
          for (int mask_count = 0; mask_count < 20; mask_count++) {
            if (feat_mask[mask_count] == 1)
              agent_traj.push_back(res_traj[0][mask_count / 2]);
          }
          draw_traj(image, agent_traj, blueColor, 24, 64);

          // av traj filter 20~40
          cv::Scalar greenColor(0, 255, 0);
          std::vector<std::vector<float>> av_traj;
          for (int mask_count = 20; mask_count < 40; mask_count++) {
            if (feat_mask[mask_count] == 1)
              av_traj.push_back(res_traj[1][(mask_count - 20) / 2]);
          }
          draw_traj(image, av_traj, greenColor, 24, 64);
        }

        // process pred
        cv::Scalar redColor(0, 0, 255);

        std::vector<std::vector<float>> pred_res;
        // only need draw scores max
        for (int j = 0; j < 1; j++) {
          std::vector<float> tmp_x_y;
          for (int pred_indx = 0;
               pred_indx < perception->trajPred[i][j].traj.size();
               pred_indx += 2) {
            std::vector<float> tmp_x_y;
            tmp_x_y.push_back(perception->trajPred[i][j].traj[pred_indx]);
            tmp_x_y.push_back(perception->trajPred[i][j].traj[pred_indx + 1]);
            pred_res.push_back(tmp_x_y);
          }
        }
        draw_pred(image, pred_res, redColor, 24, 64);

        std::string pred_save_name =
            image_output_dir_ + "/" + std::to_string(frame->frame_id) + "_" +
            perception->trajPred[i][0].single_bacth_name + ".png";
        cv::imwrite(pred_save_name, image);
        std::cout << pred_save_name << " has drawed! " << std::endl;
      }
    } else if (perception->type == Perception::DEPTH) {
      size_t pos = ss.str().rfind('.');
      std::string img_name = ss.str().substr(0, pos);
      int32_t height = perception->pt.height;
      int32_t width = perception->pt.width;
      float baseline = 0.54;
      int32_t f = 1050;
      double alpha = 11.0;
      std::vector<float> &disparity = perception->pt.point;
      std::vector<float> depth(disparity.size());
      for (int32_t i = 0; i < disparity.size(); i++) {
        depth[i] = baseline * f / disparity[i];
      }

      cv::Mat disparity_mat(height, width, CV_32F, disparity.data());
      cv::Mat depth_mat(height, width, CV_32F, depth.data());

      cv::Mat enhanced_disparity;
      cv::convertScaleAbs(disparity_mat, enhanced_disparity, alpha);
      cv::Mat enhanced_depth;
      cv::convertScaleAbs(depth_mat, enhanced_depth, alpha);

      cv::Mat disp_mat;
      cv::applyColorMap(enhanced_disparity, disp_mat, cv::COLORMAP_JET);
      std::string disparity_name = img_name + "_disparity.png";
      cv::imwrite(disparity_name, disp_mat);

      cv::Mat deep_mat;
      cv::applyColorMap(enhanced_depth, deep_mat, cv::COLORMAP_JET);
      std::string deep_name = img_name + "_depth.png";
      cv::imwrite(deep_name, deep_mat);
    } else if (perception->type == Perception::SEG3D) {
      cv::Mat occ_bev_vis;
      bool reverse_rgb = true;
      draw_2d_occ(perception->seg3d, occ_bev_vis, reverse_rgb);
      cv::imwrite(ss.str() + "occ_vis.jpg", occ_bev_vis);
    } else if (perception->type == Perception::MAP) {
      std::vector<cv::Mat> img(6);
      cv::Mat img_lines = cv::Mat::zeros(600, 1200, CV_8UC3);
      draw_map_detection_mul_imgs(frame, perception->mapDet, img, bev_range_,
                                  img_lines);
      for (int i = 0; i < 6; i++) {
        cv::imwrite(ss.str() + cam_names[i] + "_det.png", img[i]);
      }
      cv::imwrite(ss.str() + "_lines.png", img_lines);
    } else {
      cv::Mat mat;
      if (draw_perception(frame, perception, mat) == 0) {
        if (perception->type == Perception::POINT ||
            perception->type == Perception::DET3D ||
            perception->type == Perception::LIDAR3D) {
          ss << ".png";
        }
        if (get_render_imgs_func_) {
          std::vector<cv::Mat> imgs;
          imgs.push_back(mat);
          get_render_imgs_func_(imgs, frame, perception);
        }
        if (save_flag_) {
          // 输出文件名
          VLOG(EXAMPLE_SYSTEM) << "save file: " << ss.str();
          cv::imwrite(ss.str(), mat);
        }
      }
    }
    image_counter_++;
  }
}

int ImageListOutputModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("enable_view_output")) {
    output_flag_ = document["enable_view_output"].GetBool();
  }

  if (document.HasMember("enable_save_output")) {
    save_flag_ = document["enable_save_output"].GetBool();
  }

  VLOG(EXAMPLE_SYSTEM) << "save_flag: " << save_flag_;

  if (document.HasMember("view_output_dir")) {
    image_output_dir_ = document["view_output_dir"].GetString();
  }

  if (document.HasMember("bev_range")) {
    auto bev_range = document["bev_range"].GetArray();
    bev_range_.resize(bev_range.Size());
    for (int i = 0; i < bev_range.Size(); i++) {
      bev_range_[i] = bev_range[i].GetFloat();
    }
  }

  if (document.HasMember("bev_ego2img_info")) {
    auto file_list = document["bev_ego2img_info"].GetArray();
    if (file_list.Size() != 3) {
      VLOG(EXAMPLE_SYSTEM) << "bev homography and scenes given error !";
      return -1;
    }

    std::ifstream ifs(file_list[0].GetString());
    if (!ifs) {
      VLOG(EXAMPLE_SYSTEM) << "Open bev scenes file "
                           << file_list[0].GetString() << " failed";
      return -1;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    bev_scenes_info_.Parse(buffer.str().data());
    ifs.close();
    if (bev_scenes_info_.HasParseError()) {
      VLOG(EXAMPLE_SYSTEM) << "Parsing bev scenes config file failed "
                           << bev_scenes_info_.HasParseError();
      return -1;
    }
    bev_homo_boston_dir_ = file_list[1].GetString();
    bev_homo_singapore_dir_ = file_list[2].GetString();

    create_pascal_label_colormap(color_map_);
  }

  return 0;
}
