// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/raw_output.h"

#include <iterator>

#include "glog/logging.h"
#include "rapidjson/document.h"

DEFINE_AND_REGISTER_OUTPUT(eval, RawOutputModule)

int RawOutputModule::Init(std::string config_file, std::string config_string) {
  int ret_code = OutputModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  if (!output_file_.empty()) {
    // remove old log, and create new log
    ofs_.open(output_file_.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << output_file_ << " failed";
    }
    ofs_.close();
    // 'std::ios::app' is sure the res can write directly, close cache func.
    //  otherwise the previous mem will be modified when writing very quickly
    ofs_.open(output_file_.c_str(), std::ios::out | std::ios::app);
    if (!ofs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << output_file_ << " failed";
    }
  }

  if (!mask_file_.empty()) {
    mask_ofs_.open(mask_file_.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << mask_file_ << " failed";
    }
  }

  if (multitask_output_file_.size() != 0) {
    det_ofs_.open(multitask_output_file_[0].c_str(),
                  std::ios::out | std::ios::trunc);
    if (!det_ofs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << multitask_output_file_[0] << " failed";
    }

    seg_ofs_.open(multitask_output_file_[1].c_str(),
                  std::ios::out | std::ios::trunc);
    if (!seg_ofs_.is_open()) {
      VLOG(EXAMPLE_SYSTEM) << "Open " << multitask_output_file_[1] << " failed";
    }
  }

  return 0;
}

int RawOutputModule::Init(rapidjson::Document &document) { return 0; }

void RawOutputModule::Write(ImageTensor *frame, Perception *perception) {
  switch (perception->type) {
    case Perception::DET:
      WriteDetLog(frame, perception);
      break;
    case Perception::CLS:
      WriteClsLog(frame, perception);
      break;
    case Perception::SEG:
      WriteParsingLog(frame, perception);
      break;
    case Perception::SEG3D:
      WriteSeg3dLog(frame, perception);
      break;
    case Perception::MASK:
      WriteMaskLog(frame, perception);
      break;
    case Perception::POINT:
      WriteOpticalFlowLog(frame, perception);
      break;
    case Perception::DET3D:
      WriteDet3DLog(frame, perception);
      break;
    case Perception::KEYPOINT:
      WriteKeyPointLog(frame, perception);
      break;
    case Perception::DETCAM3D:
      WriteKeyDetCam3DLog(frame, perception);
      break;
    case Perception::LIDAR3D:
      WriteLidar3DLog(frame, perception);
      break;
    case Perception::BEV:
      WriteBevLog(frame, perception);
      break;
    case Perception::LIDARMULTITASK:
      WriteMultiTaskLog(frame, perception);
      break;
    case Perception::TRAJPRED:
      WriteTrajPredLog(frame, perception);
      break;
    case Perception::TRAJPRED_qcnet:
      WriteQCNetPredLog(frame, perception);
      break;
    case Perception::DEPTH:
      WriteDepthLog(frame, perception);
      break;
    case Perception::MAP:
      WriteMapLog(frame, perception);
      break;
    default: {
      VLOG(EXAMPLE_SYSTEM) << "invaild type in write raw file, the type is : "
                           << perception->type;
    }
  }
}

void RawOutputModule::WriteDetLog(ImageTensor *frame, Perception *perception) {
  std::stringstream result_log;

  result_log << std::setprecision(18)
             << "input_image_name: " << frame->image_name << " ";
  for (auto &box : perception->det) {
    result_log << box.bbox.xmin << "," << box.bbox.ymin << "," << box.bbox.xmax
               << "," << box.bbox.ymax << "," << box.score << "," << box.id
               << " ";
  }
  ofs_ << result_log.str() << std::endl;
}

void RawOutputModule::WriteMaskLog(ImageTensor *frame, Perception *perception) {
  std::stringstream result_log;

  // write mask
  mask_ofs_.write(reinterpret_cast<char *>(perception->mask.mask_info.data()),
                  perception->mask.mask_info.size() * sizeof(float));

  result_log << std::setprecision(18)
             << "input_image_name: " << frame->image_name << " ";
  for (auto &box : perception->mask.det_info) {
    result_log << box.bbox.xmin << "," << box.bbox.ymin << "," << box.bbox.xmax
               << "," << box.bbox.ymax << "," << box.score << "," << box.id
               << " ";
  }
  ofs_ << result_log.str() << std::endl;
}

void RawOutputModule::WriteClsLog(ImageTensor *frame, Perception *perception) {
  std::string log_str = std::string("input_image_name: ") + frame->image_name;

  for (auto cls : perception->cls) {
    if (cls.id < 0) break;
    log_str += std::string(" class_id: ") + std::to_string(cls.id) +
               std::string(" class_name: ") + cls.class_name;
  }

  printf("the str is %s\n", log_str.c_str());
  ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteParsingLog(ImageTensor *frame,
                                      Perception *perception) {
  auto result_ptr = static_cast<int8_t *>(malloc(perception->seg.seg.size()));
  std::string log_str;
  int recorded_item_count = 0;
  int block_index = 0;
  for (auto block : perception->seg.seg) {
    if (log_str.length() == 0)
      log_str = std::string("input_image_name: ") + frame->image_name +
                std::string("_block_") + std::to_string(block_index) +
                std::string(" ");
    log_str += std::to_string(block) + " ";
    recorded_item_count++;
    if (recorded_item_count >= 8000) {
      recorded_item_count = 0;
      ofs_ << log_str << std::endl;
      log_str = "";
      block_index += 1;
    }
  }
  if (log_str.length() > 0) ofs_ << log_str << std::endl;

  free(result_ptr);
}

void RawOutputModule::WriteOpticalFlowLog(ImageTensor *frame,
                                          Perception *perception) {
  float *flow_chw = perception->pt.point.data();
  int c_stride = perception->pt.height * perception->pt.width;
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::string log_str;
  for (uint32_t i = 0; i < c_stride * 2; ++i) {
    if (log_str.length() == 0) {
      log_str = std::string("input_image_name: ") + frame->image_name +
                std::string(" ");
    }
    log_str += std::to_string(flow_chw[i]) + " ";
  }

  if (log_str.length() > 0) ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteDet3DLog(ImageTensor *frame,
                                    Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::string log_str = frame->image_name + ": ";
  for (auto &rst : perception->det3d) {
    log_str += std::to_string(rst.bbox.x) + " " + std::to_string(rst.bbox.y) +
               " " + std::to_string(rst.bbox.z) + " " +
               std::to_string(rst.bbox.w) + " " + std::to_string(rst.bbox.l) +
               " " + std::to_string(rst.bbox.h) + " " +
               std::to_string(rst.bbox.r) + " " + std::to_string(rst.score) +
               " " + std::to_string(rst.top_label) + "; ";
  }
  ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteKeyPointLog(ImageTensor *frame,
                                       Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::string log_str = frame->image_name + ":" +
                        std::to_string(perception->kpt.groups_size) + ";";
  for (int i = 0; i < perception->kpt.groups_size; i++) {
    for (auto &kpts : perception->kpt.points[i]) {
      log_str +=
          std::to_string(kpts.first) + " " + std::to_string(kpts.second) + " ";
    }
    log_str += ";";
  }
  ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteKeyDetCam3DLog(ImageTensor *frame,
                                          Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::string log_str = frame->image_name + ":";
  for (auto &rst : perception->detcam3d) {
    log_str += std::to_string(rst.box_center_.dim1_) + " " +
               std::to_string(rst.box_center_.dim2_) + " " +
               std::to_string(rst.box_center_.dim3_) + " " +
               std::to_string(rst.box_dim_.dim1_) + " " +
               std::to_string(rst.box_dim_.dim2_) + " " +
               std::to_string(rst.box_dim_.dim3_) + " " +
               std::to_string(rst.box_yaw_) + " " +
               std::to_string(rst.velocity_.dim1_) + " " +
               std::to_string(rst.velocity_.dim2_) + " " +
               std::to_string(rst.score_) + " " + std::to_string(rst.label_) +
               " " + std::to_string(rst.attr_) + ";";
  }
  ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteLidar3DLog(ImageTensor *frame,
                                      Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::stringstream result_log;
  result_log << std::setprecision(10) << frame->image_name << ":";
  for (auto &rst : perception->lidar3d) {
    result_log << rst.label << " " << rst.score << " " << rst.bbox.xs << " "
               << rst.bbox.ys << " " << rst.bbox.height << " " << rst.bbox.dim_0
               << " " << rst.bbox.dim_1 << " " << rst.bbox.dim_2 << " "
               << rst.bbox.rot << " " << rst.bbox.vel_0 << " " << rst.bbox.vel_1
               << "; ";
  }
  ofs_ << result_log.str() << std::endl;
}

void RawOutputModule::WriteBevLog(ImageTensor *frame, Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;

  if (multitask_output_file_.size() != 0) {
    // write bev_det.log
    std::stringstream det_log;
    det_log << std::setprecision(10) << std::fixed << frame->image_name << ":";
    for (auto &rst : perception->bevDet3d) {
      det_log << rst.bbox.xs << " " << rst.bbox.ys << " " << rst.bbox.height
              << " " << rst.bbox.dim_0 << " " << rst.bbox.dim_1 << " "
              << rst.bbox.dim_2 << " " << rst.bbox.rot << " " << rst.bbox.vel_0
              << " " << rst.bbox.vel_1 << " " << rst.score << " " << rst.label
              << "; ";
    }
    det_ofs_ << det_log.str() << std::endl;

    // write bev_seg.log
    auto result_ptr =
        static_cast<int32_t *>(malloc(perception->bevSeg.seg.size()));
    std::string seg_log;
    int recorded_item_count = 0;
    int block_index = 0;
    for (auto block : perception->bevSeg.seg) {
      if (seg_log.length() == 0)
        seg_log = std::string("input_image_name: ") + frame->image_name +
                  std::string("_block_") + std::to_string(block_index) +
                  std::string(" ");
      seg_log += std::to_string(block) + " ";
      recorded_item_count++;
      if (recorded_item_count >= 8000) {
        recorded_item_count = 0;
        seg_ofs_ << seg_log << std::endl;
        seg_log = "";
        block_index += 1;
      }
    }
    if (seg_log.length() > 0) seg_ofs_ << seg_log << std::endl;

    free(result_ptr);
  } else {
    // write bev single task.log
    std::stringstream det_log;
    det_log << std::setprecision(10) << std::fixed << frame->image_name << ":";
    for (auto &rst : perception->bevDet3d) {
      det_log << rst.bbox.xs << " " << rst.bbox.ys << " " << rst.bbox.height
              << " " << rst.bbox.dim_0 << " " << rst.bbox.dim_1 << " "
              << rst.bbox.dim_2 << " " << rst.bbox.rot << " " << rst.bbox.vel_0
              << " " << rst.bbox.vel_1 << " " << rst.score << " " << rst.label
              << "; ";
    }
    ofs_ << det_log.str() << std::endl;
  }
}

void RawOutputModule::WriteSeg3dLog(ImageTensor *frame,
                                    Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  // write 3d_seg.log
  std::string seg_log;
  int recorded_item_count = 0;
  int block_index = 0;
  for (auto block : perception->seg3d.seg) {
    if (seg_log.length() == 0)
      seg_log = std::string("input_image_name: ") + frame->image_name +
                std::string("_block_") + std::to_string(block_index) +
                std::string(" ");
    seg_log += std::to_string(block) + " ";
    recorded_item_count++;
    if (recorded_item_count >= 8000) {
      recorded_item_count = 0;
      ofs_ << seg_log << std::endl;
      seg_log = "";
      block_index += 1;
    }
  }
  if (seg_log.length() > 0) ofs_ << seg_log << std::endl;
}

void RawOutputModule::WriteMultiTaskLog(ImageTensor *frame,
                                        Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "input_name: " << frame->image_name;

  // write det.log
  std::stringstream det_log;
  det_log << std::setprecision(10) << frame->image_name << ":";
  for (auto &rst : perception->lidar3d) {
    det_log << rst.label << " " << rst.score << " " << rst.bbox.xs << " "
            << rst.bbox.ys << " " << rst.bbox.height << " " << rst.bbox.dim_0
            << " " << rst.bbox.dim_1 << " " << rst.bbox.dim_2 << " "
            << rst.bbox.rot << " " << rst.bbox.vel_0 << " " << rst.bbox.vel_1
            << "; ";
  }
  det_ofs_ << det_log.str() << std::endl;

  // write seg.log
  auto result_ptr =
      static_cast<uint8_t *>(malloc(perception->lidarSeg.seg.size()));
  std::string seg_log;
  int recorded_item_count = 0;
  int block_index = 0;
  for (auto block : perception->lidarSeg.seg) {
    if (seg_log.length() == 0)
      seg_log = std::string("input_name: ") + frame->image_name +
                std::string("_block_") + std::to_string(block_index) +
                std::string(" ");
    seg_log += std::to_string(block) + " ";
    recorded_item_count++;
    if (recorded_item_count >= 8000) {
      recorded_item_count = 0;
      seg_ofs_ << seg_log << std::endl;
      seg_log = "";
      block_index += 1;
    }
  }
  if (seg_log.length() > 0) seg_ofs_ << seg_log << std::endl;

  free(result_ptr);
}

void RawOutputModule::WriteTrajPredLog(ImageTensor *frame,
                                       Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;
  std::string log_str = "";
  for (int i = 0; i < perception->trajPred.size(); i++) {
    for (int j = 0; j < perception->trajPred[i].size(); j++) {
      if (j == 0) {
        log_str += perception->trajPred[i][j].single_bacth_name + ":";
      }
      for (auto &kpts : perception->trajPred[i][j].traj) {
        log_str += std::to_string(kpts) + " ";
      }
      log_str += ",";
      log_str += std::to_string(perception->trajPred[i][j].score);
      log_str += "; ";
    }
    log_str += "\n";
  }
  ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteQCNetPredLog(ImageTensor *frame,
                                        Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "raw_output WriteQCNetPredLog image_name: "
                       << frame->image_name;
  std::string log_str = "";
  std::string log_str_score = frame->image_name + "_frame_id_PerSampleInfer-" +
                              std::to_string(frame->frames_per_sample_infer) +
                              "_scores (30, 6)";
  log_str_score += "\n";
  log_str += frame->image_name + "_frame_id_PerSampleInfer-" +
             std::to_string(frame->frames_per_sample_infer) +
             "_traj(30, 6, 12, 2)";
  log_str += "\n";
  for (int i = 0; i < perception->qcnet.size(); i++) {
    for (int j = 0; j < perception->qcnet[i].size(); j++) {
      log_str_score += std::to_string(perception->qcnet[i][j].score) + " ";
      for (int k = 0; k < perception->qcnet[i][j].traj.size(); k++) {
        for (auto &kpts : perception->qcnet[i][j].traj[k]) {
          log_str += std::to_string(kpts) + " ";
        }
      }
      log_str += "\n";
    }
  }
  ofs_ << log_str << std::endl;
  ofs_ << log_str_score << std::endl;
}
void RawOutputModule::WriteDepthLog(ImageTensor *frame,
                                    Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "input_name: " << frame->image_name;
  std::string log_str;
  int recorded_item_count = 0;
  int block_index = 0;
  for (auto block : perception->pt.point) {
    if (log_str.length() == 0)
      log_str = std::string("input_name: ") + frame->image_name +
                std::string("_block_") + std::to_string(block_index) +
                std::string(" ");
    log_str += std::to_string(block) + " ";
    recorded_item_count++;
    if (recorded_item_count >= 8000) {
      recorded_item_count = 0;
      ofs_ << log_str << std::endl;
      log_str = "";
      block_index += 1;
    }
  }
  if (log_str.length() > 0) ofs_ << log_str << std::endl;
}

void RawOutputModule::WriteMapLog(ImageTensor *frame, Perception *perception) {
  VLOG(EXAMPLE_SYSTEM) << "image_name: " << frame->image_name;

  // write bev single task.log
  std::stringstream det_log;
  det_log << std::setprecision(10) << std::fixed << frame->image_name << ":";
  for (auto &rst : perception->mapDet) {
    for (auto &pt : rst.pts) {
      det_log << pt.first << " " << pt.second << " ";
    }
    det_log << rst.score << " " << rst.label << ";";
  }
  ofs_ << det_log.str() << std::endl;
}

int RawOutputModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("output_file")) {
    output_file_ = document["output_file"].GetString();
  }

  if (document.HasMember("mask_output_file")) {
    mask_file_ = document["mask_output_file"].GetString();
  }

  if (document.HasMember("multitask_output_file")) {
    auto file_list = document["multitask_output_file"].GetArray();
    multitask_output_file_.resize(file_list.Size());
    for (int i = 0; i < file_list.Size(); i++) {
      multitask_output_file_[i] = file_list[i].GetString();
    }
  }

  return 0;
}

RawOutputModule::~RawOutputModule() {
  if (ofs_.is_open()) {
    ofs_.close();
  }

  if (mask_ofs_.is_open()) {
    mask_ofs_.close();
  }

  if (det_ofs_.is_open()) {
    det_ofs_.close();
  }

  if (seg_ofs_.is_open()) {
    seg_ofs_.close();
  }
}
