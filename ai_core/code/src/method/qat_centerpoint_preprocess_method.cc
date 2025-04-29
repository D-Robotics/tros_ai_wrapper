// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_centerpoint_preprocess_method.h"

#include <limits.h>

#include <cfenv>
#include <chrono>
#include <ctime>
#include <iostream>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"

DEFINE_AND_REGISTER_METHOD(QATCenterPointPreProcessMethod);

struct FeatureInfo {
  uint16_t idx;
  uint16_t idy;
  int32_t xyzr;
  int32_t t;
};

namespace __detail {
inline float round(float const input) {
  std::fesetround(FE_TONEAREST);
  float const result{std::nearbyintf(input)};
  return result;
}
}  // namespace __detail

int32_t QATCenterPointPreProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATCenterPointPreProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());
  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }
  
  float back = document["back"].GetFloat();

  float front = document["front"].GetFloat();

  float right = document["right"].GetFloat();

  float left = document["left"].GetFloat();

  float bottom = document["bottom"].GetFloat();

  float top = document["top"].GetFloat();

  float r_lower = document["r_lower"].GetFloat();

  float r_upper = document["r_upper"].GetFloat();

  float x_scale = document["x_scale"].GetFloat();

  float y_scale = document["y_scale"].GetFloat();

  int32_t max_num_point_pillar = document["max_num_point_pillar"].GetInt();

  int32_t max_num_point = document["max_num_point"].GetInt();

  int32_t align_padding_point = document["align_padding_point"].GetInt();

  int32_t dim = document["dim"].GetInt();

  bool run_on_dsp = false;
  if (document.HasMember("run_on_dsp")) {
    run_on_dsp = document["run_on_dsp"].GetBool();
  }
  
  config_ =
      new VoxelConfig(back, front, right, left, bottom, top, r_lower, r_upper,
                      x_scale, y_scale, max_num_point_pillar, max_num_point,
                      dim, run_on_dsp, align_padding_point);

  if (run_on_dsp) {
    hbUCPMallocCached(&point_cloud_data_mem_,
                      300000 * config_->kdim * sizeof(float), 0);
    hbUCPMalloc(&voxel_mem_,
                config_->align_padding_point * config_->kmax_num_point_pillar *
                    config_->kdim * sizeof(int8_t),
                0);
    hbUCPMallocCached(&pillar_id_mem_, 300000 * sizeof(FeatureInfo), 0);
    hbUCPMalloc(&task_spec_mem_, sizeof(DSPPointPillarPreProcessParam), 0);

    coord_to_voxel_id_.resize(config_->kx_length);
    for (auto &&item : coord_to_voxel_id_) {
      item.resize(config_->ky_width);
    }
#ifdef HAVE_DSP
    hbDSPAddrMap(&point_cloud_data_mem_, &point_cloud_data_mem_);
    hbDSPAddrMap(&voxel_mem_, &voxel_mem_);
    hbDSPAddrMap(&pillar_id_mem_, &pillar_id_mem_);
    hbDSPAddrMap(&task_spec_mem_, &task_spec_mem_);
#endif
  } else {
    point_cloud_data_ =
        static_cast<float *>(malloc(300000 * config_->kdim * sizeof(float)));
    voxel_data_ = static_cast<float *>(
        malloc(config_->kmax_num_point * config_->kmax_num_point_pillar *
               config_->kdim * sizeof(float)));
    features_s8_ = static_cast<int8_t *>(
        malloc(config_->align_padding_point * config_->kmax_num_point_pillar *
               config_->kdim * sizeof(int8_t)));
  }

  VLOG(EXAMPLE_DEBUG) << "VoxelConfig: { "
                         "kback_border: "
                      << config_->kback_border
                      << ", kfront_border: " << config_->kfront_border
                      << ", kright_border: " << config_->kright_border
                      << ", kleft_border: " << config_->kleft_border
                      << ", kbottom_border: " << config_->kbottom_border
                      << ", ktop_border: " << config_->ktop_border
                      << ", kr_lower: " << config_->kr_lower
                      << ", kr_upper: " << config_->kr_upper
                      << ", kx_range: " << config_->kx_range
                      << ", ky_range: " << config_->ky_range
                      << ", kz_range: " << config_->kz_range
                      << ", kr_range: " << config_->kr_range
                      << ", kx_scale: " << config_->kx_scale
                      << ", ky_scale: " << config_->ky_scale
                      << ", kx_length: " << config_->kx_length
                      << ", ky_width: " << config_->ky_width
                      << ", kmax_num_point_pillar: "
                      << config_->kmax_num_point_pillar
                      << ", kmax_num_point: " << config_->kmax_num_point
                      << ", kdim: " << config_->kdim
                      << ", krun_on_dsp: " << config_->krun_on_dsp << " }";

  return 0;
}

int32_t QATCenterPointPreProcessMethod::FetchBinary(
    float *buffer, int32_t &num_points, std::string const &point_cloud_path,
    int32_t length) {
  FILE *stream;
  if (point_cloud_path.empty()) {
    VLOG(EXAMPLE_REPORT) << "bin file is empty! " << point_cloud_path;
    return -1;
  }

  stream = fopen(point_cloud_path.c_str(), "rb");
  if (!stream) {
    VLOG(EXAMPLE_REPORT) << "bin file open failed! " << point_cloud_path;
    return -1;
  }

  size_t ret_code = fread(buffer, sizeof(float), length, stream);
  num_points = ret_code / config_->kdim;
  fclose(stream);
  return 0;
}

void QATCenterPointPreProcessMethod::Reset() {
  // Reset # of points in each pillar.
  if (!config_->krun_on_dsp) {
    memset(config_->pillar_point_num, 0, config_->kmax_num_point * sizeof(int));

    memset(config_->coor_to_voxel_idx, -1,
           config_->kx_length * config_->ky_width * sizeof(int));

    memset(config_->coors, -1, config_->kmax_num_point * 4 * sizeof(int));
    for (int32_t i = 0; i < config_->kmax_num_point; i++) {
      config_->coors[i * 4] = 0;
    }

    memset(voxel_data_, 0,
           config_->kmax_num_point * config_->kmax_num_point_pillar *
               config_->kdim * sizeof(float));

    memset(features_s8_, 0,
           config_->align_padding_point * config_->kmax_num_point_pillar *
               config_->kdim * sizeof(int8_t));
  }
  voxel_num_ = 0;
}

void QATCenterPointPreProcessMethod::GenVoxel(int start, int end) {
  for (int i = start; i < end; i++) {
    float *point;
    point = point_cloud_data_ + i * config_->kdim;

    float &point_x = point[0];
    float &point_y = point[1];
    float &point_z = point[2];
    float &point_r = point[3];

    if (point_x <= config_->kback_border || point_x >= config_->kfront_border ||
        point_y <= config_->kright_border || point_y >= config_->kleft_border ||
        point_z <= config_->kbottom_border || point_z >= config_->ktop_border) {
      continue;
    }

    int idx = (point_x - config_->kback_border) / config_->kx_scale;
    int idy = (point_y - config_->kright_border) / config_->ky_scale;

    // zyx
    int pillar_index = idy * config_->kx_length + idx;
    int voxelidx = config_->coor_to_voxel_idx[pillar_index];
    if (voxelidx == -1) {
      if (voxel_num_ < config_->kmax_num_point) {
        voxelidx = voxel_num_;
        voxel_num_ += 1;
      } else {
        voxelidx = config_->kmax_num_point - 1;
      }
      config_->coor_to_voxel_idx[pillar_index] = voxelidx;
      config_->coors[voxelidx * 4 + 1] = 0;
      config_->coors[voxelidx * 4 + 2] = idy;
      config_->coors[voxelidx * 4 + 3] = idx;
    }

    if (config_->pillar_point_num[voxelidx] >= config_->kmax_num_point_pillar) {
      continue;
    } else {
      auto total_offset = (config_->kmax_num_point_pillar * voxelidx +
                           config_->pillar_point_num[voxelidx]) *
                          config_->kdim;
      config_->pillar_point_num[voxelidx] += 1;

      *(voxel_data_ + total_offset) = point_x;
      *(voxel_data_ + total_offset + 1) = point_y;
      *(voxel_data_ + total_offset + 2) = point_z;
      *(voxel_data_ + total_offset + 3) = point_r;
      *(voxel_data_ + total_offset + 4) = point[4];
    }
  }
}

void QATCenterPointPreProcessMethod::GenFeatureDim5(float scale) {
  for (int i = 0; i < voxel_num_; i++) {
    int idx = i * config_->kmax_num_point_pillar * config_->kdim;
    for (int j = 0; j < config_->kmax_num_point_pillar; ++j) {
      if (config_->pillar_point_num[i] >
          config_->kmax_num_point_pillar_vec[j]) {
        int index = idx + j * config_->kdim;
        voxel_data_[index + 0] =
            (voxel_data_[index + 0] - config_->kback_border) /
            config_->kx_range / scale;
        voxel_data_[index + 1] =
            (voxel_data_[index + 1] - config_->kright_border) /
            config_->ky_range / scale;
        voxel_data_[index + 2] =
            (voxel_data_[index + 2] - config_->kbottom_border) /
            config_->kz_range / scale;
        voxel_data_[index + 3] = (voxel_data_[index + 3] - config_->kr_lower) /
                                 config_->kr_range / scale;
        if (voxel_data_[index + 4] != 0) {
          voxel_data_[index + 4] = voxel_data_[index + 4] / scale;
        }
      }
    }
  }
}

void QATCenterPointPreProcessMethod::TransposeDim5(int model_aligned_w) {
  // Transpose to 1x5x20x40000
  int kWC = config_->kmax_num_point_pillar * config_->kdim;
  int kHW = config_->kmax_num_point * config_->kmax_num_point_pillar;
  // there c h w is not the model real nchw, just define the order index
  for (int c = 0; c < config_->kdim; ++c) {
    int input_wh = c * config_->kmax_num_point_pillar * model_aligned_w;
    for (int w = 0; w < config_->kmax_num_point_pillar; ++w) {
      int input_h = input_wh + w * model_aligned_w;
      for (int h = 0; h < voxel_num_; ++h) {
        int old_index = h * kWC + w * config_->kdim + c;
        int new_index = h + input_h;
        float features_tmp =
            __detail::round(static_cast<float>(voxel_data_[old_index]));
        features_tmp = std::min(std::max(features_tmp, -128.f), 127.f);
        features_s8_[new_index] = static_cast<int8_t>(features_tmp);
      }
    }
  }
}

#ifdef HAVE_DSP
int32_t QATCenterPointPreProcessMethod::CallDspRpc(
    hbUCPTaskHandle_t &task, DSPPointPillarPreProcessParam &param,
    int32_t rpc_cmd) {
  memcpy(task_spec_mem_.virAddr, &param, sizeof(DSPPointPillarPreProcessParam));
  int32_t ret = hbDSPRpcV2(&task, &task_spec_mem_, nullptr, rpc_cmd);
  if (ret != 0) {
    VLOG(EXAMPLE_SYSTEM) << "hbDSPRpcV2 failed, errno: " << ret;
    VLOG(EXAMPLE_SYSTEM)
        << "Please check whether the dsp image is deployed correctly"
        << ", you can run dsp_deploy.sh and retry again.";
    return ret;
  }
  hbUCPSchedParam sched_param;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
  ret = hbUCPSubmitTask(task, &sched_param);
  if (ret != 0) {
    VLOG(EXAMPLE_SYSTEM) << "hbUCPSubmitTask failed, errno: " << ret;
    VLOG(EXAMPLE_SYSTEM)
        << "Please check whether the dsp image is deployed correctly"
        << ", you can run dsp_deploy.sh and retry again.";
    static_cast<void>(hbUCPReleaseTask(task));
    return ret;
  }

  ret = hbUCPWaitTaskDone(task, 2000);
  if (ret != 0) {
    VLOG(EXAMPLE_SYSTEM) << "hbUCPWaitTaskDone failed, errno: " << ret;
    VLOG(EXAMPLE_SYSTEM)
        << "Please check whether the dsp image is deployed correctly"
        << ", you can run dsp_deploy.sh and retry again.";
    static_cast<void>(hbUCPReleaseTask(task));
    return ret;
  }

  ret = hbUCPReleaseTask(task);
  if (ret != 0) {
    VLOG(EXAMPLE_SYSTEM) << "hbUCPReleaseTask failed, errno: " << ret;
  }
  return ret;
}
#endif

void QATCenterPointPreProcessMethod::GenVoxelForDsp(int32_t *coords,
                                                    int32_t num_points) {
  // init coordinatet to voxel id map
  static VoxelInfo const invalid_voxel_info{0U, 0xFFFFU};
  static std::vector<VoxelInfo> const init_vec(config_->ky_width,
                                               invalid_voxel_info);
  for (auto &&voxel_infos : coord_to_voxel_id_) {
    voxel_infos.resize(config_->ky_width, invalid_voxel_info);
    memcpy(voxel_infos.data(), init_vec.data(),
           init_vec.size() * sizeof(VoxelInfo));
  }

  uint16_t max_voxel_idx{0U};
  uint16_t max_voxel_idy{0U};

  FeatureInfo *p_feature{
      reinterpret_cast<FeatureInfo *>(pillar_id_mem_.virAddr)};
  int8_t *voxel_data{reinterpret_cast<int8_t *>(voxel_mem_.virAddr)};
  uint32_t const k_hc{static_cast<uint32_t>(config_->align_padding_point) *
                      static_cast<uint32_t>(config_->kdim)};
  for (int32_t i{0}; i < num_points; ++i) {
    uint16_t idx{p_feature[i].idx};
    uint16_t idy{p_feature[i].idy};

    if (idx == 0xFFFFU || idy == 0xFFFFU) {
      continue;
    }

    VoxelInfo *pvoxel_info{&(coord_to_voxel_id_[idx][idy])};
    uint32_t voxel_idx{pvoxel_info->index};
    uint32_t voxel_num{pvoxel_info->num};

    if (voxel_idx == 0xFFFFU) {
      if (voxel_num_ < config_->kmax_num_point) {
        voxel_idx = voxel_num_;
        ++voxel_num_;
        coord_to_voxel_id_[idx][idy].index = voxel_idx;
        max_voxel_idx = idx;
        max_voxel_idy = idy;
      } else {
        voxel_idx = config_->kmax_num_point - 1;
        pvoxel_info = &(coord_to_voxel_id_[max_voxel_idx][max_voxel_idy]);
        voxel_num = pvoxel_info->num;
      }
      coords[voxel_idx * 4] = 0;
      coords[voxel_idx * 4 + 1] = 0;
      coords[voxel_idx * 4 + 2] = idy;
      coords[voxel_idx * 4 + 3] = idx;
    }

    if (voxel_num >= config_->kmax_num_point_pillar) {
      continue;
    }
    pvoxel_info->num = voxel_num + 1U;

    uint32_t const total_offset{
        voxel_idx * static_cast<uint32_t>(config_->kdim) + voxel_num * k_hc};
    int32_t *voxel_xyzr{reinterpret_cast<int32_t *>(voxel_data + total_offset)};
    *voxel_xyzr = p_feature[i].xyzr;
    voxel_data[total_offset + 4] = static_cast<int8_t>(p_feature[i].t);
  }

  int32_t const invalid_coords{config_->kmax_num_point - voxel_num_};
  if (invalid_coords > 0) {
    int32_t *invalid_coors_start{coords + voxel_num_ * 4};
    uint32_t index{0U};
    for (uint32_t i{0U}; i < invalid_coords; ++i) {
      invalid_coors_start[index] = 0;
      invalid_coors_start[index + 1] = -1;
      invalid_coors_start[index + 2] = -1;
      invalid_coors_start[index + 3] = -1;
      index += 4;
    }
  }
}

int32_t QATCenterPointPreProcessMethod::ProcessInDsp(ImageTensor *image_tensor,
                                                     int32_t num_points,
                                                     float scale) {
#ifdef HAVE_DSP
  // dsp stage 1, caculate pillar index
  DSPPointPillarPreProcessParam param;
  param.dim = config_->kdim;
  param.back = config_->kback_border;
  param.front = config_->kfront_border;
  param.left = config_->kleft_border;
  param.right = config_->kright_border;
  param.bottom = config_->kbottom_border;
  param.top = config_->ktop_border;
  param.r_lower = config_->kr_lower;
  param.r_upper = config_->kr_upper;
  param.x_scale = config_->kx_scale;
  param.y_scale = config_->ky_scale;
  param.max_num_points = config_->kmax_num_point;
  param.max_num_points_align = config_->align_padding_point;
  param.num_points = num_points;
  param.max_num_point_pillar = config_->kmax_num_point_pillar;
  param.max_num_point_pillars_align = param.max_num_point_pillar;
  param.scale = scale;

  auto &tensors = image_tensor->tensors;

  param.input_phy_addr = point_cloud_data_mem_.phyAddr;
  param.feature_phy_addr = pillar_id_mem_.phyAddr;
  param.voxel_phy_addr = voxel_mem_.phyAddr;
  // calculate pillar id
  param.stage = 0U;

  hbUCPTaskHandle_t task{nullptr};
  int32_t rpc_cmd = 0x1403;

  int32_t ret{CallDspRpc(task, param, rpc_cmd)};
  if (ret != 0) {
    return ret;
  }
  hbUCPMemFlush(&pillar_id_mem_, HB_SYS_MEM_CACHE_INVALIDATE);

  // generate coords and voxel data through pillar id
  int32_t *coords{reinterpret_cast<int32_t *>(tensors[1].sysMem.virAddr)};
  GenVoxelForDsp(coords, num_points);
  hbUCPMemFlush(&tensors[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);

  // dsp stage 2, generate features
  hbDSPAddrMap(&(tensors[0].sysMem), &(tensors[0].sysMem));

  param.input_phy_addr = voxel_mem_.phyAddr;
  param.feature_phy_addr = tensors[0].sysMem.phyAddr;

  param.voxel_num = voxel_num_;
  // generate features through voxel data
  param.stage = 1U;
  param.scale = scale;
  param.max_num_points =
      tensors[0].properties.stride[2] / tensors[0].properties.stride[3];

  task = nullptr;
  ret = CallDspRpc(task, param, rpc_cmd);
  if (ret != 0) {
    return ret;
  }
  hbUCPMemFlush(&tensors[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
  hbDSPAddrUnmap(&(tensors[0].sysMem));
  return 0;
#else
  VLOG(EXAMPLE_SYSTEM) << "DSP not available in this platform";
  return -1;
#endif
}

int32_t QATCenterPointPreProcessMethod::DoProcess(std::string path,
                                                  int32_t input_count,
                                                  ImageTensor *image_tensor) {
  Reset();
  float32_t *point_cloud_data =
      config_->krun_on_dsp
          ? reinterpret_cast<float32_t *>(point_cloud_data_mem_.virAddr)
          : point_cloud_data_;
  int32_t num_points = 0;
  int32_t ret =
      FetchBinary(point_cloud_data, num_points, path, 300000 * config_->kdim);
  if (ret != 0 || num_points <= 0) {
    VLOG(EXAMPLE_SYSTEM) << "Read file error. " << path;
    return -1;
  }

  if (config_->krun_on_dsp) {
    hbUCPMemFlush(&point_cloud_data_mem_, HB_SYS_MEM_CACHE_CLEAN);
  }

  VLOG(EXAMPLE_DEBUG) << "Load from:" << image_tensor->image_name
                      << "; points count: " << num_points;

  auto start = Stopwatch::CurrentTs();
  auto &tensors = image_tensor->tensors;

  tensors.resize(input_count);

  for (int i = 0; i < input_count; i++) {
    hbDNNGetInputTensorProperties(&(tensors[i].properties), dnn_handle_, i);
    int aligned_size = tensors[i].properties.alignedByteSize;
    hbUCPMallocCached(&(tensors[i].sysMem), aligned_size, 0);
  }

  float32_t scale_data = tensors[0].properties.scale.scaleData[0];
  if (config_->krun_on_dsp) {
    ret = ProcessInDsp(image_tensor, num_points, scale_data);
    if (ret != 0) {
      VLOG(EXAMPLE_REPORT) << "run preprocess in dsp failed!";
      return -1;
    }
  } else {
    // gen voxel
    int32_t *coors = reinterpret_cast<int32_t *>(tensors[1].sysMem.virAddr);
    GenVoxel(0, num_points);
    memcpy(coors, config_->coors, tensors[1].sysMem.memSize);

    // gen feature
    GenFeatureDim5(scale_data);

    // transpose
    int8_t *features = reinterpret_cast<int8_t *>(tensors[0].sysMem.virAddr);
    TransposeDim5(tensors[0].properties.stride[2] /
                  tensors[0].properties.stride[3]);

    memcpy(features, features_s8_, tensors[0].sysMem.memSize);

    for (int32_t i = 0; i < input_count; i++) {
      hbUCPMemFlush(&(tensors[i].sysMem), HB_SYS_MEM_CACHE_CLEAN);
    }
  }
  auto end = Stopwatch::CurrentTs();
  image_tensor->pre_duration = end - start;
  return 0;
}

QATCenterPointPreProcessMethod::~QATCenterPointPreProcessMethod() {
  if (!config_->krun_on_dsp) {
    free(voxel_data_);
    free(point_cloud_data_);
    free(features_s8_);
  } else {
#if HAVE_DSP
    hbDSPAddrUnmap(&point_cloud_data_mem_);
    hbDSPAddrUnmap(&voxel_mem_);
    hbDSPAddrUnmap(&pillar_id_mem_);
    hbDSPAddrUnmap(&task_spec_mem_);
#endif
    if (point_cloud_data_mem_.virAddr != nullptr) {
      hbUCPFree(&point_cloud_data_mem_);
    }

    if (voxel_mem_.virAddr != nullptr) {
      hbUCPFree(&voxel_mem_);
    }

    if (pillar_id_mem_.virAddr != nullptr) {
      hbUCPFree(&pillar_id_mem_);
    }

    if (task_spec_mem_.virAddr != nullptr) {
      hbUCPFree(&task_spec_mem_);
    }
  }

  delete config_;
  config_ = nullptr;
}
