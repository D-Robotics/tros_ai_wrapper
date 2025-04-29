// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_PREPROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_PREPROCESS_METHOD_H_

#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/preprocess_method.h"

#ifdef HAVE_DSP
#include "hobot/plugin/dsp_plugin/hb_dsp.h"
#include "hobot/plugin/dsp_plugin/hb_dsp_addr_map.h"
#endif

struct VoxelConfig {
  const float kback_border;
  const float kfront_border;
  const float kright_border;
  const float kleft_border;
  const float kbottom_border;
  const float ktop_border;
  const float kr_lower;
  const float kr_upper;
  const float kx_range;
  const float ky_range;
  const float kz_range;
  const float kr_range;
  const float kx_scale;
  const float ky_scale;
  const int kx_length;
  const int ky_width;
  const int kmax_num_point_pillar;
  const int kmax_num_point;
  const int kdim;
  const int align_padding_point;
  bool krun_on_dsp;

  std::vector<int> kmax_num_point_pillar_vec = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  int *pillar_point_num;
  int *coor_to_voxel_idx;
  int *coors;

  VoxelConfig(float back, float front, float right, float left, float bottom,
              float top, float r_lower, float r_upper, float x_scale,
              float y_scale, int max_num_point_pillar, int max_num_point,
              int dim, bool run_on_dsp, int align_padding_point)
      : kback_border(back),
        kfront_border(front),
        kright_border(right),
        kleft_border(left),
        kbottom_border(bottom),
        ktop_border(top),
        kr_lower(r_lower),
        kr_upper(r_upper),
        kx_scale(x_scale),
        ky_scale(y_scale),
        kmax_num_point_pillar(max_num_point_pillar),
        kmax_num_point(max_num_point),
        kdim(dim),
        kx_length((kfront_border - kback_border) / kx_scale),
        ky_width((kleft_border - kright_border) / ky_scale),
        kx_range(kfront_border - kback_border),
        ky_range(kleft_border - kright_border),
        kz_range(ktop_border - kbottom_border),
        kr_range(kr_upper - kr_lower),
        align_padding_point(align_padding_point),
        krun_on_dsp(run_on_dsp) {
    if (!run_on_dsp) {
      pillar_point_num = new int[this->kmax_num_point];
      coor_to_voxel_idx = new int[kx_length * ky_width];
      coors = new int[this->kmax_num_point * 4];
    }
  }

  ~VoxelConfig() {
    if (!krun_on_dsp) {
      delete[] pillar_point_num;
      delete[] coor_to_voxel_idx;
      delete[] coors;
    }
  }
};

struct VoxelInfo {
  uint16_t num;
  uint16_t index;
};

struct DSPPointPillarPreProcessParam {
  float scale;
  float back;
  float front;
  float right;
  float left;
  float bottom;
  float top;
  float r_lower;
  float r_upper;
  float x_scale;
  float y_scale;
  uint32_t max_num_points;
  uint32_t max_num_points_align;
  uint32_t max_num_point_pillar;
  uint32_t max_num_point_pillars_align;
  uint32_t num_points;
  uint64_t input_phy_addr;
  uint64_t feature_phy_addr;
  uint64_t voxel_phy_addr;
  uint8_t dim;
  uint8_t stage;
  int32_t voxel_num;
};

/**
 * Method for pre processing
 */
class QATCenterPointPreProcessMethod : public PreProcessMethod {
 public:
  int32_t InitFromJsonString(const std::string &config) override;

  int32_t DoProcess(std::string path, int32_t input_count,
                    ImageTensor *image_tensor) override;

  ~QATCenterPointPreProcessMethod();

 private:
  int32_t FetchBinary(float *buffer, int32_t &num_points,
                      std::string const &point_cloud_path, int32_t length);

  void GenVoxel(int start, int end);
  void GenFeatureDim5(float scale);
  void TransposeDim5(int model_aligned_w);
  void Reset();

#ifdef HAVE_DSP
  int32_t CallDspRpc(hbUCPTaskHandle_t &task,
                     DSPPointPillarPreProcessParam &param, int32_t rpc_cmd);
#endif

  void GenVoxelForDsp(int32_t *coords, int32_t num_points);

  int32_t ProcessInDsp(ImageTensor *image_tensor, int32_t num_points,
                       float scale);

 private:
  VoxelConfig *config_;

  float *point_cloud_data_;
  float *voxel_data_;
  int8_t *features_s8_;

  int voxel_num_;
  hbUCPSysMem point_cloud_data_mem_{};
  hbUCPSysMem voxel_mem_{};
  hbUCPSysMem pillar_id_mem_{};
  hbUCPSysMem task_spec_mem_{};
  std::vector<std::vector<VoxelInfo>> coord_to_voxel_id_;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_CENTERPOINT_PREPROCESS_METHOD_H_
