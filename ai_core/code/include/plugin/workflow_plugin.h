// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_WORKFLOW_PLUGIN_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_WORKFLOW_PLUGIN_H_

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base_plugin.h"
#include "method/infer_method.h"
#include "method/post_process_method.h"
#include "utils/pc_queue.h"

class WorkflowPlugin : public BasePlugin {
 public:
  static WorkflowPlugin* GetInstance() {
    static WorkflowPlugin instance;
    return &instance;
  }

  int Init(const std::string& config_file,
           const std::string& config_json_string);

  int Start() override;

  int FeedWorkflow(ImageTensorPtr msg);

  int GetQueueSize();

  void Run(int instance_id);

  hbDNNHandle_t GetModelHandle() { return dnn_handle_; }

  int GetInputFeatureProperties(hbDNNTensorProperties* properties,
                                int input_idx);

  int Stop() override;

  int SetFirstUpdatedStatus(int num);

  bool GetFirstUpdatedStatus(int num);

  template <typename T>
  int UpdateMajorModelTensor(hbDNNTensor* tensor, int num) {
    hbUCPMemFlush(&(reletated_tensor_[num].sysMem),
                  HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(tensor->sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    memcpy(reinterpret_cast<T>(reletated_tensor_[num].sysMem.virAddr),
           reinterpret_cast<T>(tensor->sysMem.virAddr),
           reletated_tensor_[num].properties.alignedByteSize);
    WorkflowPlugin::SetFirstUpdatedStatus(num);
    return 0;
  }

  int GetMajorModelTensor(std::vector<hbDNNTensor>& tensor);

  int ReleaseMajorModelTensor();

  int IsTemporalModel() { return is_temporal_model_; }

  int Is_temporal_multiframe_model() { return is_temporal_multiframe_model_; }
  void SetTemporalUpdateStatus(int idx);

  bool GetTemporalUpdateStatus(int idx);
  void SetTemporalMultiFrameUpdateStatus(bool flag);
  bool GetTemporalMultiFrameUpdateStatus();
  int GetTemporalModelTensor(std::vector<hbDNNTensor>& tensor, int frame_id);
  void GetTemporalAgentEnc(std::vector<hbDNNTensor>& tensor, int frame_id,
                           int frames_per_sample_infer);
  void GetTemporalAgentEmb(std::vector<hbDNNTensor>& tensor, int frame_id,
                           int frames_per_sample_infer);
  void exchange(hbDNNTensor* src_tensor, hbDNNTensor* dest_tensor, int src_h,
                int dest_h);
  int GetInferFramNums() { return infer_fram_nums_; }

  int ReleaseTemporalModelTensor();

  int GetAssociatedPrevNum() { return associated_prev_num_; }

  // AgentEmb (1 30 1 128) --> temporal_tensor_[0][1]
  template <typename T>
  int UpdateAgentEmb(hbDNNTensor* tensor, int prev_num, int idx) {
    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(tensor->sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    memcpy(reinterpret_cast<T*>(temporal_tensor_[idx][prev_num].sysMem.virAddr),
           reinterpret_cast<T*>(tensor->sysMem.virAddr),
           temporal_tensor_[idx][prev_num].properties.alignedByteSize);
    if (memcmp(reinterpret_cast<T*>(
                   temporal_tensor_[idx][prev_num].sysMem.virAddr),
               reinterpret_cast<T*>(tensor->sysMem.virAddr),
               temporal_tensor_[idx][prev_num].properties.alignedByteSize) !=
        0) {
      VLOG(EXAMPLE_DETAIL) << "Memory copy failed: data mismatch.";
    }
    WorkflowPlugin::SetTemporalUpdateStatus(idx);

    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_CLEAN);
    return 0;
  }
  // AgentEnc (1 30 1 128) --> temporal_tensor_[1][k]
  void UpdateAgentEnc(int16_t* flat_agent_enc, int prev_num, int idx,
                      int copy_size) {
    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_INVALIDATE);
    int32_t temporal_alignsize =
        temporal_tensor_[idx][prev_num].properties.alignedByteSize;
    memcpy(reinterpret_cast<int16_t*>(
               temporal_tensor_[idx][prev_num].sysMem.virAddr),
           reinterpret_cast<int16_t*>(flat_agent_enc),
           copy_size * sizeof(int16_t));
    if (memcmp(reinterpret_cast<int16_t*>(
                   temporal_tensor_[idx][prev_num].sysMem.virAddr),
               reinterpret_cast<int16_t*>(flat_agent_enc),
               copy_size * sizeof(int16_t)) != 0) {
      VLOG(EXAMPLE_DETAIL) << " Error: Agent Embedding copy fail ! ";
    }
    WorkflowPlugin::SetTemporalUpdateStatus(idx);
    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_CLEAN);
  }
  template <typename T>
  int UpdateTemporalTensor(hbDNNTensor* tensor, int prev_num, int idx) {
    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(tensor->sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    memcpy(reinterpret_cast<T>(temporal_tensor_[idx][prev_num].sysMem.virAddr),
           reinterpret_cast<T>(tensor->sysMem.virAddr),
           temporal_tensor_[idx][prev_num].properties.alignedByteSize);
    WorkflowPlugin::SetTemporalUpdateStatus(idx);

    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_CLEAN);

    return 0;
  }

  template <typename T>
  int UpdateTemporalQuantTensor(hbUCPSysMem* tensor_data, int prev_num,
                                int idx) {
    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(tensor_data, HB_SYS_MEM_CACHE_INVALIDATE);
    int copy_size = temporal_tensor_[idx][prev_num].properties.alignedByteSize;
    if (temporal_tensor_[idx][prev_num].properties.tensorType ==
        HB_DNN_TENSOR_TYPE_S16) {
      copy_size /= sizeof(int16_t);
    } else if (temporal_tensor_[idx][prev_num].properties.tensorType ==
               HB_DNN_TENSOR_TYPE_S32) {
      copy_size /= sizeof(int32_t);
    }
    memcpy(reinterpret_cast<T>(temporal_tensor_[idx][prev_num].sysMem.virAddr),
           reinterpret_cast<T>(tensor_data->virAddr),
           copy_size * sizeof(float));
    WorkflowPlugin::SetTemporalUpdateStatus(idx);

    hbUCPMemFlush(&(temporal_tensor_[idx][prev_num].sysMem),
                  HB_SYS_MEM_CACHE_CLEAN);

    return 0;
  }

  ~WorkflowPlugin() override;

 private:
  WorkflowPlugin() = default;

  bool stop_ = false;
  std::vector<std::shared_ptr<std::thread>> threads_;

  std::vector<std::pair<std::shared_ptr<InferMethod>,
                        std::shared_ptr<PostProcessMethod>>>
      instances;

  PCQueue<ImageTensorPtr> pc_queue_;

  hbDNNHandle_t dnn_handle_{nullptr};
  std::vector<hbDNNTensorProperties> input_tensor_properties_;

  bool is_releated_model_{false};
  int32_t need_update_tensor_num_{0};
  // update major tensor
  std::vector<hbDNNTensor> reletated_tensor_;
  // Determines whether the first model in the next frame needs to be updated.
  std::vector<bool> is_reletated_first_model_update_{false};

  bool is_temporal_model_{false};
  bool is_temporal_multiframe_model_{false};
  bool is_temporal_multiframe_update_{true};
  int32_t steps_to_decode_{5};
  int32_t infer_fram_nums_{8};
  int32_t associated_prev_num_{1};
  std::vector<int32_t> temporal_input_tensors_idx_;
  std::vector<int32_t> temporal_output_tensors_idx_;
  std::vector<std::vector<hbDNNTensor>> temporal_tensor_;
  std::vector<bool> is_temporal_update_{false, false};
  std::vector<int> input_RelPos_idx{3,  4,  5,  6,  7,  8,  9,  10, 11,
                                    12, 13, 14, 15, 16, 17, 18, 19};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_WORKFLOW_PLUGIN_H_
