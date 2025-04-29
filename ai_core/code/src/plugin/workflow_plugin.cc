// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "plugin/workflow_plugin.h"

#include <iostream>

#include "glog/logging.h"
#include "method/method_factory.h"
#include "plugin/output_plugin.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"

int WorkflowPlugin::Init(const std::string& config_file,
                         const std::string& config_json_string) {
  rapidjson::Document document;
  document.Parse(config_json_string.data());
  auto array = document["workflow"].GetArray();
  auto pp_type = array[1]["method_type"].GetString();
  auto thread_count = array[1]["thread_count"].GetInt();

  std::string method_config_tmp = json_to_string(array[0]["method_config"]);
  rapidjson::Document document_method_config;
  document_method_config.Parse(method_config_tmp.data());
  if (document_method_config.HasMember("is_releated_model")) {
    is_releated_model_ = document_method_config["is_releated_model"].GetBool();
  }
  if (document_method_config.HasMember("need_update_tensor_num")) {
    need_update_tensor_num_ =
        document_method_config["need_update_tensor_num"].GetInt();
  }
  if (document_method_config.HasMember("is_temporal_model")) {
    is_temporal_model_ = document_method_config["is_temporal_model"].GetBool();
    if (document_method_config.HasMember("associated_prev_num")) {
      associated_prev_num_ =
          document_method_config["associated_prev_num"].GetInt();
    }
  }
  // Attention! The current temporal multi-frame inference model only supports QCNet.
  if (document_method_config.HasMember("is_temporal_multiframe_model")) {
    is_temporal_multiframe_model_ =
        document_method_config["is_temporal_multiframe_model"].GetBool();
    if (document_method_config.HasMember("associated_prev_num")) {
      associated_prev_num_ =
          document_method_config["associated_prev_num"].GetInt();
    }
    // Number of frames to encode required for decoding in the trajectory prediction model QCNet
    if (document_method_config.HasMember("steps_to_decode")) {
      steps_to_decode_ = document_method_config["steps_to_decode"].GetInt();
    }
    if (document_method_config.HasMember("infer_fram_nums")) {
      infer_fram_nums_ = document_method_config["infer_fram_nums"].GetInt();
    }
  }
  if (document_method_config.HasMember("temporal_input_tensors_idx")) {
    rapidjson::Value& temporal =
        document_method_config["temporal_input_tensors_idx"];
    auto temporal_array = temporal.GetArray();
    for (int i = 0; i < temporal_array.Size(); ++i) {
      temporal_input_tensors_idx_.push_back(temporal_array[i].GetInt());
    }
  }

  if (document_method_config.HasMember("temporal_output_tensors_idx")) {
    rapidjson::Value& temporal =
        document_method_config["temporal_output_tensors_idx"];
    auto temporal_array = temporal.GetArray();
    for (int i = 0; i < temporal_array.Size(); ++i) {
      temporal_output_tensors_idx_.push_back(temporal_array[i].GetInt());
    }
  }

  if (temporal_output_tensors_idx_.size() !=
      temporal_input_tensors_idx_.size()) {
    VLOG(EXAMPLE_SYSTEM) << "temporal tensors idx error!";
  }

  std::shared_ptr<InferMethod> infer_method =
      std::shared_ptr<InferMethod>(dynamic_cast<InferMethod*>(
          MethodFactory::GetInstance()->GetMethod("InferMethod")));
  infer_method->InitFromJsonString(json_to_string(array[0]["method_config"]));
  dnn_handle_ = infer_method->GetModelHandle();
  for (int i = 0; i < thread_count; i++) {
    std::shared_ptr<PostProcessMethod> pp_method =
        std::shared_ptr<PostProcessMethod>(dynamic_cast<PostProcessMethod*>(
            MethodFactory::GetInstance()->GetMethod(pp_type)));
    pp_method->InitFromJsonString(json_to_string(array[1]["method_config"]));
    instances.emplace_back(infer_method, pp_method);
  }

  int32_t input_count{0};
  hbDNNGetInputCount(&input_count, dnn_handle_);
  input_tensor_properties_.resize(input_count);
  for (int i = 0; i < input_count; i++) {
    hbDNNGetInputTensorProperties(&input_tensor_properties_[i], dnn_handle_, i);
  }

  if (is_releated_model_) {
    reletated_tensor_.resize(need_update_tensor_num_);
    hbDNNHandle_t dnn_handle_major =
        WorkflowPlugin::GetInstance()->GetModelHandle();
    for (int i = 0; i < need_update_tensor_num_; i++) {
      hbDNNGetInputTensorProperties(&reletated_tensor_[i].properties,
                                    dnn_handle_major, i + 2);
      int aligned_size = reletated_tensor_[i].properties.alignedByteSize;
      hbUCPMallocCached(&reletated_tensor_[i].sysMem, aligned_size, 0);
    }
    is_reletated_first_model_update_.resize(need_update_tensor_num_, false);
  }
  if (is_temporal_model_) {
    VLOG(EXAMPLE_DEBUG) << "the temporal tensor count is: "
                        << temporal_output_tensors_idx_.size();
    VLOG(EXAMPLE_DEBUG) << "the temporal model is related to the previous "
                        << associated_prev_num_ << " frames.";
    temporal_tensor_.resize(temporal_output_tensors_idx_.size());
    hbDNNHandle_t temporal_dnn_handle =
        WorkflowPlugin::GetInstance()->GetModelHandle();
    for (int i = 0; i < temporal_tensor_.size(); i++) {
      temporal_tensor_[i].resize(associated_prev_num_);
      for (int k = 0; k < temporal_tensor_[i].size(); k++) {
        hbDNNGetOutputTensorProperties(&temporal_tensor_[i][k].properties,
                                       temporal_dnn_handle,
                                       temporal_output_tensors_idx_[i]);
        float* input_scaleData =
            input_tensor_properties_[temporal_input_tensors_idx_[i]]
                .scale.scaleData;
        float* output_scaleData =
            temporal_tensor_[i][k].properties.scale.scaleData;
        bool need_dequant =
            std::fabs(*input_scaleData - *output_scaleData) > 1e-6;

        int aligned_size = aligned_size =
            temporal_tensor_[i][k].properties.alignedByteSize;
        if (need_dequant) {
          if (temporal_tensor_[i][k].properties.tensorType ==
              HB_DNN_TENSOR_TYPE_S16) {
            aligned_size /= sizeof(int16_t);
          } else if (temporal_tensor_[i][k].properties.tensorType ==
                     HB_DNN_TENSOR_TYPE_S32) {
            aligned_size /= sizeof(int32_t);
          }
          aligned_size =
              temporal_tensor_[i][k].properties.alignedByteSize * sizeof(float);
        }

        hbUCPMallocCached(&temporal_tensor_[i][k].sysMem, aligned_size, 0);
      }
    }
  }
  if (is_temporal_multiframe_model_) {
    VLOG(EXAMPLE_DETAIL) << "Attention! The current temporal multi-frame "
                            "inference model only supports QCNet.";
    is_temporal_update_.resize(temporal_output_tensors_idx_.size());
    for (int i = 0; i < temporal_output_tensors_idx_.size(); i++) {
      is_temporal_update_[i] = false;
    }
    // Initialize variables to be updated in sequence: output3 -> input20, output2 -> input21.
    temporal_tensor_.resize(temporal_output_tensors_idx_.size());
    hbDNNHandle_t temporal_dnn_handle =
        WorkflowPlugin::GetInstance()->GetModelHandle();
    temporal_tensor_[0].resize(associated_prev_num_);
    for (int k = 0; k < temporal_tensor_[0].size(); k++) {
      hbDNNGetInputTensorProperties(&temporal_tensor_[0][k].properties,
                                    temporal_dnn_handle,
                                    temporal_input_tensors_idx_[0]);
      float* input_scaleData =
          input_tensor_properties_[temporal_input_tensors_idx_[0]]
              .scale.scaleData;
      float* output_scaleData =
          temporal_tensor_[0][k].properties.scale.scaleData;
      bool need_dequant =
          std::fabs(*input_scaleData - *output_scaleData) > 1e-6;
      int aligned_size = aligned_size =
          temporal_tensor_[0][k].properties.alignedByteSize;
      HB_CHECK_SUCCESS(
          hbUCPMallocCached(&temporal_tensor_[0][k].sysMem, aligned_size, 0),
          "hbUCPMallocCached failed");
      auto& mem = temporal_tensor_[0][k].sysMem;
      memset(mem.virAddr, 0, aligned_size);
    }
    // init agent encoder --> temporal_tensor_[1][0~5]
    temporal_tensor_[1].resize(steps_to_decode_);
    for (int k = 0; k < steps_to_decode_; k++) {
      hbDNNGetInputTensorProperties(&temporal_tensor_[1][k].properties,
                                    temporal_dnn_handle,
                                    temporal_input_tensors_idx_[1]);
      int32_t aligned_size = temporal_tensor_[1][k].properties.alignedByteSize;

      HB_CHECK_SUCCESS(
          hbUCPMallocCached(&temporal_tensor_[1][k].sysMem, aligned_size, 0),
          "hbUCPMallocCached failed");
      auto& mem = temporal_tensor_[1][k].sysMem;
      memset(mem.virAddr, 0, aligned_size);
    }
  }
  VLOG(EXAMPLE_DETAIL) << "WorkflowPlugin inited.";
  return 0;
}

int WorkflowPlugin::Start() {
  VLOG(EXAMPLE_DETAIL) << "WorkflowPlugin start.";
  stop_ = false;
  for (int i = 0; i < instances.size(); i++) {
    auto t = std::make_shared<std::thread>(&WorkflowPlugin::Run, this, i);
    threads_.push_back(t);
  }
  return 0;
}

int WorkflowPlugin::FeedWorkflow(ImageTensorPtr image_tensor) {
  pc_queue_.put(image_tensor);
  return 0;
}

void WorkflowPlugin::Run(int instance_id) {
  auto infer_method = instances[instance_id].first;
  auto pp_method = instances[instance_id].second;
  while (!stop_) {
    ImageTensorPtr image_tensor;
    if (!pc_queue_.get(image_tensor, 1000)) {
      if (stop_) {
        break;
      }
      continue;
    }
    auto start = Stopwatch::CurrentTs();
    auto output_tensors = infer_method->DoProcess(image_tensor.get());
    auto end = Stopwatch::CurrentTs();
    auto infer_duration = end - start;
    auto perception = pp_method->DoProcess(image_tensor.get(), output_tensors);
    auto pp_duration = Stopwatch::CurrentTs() - end;
    perception->infer_duration = infer_duration;
    perception->pp_duration = pp_duration;
    perception->pre_duration = image_tensor->pre_duration;
    OutputConsumerPlugin::GetInstance()->Send(
        std::make_pair(image_tensor, perception));
  }
}

int WorkflowPlugin::Stop() {
  stop_ = true;
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i]->join();
  }

  auto infer_method = instances[0].first;
  infer_method->Rlease();

  VLOG(EXAMPLE_DETAIL) << "WorkflowPlugin stop.";
  return 0;
}

int WorkflowPlugin::SetFirstUpdatedStatus(int num) {
  is_reletated_first_model_update_[num] = true;
  return 0;
}

bool WorkflowPlugin::GetFirstUpdatedStatus(int num) {
  return is_reletated_first_model_update_[num];
}

int WorkflowPlugin::GetMajorModelTensor(std::vector<hbDNNTensor>& tensor) {
  for (int i = 0; i < need_update_tensor_num_; i++) {
    if (GetFirstUpdatedStatus(i)) {
      hbUCPMemFlush(&(tensor[i + 2].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
      hbUCPMemFlush(&(reletated_tensor_[i].sysMem),
                    HB_SYS_MEM_CACHE_INVALIDATE);
      int tensor_type = reletated_tensor_[i].properties.tensorType;
      switch (tensor_type) {
        case HB_DNN_TENSOR_TYPE_S8:
          memcpy(reinterpret_cast<int8_t*>(tensor[i + 2].sysMem.virAddr),
                 reinterpret_cast<int8_t*>(reletated_tensor_[i].sysMem.virAddr),
                 tensor[i + 2].properties.alignedByteSize);
          break;
        case HB_DNN_TENSOR_TYPE_S16:
          memcpy(
              reinterpret_cast<int16_t*>(tensor[i + 2].sysMem.virAddr),
              reinterpret_cast<int16_t*>(reletated_tensor_[i].sysMem.virAddr),
              tensor[i + 2].properties.alignedByteSize);
          break;
        case HB_DNN_TENSOR_TYPE_S32:
          memcpy(
              reinterpret_cast<int32_t*>(tensor[i + 2].sysMem.virAddr),
              reinterpret_cast<int32_t*>(reletated_tensor_[i].sysMem.virAddr),
              tensor[i + 2].properties.alignedByteSize);
          break;
        default:
          VLOG(EXAMPLE_SYSTEM)
              << "GetMajorModelTensor is not support for data type:"
              << tensor_type;
          break;
      }
      // update releated_tensor status,
      is_reletated_first_model_update_[i] = false;

      flush_tensor(&tensor[i + 2]);
    }
  }

  return 0;
}

int WorkflowPlugin::ReleaseMajorModelTensor() {
  for (int i = 0; i < need_update_tensor_num_; i++) {
    release_tensor(&reletated_tensor_[i]);
    is_reletated_first_model_update_[i] = false;
  }
  return 0;
}

int WorkflowPlugin::GetTemporalModelTensor(std::vector<hbDNNTensor>& tensor,
                                           int frame_id) {
  for (int i = 0; i < temporal_input_tensors_idx_.size(); i++) {
    float* input_scaleData =
        input_tensor_properties_[temporal_input_tensors_idx_[i]]
            .scale.scaleData;
    float* output_scaleData = temporal_tensor_[i][0].properties.scale.scaleData;
    bool need_dequant = std::fabs(*input_scaleData - *output_scaleData) > 1e-6;

    VLOG(EXAMPLE_DEBUG) << "temporal tensor " << i << ", update status is "
                        << GetTemporalUpdateStatus(i);
    if (GetTemporalUpdateStatus(i)) {
      int tensor_idx = temporal_input_tensors_idx_[i];
      VLOG(EXAMPLE_DEBUG) << "update tensor index is " << tensor_idx
                          << ", associated previous " << associated_prev_num_
                          << " frame.";
      if (associated_prev_num_ == 2) {
        VLOG(EXAMPLE_DEBUG) << "Frame id " << frame_id << " is reversed.";
        std::reverse(temporal_tensor_[i].begin(), temporal_tensor_[i].end());
      }

      hbUCPMemFlush(&(tensor[tensor_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
      for (int k = 0; k < associated_prev_num_; k++) {
        hbUCPMemFlush(&(temporal_tensor_[i][k].sysMem),
                      HB_SYS_MEM_CACHE_INVALIDATE);
      }
      int tensor_type = temporal_tensor_[0][i].properties.tensorType;
      int align_size = temporal_tensor_[0][i].properties.alignedByteSize;
      if (align_size * associated_prev_num_ !=
          tensor[tensor_idx].properties.alignedByteSize) {
        VLOG(EXAMPLE_SYSTEM)
            << "aligned byte size error! temporal tensor aligned byte size is "
            << align_size << " but tensor aligned byte size is "
            << tensor[tensor_idx].properties.alignedByteSize;
        return -1;
      }

      std::vector<float> source_data;
      switch (tensor_type) {
        case HB_DNN_TENSOR_TYPE_S8:
          for (int k = 0; k < associated_prev_num_; k++) {
            if (!need_dequant) {
              memcpy(
                  reinterpret_cast<int8_t*>(tensor[tensor_idx].sysMem.virAddr) +
                      k * align_size,
                  reinterpret_cast<int8_t*>(
                      temporal_tensor_[i][k].sysMem.virAddr),
                  align_size);
            } else {
              float* data = reinterpret_cast<float*>(
                  temporal_tensor_[i][k].sysMem.virAddr);
              source_data.assign(data, data + align_size);
              void* mem =
                  reinterpret_cast<float*>(tensor[tensor_idx].sysMem.virAddr) +
                  k * align_size;
              auto align_shape =
                  properies2alignshape(tensor[tensor_idx].properties);
              quanti_tensor<int8_t>(
                  &mem, source_data,
                  tensor[tensor_idx].properties.validShape.dimensionSize,
                  align_shape.data(),
                  tensor[tensor_idx].properties.scale.scaleData, -128.0f,
                  127.0f,
                  tensor[tensor_idx].properties.validShape.numDimensions);
            }
          }
          break;
        case HB_DNN_TENSOR_TYPE_S16:
          for (int k = 0; k < associated_prev_num_; k++) {
            if (!need_dequant) {
              memcpy(reinterpret_cast<int16_t*>(
                         tensor[tensor_idx].sysMem.virAddr) +
                         k * align_size,
                     reinterpret_cast<int16_t*>(
                         temporal_tensor_[i][k].sysMem.virAddr),
                     align_size);
            } else {
              float* data = reinterpret_cast<float*>(
                  temporal_tensor_[i][k].sysMem.virAddr);
              source_data.assign(data, data + align_size);
              void* mem =
                  reinterpret_cast<float*>(tensor[tensor_idx].sysMem.virAddr) +
                  k * align_size;
              auto align_shape =
                  properies2alignshape(tensor[tensor_idx].properties);
              quanti_tensor<int16_t>(
                  &mem, source_data,
                  tensor[tensor_idx].properties.validShape.dimensionSize,
                  align_shape.data(),
                  tensor[tensor_idx].properties.scale.scaleData, -32768.0f,
                  32767.0f,
                  tensor[tensor_idx].properties.validShape.numDimensions);
            }
          }
          break;
        case HB_DNN_TENSOR_TYPE_S32:
          for (int k = 0; k < associated_prev_num_; k++) {
            if (!need_dequant) {
              memcpy(reinterpret_cast<int32_t*>(
                         tensor[tensor_idx].sysMem.virAddr) +
                         k * align_size,
                     reinterpret_cast<int32_t*>(
                         temporal_tensor_[i][k].sysMem.virAddr),
                     align_size);
            } else {
              float* data = reinterpret_cast<float*>(
                  temporal_tensor_[i][k].sysMem.virAddr);
              source_data.assign(data, data + align_size);
              void* mem =
                  reinterpret_cast<float*>(tensor[tensor_idx].sysMem.virAddr) +
                  k * align_size;
              auto align_shape =
                  properies2alignshape(tensor[tensor_idx].properties);
              quanti_tensor<int32_t>(
                  &mem, source_data,
                  tensor[tensor_idx].properties.validShape.dimensionSize,
                  align_shape.data(),
                  tensor[tensor_idx].properties.scale.scaleData, -2147483648.0f,
                  2147483647.0f,
                  tensor[tensor_idx].properties.validShape.numDimensions);
            }
          }
          break;
        default:
          VLOG(EXAMPLE_SYSTEM)
              << "GetTemporalModelTensor is not support for data type:"
              << tensor_type;
          break;
      }
      is_temporal_update_[i] = false;

      flush_tensor(&tensor[tensor_idx]);
    }
  }

  return 0;
}

void WorkflowPlugin::SetTemporalUpdateStatus(int idx) {
  is_temporal_update_[idx] = true;
}

bool WorkflowPlugin::GetTemporalUpdateStatus(int idx) {
  return is_temporal_update_[idx];
}

void WorkflowPlugin::SetTemporalMultiFrameUpdateStatus(bool flag) {
  is_temporal_multiframe_update_ = flag;
}
bool WorkflowPlugin::GetTemporalMultiFrameUpdateStatus() {
  return is_temporal_multiframe_update_;
}

int WorkflowPlugin::GetInputFeatureProperties(hbDNNTensorProperties* properties,
                                              int input_idx) {
  if ((input_idx >= input_tensor_properties_.size()) || (input_idx < 0)) {
    VLOG(EXAMPLE_SYSTEM) << "input_idx exceeds limit!";
    return -1;
  }
  *properties = input_tensor_properties_[input_idx];
  return 0;
}

int WorkflowPlugin::ReleaseTemporalModelTensor() {
  for (int i = 0; i < temporal_tensor_.size(); i++) {
    for (int k = 0; k < temporal_tensor_[i].size(); k++) {
      release_tensor(&temporal_tensor_[i][k]);
      is_temporal_update_[i] = false;
    }
  }
  return 0;
}

void WorkflowPlugin::GetTemporalAgentEmb(std::vector<hbDNNTensor>& tensor,
                                         int frame_id,
                                         int frames_per_sample_infer) {
  int tem_emb_idx = 0;  // agent embedding --> temporal_tensor_[0]

  if (frames_per_sample_infer == 0) {
    for (int k = 0; k < temporal_tensor_[tem_emb_idx].size(); k++) {
      auto& mem = temporal_tensor_[tem_emb_idx][k].sysMem;
      int aligned_size =
          temporal_tensor_[tem_emb_idx][k].properties.alignedByteSize;
      memset(mem.virAddr, 0, aligned_size);
      hbUCPMemFlush(&(temporal_tensor_[tem_emb_idx][k].sysMem),
                    HB_SYS_MEM_CACHE_CLEAN);
    }
  }

  if (GetTemporalUpdateStatus(tem_emb_idx)) {
    hbUCPMemFlush(&(tensor[20].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    for (int k = 0; k < associated_prev_num_; k++) {
      hbUCPMemFlush(&(temporal_tensor_[tem_emb_idx][k].sysMem),
                    HB_SYS_MEM_CACHE_INVALIDATE);
    }
    auto valid_shape = tensor[20].properties.validShape.dimensionSize;
    auto aligned_shape = properies2alignshape(tensor[20].properties);
    int tensor_type = temporal_tensor_[0][1].properties.tensorType;
    for (int k = 0; k < associated_prev_num_; k++) {
      // exchange(&tensor[tensor_idx], &temporal_tensor_[tem_emb_idx][k], k, 0);
      if (tensor_type == HB_DNN_TENSOR_TYPE_S16) {
        // HB_DNN_TENSOR_TYPE_S8
        int valid_n = valid_shape[0];
        int valid_c = valid_shape[1];
        int valid_h = valid_shape[2];
        int valid_w = valid_shape[3];
        int aligned_n = aligned_shape[0];  // 1
        int aligned_c = aligned_shape[1];  // 30
        int aligned_h = aligned_shape[2];  // 5
        int aligned_w = aligned_shape[3];  // 128
        auto copy_size = sizeof(int16_t);
        // 保持 n, c, h, w 结构
        int count = 0;
        for (int n = 0; n < valid_n; n++) {
          int dest_nchw = n * aligned_c * aligned_h * aligned_w;
          for (int c = 0; c < valid_c; c++) {
            int dest_chw = dest_nchw + c * aligned_h * aligned_w;
            int dest_hw = dest_chw + k * aligned_w;
            for (int w = 0; w < valid_w; w++) {
              int dest_w = dest_hw + w;
              memcpy(reinterpret_cast<int16_t*>(tensor[20].sysMem.virAddr) +
                         dest_w,
                     reinterpret_cast<int16_t*>(
                         temporal_tensor_[0][k].sysMem.virAddr) +
                         count,
                     copy_size);
              if (memcmp(reinterpret_cast<int16_t*>(tensor[20].sysMem.virAddr) +
                             dest_w,
                         reinterpret_cast<int16_t*>(
                             temporal_tensor_[0][k].sysMem.virAddr) +
                             count,
                         copy_size) != 0) {
                VLOG(EXAMPLE_DETAIL) << " Error: Agent Embedding copy fail ! ";
              }
              count++;
            }
          }
        }
      }
    }

    if (associated_prev_num_ == 2) {
      std::reverse(temporal_tensor_[tem_emb_idx].begin(),
                   temporal_tensor_[tem_emb_idx].end());
    }

    flush_tensor(&tensor[20]);
    is_temporal_update_[tem_emb_idx] = false;
  }
}

void WorkflowPlugin::GetTemporalAgentEnc(std::vector<hbDNNTensor>& tensor,
                                         int frame_id,
                                         int frames_per_sample_infer) {
  int tensor_idx = temporal_input_tensors_idx_[1];  // get input18 idx
  int tem_enc_idx = 1;
  if (frames_per_sample_infer == 0) {
    for (int k = 0; k < steps_to_decode_; k++) {
      auto& mem = temporal_tensor_[tem_enc_idx][k].sysMem;
      int aligned_size =
          temporal_tensor_[tem_enc_idx][k].properties.alignedByteSize;
      memset(mem.virAddr, 0, aligned_size);
      hbUCPMemFlush(&(temporal_tensor_[tem_enc_idx][k].sysMem),
                    HB_SYS_MEM_CACHE_CLEAN);
    }
    auto& mem = tensor[tensor_idx].sysMem;
    int aligned_size = tensor[tensor_idx].properties.alignedByteSize;
    memset(mem.virAddr, 0, aligned_size);
    hbUCPMemFlush(&(tensor[tensor_idx].sysMem), HB_SYS_MEM_CACHE_CLEAN);
  }
  int idx = 1;
  if (GetTemporalUpdateStatus(idx) && frames_per_sample_infer >= 3) {
    hbUCPMemFlush(&(tensor[tensor_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    for (int k = 0; k < temporal_tensor_[tem_enc_idx].size(); k++) {
      hbUCPMemFlush(&(temporal_tensor_[tem_enc_idx][k].sysMem),
                    HB_SYS_MEM_CACHE_INVALIDATE);
    }
    int tensor_type = temporal_tensor_[tem_enc_idx][1].properties.tensorType;
    int align_size =
        temporal_tensor_[tem_enc_idx][1].properties.alignedByteSize;
    if (align_size != tensor[tensor_idx].properties.alignedByteSize) {
      return;
    }
    auto valid_shape = tensor[tensor_idx].properties.validShape.dimensionSize;
    auto aligned_shape = properies2alignshape(tensor[tensor_idx].properties);

    for (int i = 0; i < steps_to_decode_; i++) {
      int dest_h = i;
      if (tensor_type == HB_DNN_TENSOR_TYPE_S16) {
        // HB_DNN_TENSOR_TYPE_S8
        int valid_n = valid_shape[0];
        int valid_c = valid_shape[1];
        int valid_h = valid_shape[2];
        int valid_w = valid_shape[3];
        int aligned_n = aligned_shape[0];  // 1
        int aligned_c = aligned_shape[1];  // 30
        int aligned_h = aligned_shape[2];  // 6
        int aligned_w = aligned_shape[3];  // 128
        auto copy_size = sizeof(int16_t);
        // 保持 n, c, h, w 结构
        int count = 0;
        for (int n = 0; n < valid_n; n++) {
          int dest_nchw = n * aligned_c * aligned_h * aligned_w;
          for (int c = 0; c < valid_c; c++) {
            int dest_chw = dest_nchw + c * aligned_h * aligned_w;
            int dest_hw = dest_chw + dest_h * aligned_w;
            for (int w = 0; w < valid_w; w++) {
              int dest_w = dest_hw + w;
              memcpy(reinterpret_cast<int16_t*>(
                         tensor[tensor_idx].sysMem.virAddr) +
                         dest_w,
                     reinterpret_cast<int16_t*>(
                         temporal_tensor_[tem_enc_idx][dest_h].sysMem.virAddr) +
                         count,
                     copy_size);
              count++;
            }
          }
        }
        if (count != valid_n * valid_c * valid_w) {
          VLOG(EXAMPLE_DETAIL)
              << " Error: count != valid_n * valid_c * valid_w"
              << " count  " << count << " but valid_n * valid_c  * valid_w is "
              << valid_n * valid_c * valid_w;
        }
      }
    }
  } else {
    VLOG(EXAMPLE_DETAIL) << "  GetTemporalUpdateStatus False "
                         << GetTemporalUpdateStatus(idx);
  }
  is_temporal_update_[tem_enc_idx] = false;
}
template <typename T>
void dequantize_x_a_cur_emb(
    std::vector<std::vector<std::vector<std::vector<T>>>>& res,
    int* valid_shape, int* aligned_shape, float* scale, T* input) {
  if (valid_shape == nullptr) {
    std::cerr << "Error: valid_shape is nullptr!" << std::endl;
  }
  if (aligned_shape == nullptr) {
    std::cerr << "Error: aligned_shape is nullptr!" << std::endl;
  }
  if (scale == nullptr) {
    std::cerr << "Error: scale is nullptr!" << std::endl;
  }
  if (input == nullptr) {
    VLOG(EXAMPLE_DETAIL) << " Error: input is nullptr! ";
  }
  for (int i = 0; i < 4; i++) {
    if (valid_shape[i] <= 0 || aligned_shape[i] <= 0) {
      std::cerr << "Error: Invalid shape value at dimension " << i << "!"
                << std::endl;
    }
  }
  int valid_n = valid_shape[0];
  int valid_c = valid_shape[1];
  int valid_h = valid_shape[2];
  int valid_w = valid_shape[3];

  int aligned_n = aligned_shape[0];
  int aligned_c = aligned_shape[1];
  int aligned_h = aligned_shape[2];
  int aligned_w = aligned_shape[3];

  std::vector<std::vector<std::vector<std::vector<T>>>> tmp_res(
      valid_n, std::vector<std::vector<std::vector<T>>>(
                   valid_c, std::vector<std::vector<T>>(
                                valid_h, std::vector<T>(valid_w, 0))));
  std::vector<std::vector<std::vector<std::vector<float>>>> fp32_tmp_res(
      valid_n, std::vector<std::vector<std::vector<float>>>(
                   valid_c, std::vector<std::vector<float>>(
                                valid_h, std::vector<float>(valid_w, 0))));
  // 保持 n, c, h, w 结构
  for (int n = 0; n < valid_n; n++) {
    int input_nchw = n * aligned_c * aligned_h * aligned_w;
    for (int c = 0; c < valid_c; c++) {
      int input_chw = input_nchw + c * aligned_h * aligned_w;
      for (int h = 0; h < valid_h; h++) {
        int input_hw = input_chw + h * aligned_w;
        for (int w = 0; w < valid_w; w++) {
          if (input_hw + w < 0 ||
              input_hw + w >= (aligned_n * aligned_c * aligned_h * aligned_w)) {
            std::cerr << "Error: input index out of bounds for n=" << n
                      << ", c=" << c << ", h=" << h << ", w=" << w << std::endl;
          }
          fp32_tmp_res[n][c][h][w] = input[input_hw + w] * scale[0];
          tmp_res[n][c][h][w] = input[input_hw + w];
        }
      }
    }
  }

  for (int n = 0; n < valid_n; n++) {
    for (int i = 0; i < valid_c; i++) {
      for (int j = 0; j < valid_h; j++) {
        for (int k = 0; k < valid_w; k++) {
          res[n][i][j][k] = tmp_res[n][i][j][k];
        }
      }
    }
  }
}

WorkflowPlugin::~WorkflowPlugin() {}
