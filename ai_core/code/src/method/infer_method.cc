// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.
#include "method/infer_method.h"

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(InferMethod);

bool InferMethod::is_releated_model_{};
bool InferMethod::is_temporal_model_{};
bool InferMethod::is_temporal_multiframe_model_{};
int InferMethod::input_num_{};

int InferMethod::InitFromJsonString(const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "InferMethod Json string:" << config.data();

  rapidjson::Document document;
  document.Parse(config.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("core")) {
    core_ = document["core"].GetInt();
  }

  if (document.HasMember("model_file")) {
    std::string model_file = document["model_file"].GetString();
    const char *model_file_path = model_file.c_str();
    HB_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_path, 1),
        "hbDNNInitializeFromFiles failed");

    const char **model_name_list;
    int model_count = 0;
    HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count,
                                           packed_dnn_handle_),
                     "hbDNNGetModelNameList failed");

    HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_,
                                         model_name_list[0]),
                     "hbDNNGetModelHandle failed");
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Please set model_file in method_config_file";
    return -1;
  }

  if (document.HasMember("tensor_pool_size")) {
    tensor_pool_size_ = document["tensor_pool_size"].GetInt();
  }

  // if it is releated_model, the other models' output is releated one model
  if (document.HasMember("is_releated_model")) {
    is_releated_model_ = document["is_releated_model"].GetBool();
  }

  if (document.HasMember("is_temporal_model")) {
    is_temporal_model_ = document["is_temporal_model"].GetBool();
  }
  if (document.HasMember("is_temporal_multiframe_model")) {
    is_temporal_multiframe_model_ =
        document["is_temporal_multiframe_model"].GetBool();
  }

  hbDNNGetInputCount(&input_num_, dnn_handle_);

  int out_num;
  hbDNNGetOutputCount(&out_num, dnn_handle_);
  std::vector<hbDNNTensorProperties> properties;
  properties.resize(out_num);
  for (int i = 0; i < out_num; ++i) {
    hbDNNGetOutputTensorProperties(&properties[i], dnn_handle_, i);
  }
  tensor_vector_pool_ = Pool<TensorVector>::Create(
      tensor_pool_size_, tensor_pool_size_, properties);

  hbDNNGetInputCount(&input_num_, dnn_handle_);

  return 0;
}

TensorVectorPtr InferMethod::DoProcess(ImageTensor *image_tensor) {
  auto output_tensors = tensor_vector_pool_->GetSharedPtr(1);
  while (!output_tensors) {
    VLOG_EVERY_N(EXAMPLE_SYSTEM, 100)
        << "Get output tensor from pool timeout, try again. "
           "You should increase the pool size or post process thread number";
    output_tensors = tensor_vector_pool_->GetSharedPtr(1);
  }
  VLOG(EXAMPLE_DEBUG) << "prepare output tensor success";

  // Run inference
  hbUCPTaskHandle_t task_handle = nullptr;
  auto output = output_tensors->tensors.data();
  if (is_releated_model_) {
    WorkflowPlugin::GetInstance()->GetMajorModelTensor(image_tensor->tensors);
  }

  if (is_temporal_model_) {
    WorkflowPlugin::GetInstance()->GetTemporalModelTensor(
        image_tensor->tensors, image_tensor->frame_id);
  }
  if (is_temporal_multiframe_model_) {
    VLOG(EXAMPLE_DETAIL) << "Attention! The current temporal multi-frame "
                            "inference model only supports QCNet.";
    // refresh agent embedding
    WorkflowPlugin::GetInstance()->GetTemporalAgentEmb(
        image_tensor->tensors, image_tensor->frame_id,
        image_tensor->frames_per_sample_infer);
    // refresh agent encoder
    WorkflowPlugin::GetInstance()->GetTemporalAgentEnc(
        image_tensor->tensors, image_tensor->frame_id,
        image_tensor->frames_per_sample_infer);
  }

  // multi input model
  METHOD_CHECK_SUCCESS(
      hbDNNInferV2(&task_handle, output, (image_tensor->tensors).data(),
                   dnn_handle_),
      "hbDNNInferV2 failed", nullptr);

  VLOG(EXAMPLE_DEBUG) << "multi input model infer success";

  // submit task
  hbUCPSchedParam sched_param;
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
  sched_param.backend = core_;
  METHOD_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &sched_param),
                       "hbUCPSubmitTask failed", nullptr);

  // wait task done
  METHOD_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                       "hbUCPWaitTaskDone failed", nullptr);
  VLOG(EXAMPLE_DEBUG) << "task done";

  // update dynamic output tensor properties
  auto& ori_output_tensors = output_tensors->tensors;
  for(size_t out_idx{0U}; out_idx < ori_output_tensors.size(); out_idx++) {
      auto& output_tensor = ori_output_tensors[out_idx];
     METHOD_CHECK_SUCCESS(hbDNNGetTaskOutputTensorProperties(&output_tensor.properties, task_handle, 0, out_idx), "hbDNNGetTaskOutputTensorProperties failed", nullptr);
  }
  VLOG(EXAMPLE_DEBUG) << "update dynamic output tensor properties success";

  // release task done
  METHOD_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed",
                       nullptr);
  VLOG(EXAMPLE_DEBUG) << "task release";

  VLOG(EXAMPLE_DETAIL) << "Predict DoProcess finished.";
  return output_tensors;
}

void InferMethod::Rlease() {
  if (packed_dnn_handle_) {
    hbDNNRelease(packed_dnn_handle_);
  }
}

hbDNNHandle_t InferMethod::GetModelHandle() {
  if (dnn_handle_ == nullptr) {
    VLOG(EXAMPLE_SYSTEM) << "Error, dnn_handle_ == nullptr!";
  }
  return dnn_handle_;
}
