// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_OUTPUT_PLUGIN_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_OUTPUT_PLUGIN_H_

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

#include "base_plugin.h"
#include "output/output.h"
#include "utils/pc_queue.h"

/**
 * Plugin for output consumer
 */
class OutputConsumerPlugin : public BasePlugin {
 public:
  static OutputConsumerPlugin *GetInstance() {
    static OutputConsumerPlugin instance;
    return &instance;
  }

  /**
   * Input output consumer plugin from config file & config string
   * @param[in] config_file: config file path
   *        the config file should be in the json format
   *        for example:
   *        {
   *            "output_type": "client", # one of [image, video, raw, client]
   *            ... # see  `client_output`
   *                # `image_list_output`
   *                # `raw_output`
   *                # `video_output` and so on
   *        }
   * @param[in] config_string: same as config file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string) override;

  int LoadConfig(std::string &config_string) override;

  int Send(OutputMessage msg);

  void Run();

  int Start() override;

  int Stop() override;
  int GetInferFramNums() { return infer_fram_nums_; }

  void SetCallback(std::function<void(ImageTensor *, Perception *)> callback);
  
  void SetRenderImgsFunc(std::function<void(const std::vector<cv::Mat>& imgs, ImageTensor *frame, Perception *perception)> func) {
    if (output_module_) {
      output_module_->SetRenderImgsFunc(func);
    }
  }

  ~OutputConsumerPlugin() override;

 private:
  OutputConsumerPlugin() = default;

 private:
  OutputModule *output_module_{nullptr};
  std::unordered_map<int, OutputMessage> cache_;
  int next_frame{0};
  // consume in order (order by frame id)
  bool in_order_{true};
  bool stop_{false};
  bool is_temporal_multiframe_model_{false};
  int infer_fram_nums_{8};
  std::shared_ptr<std::thread> consume_thread_;
  PCQueue<OutputMessage> pc_queue_;
  uint64_t frame_cnt_{0};

  std::function<void(ImageTensor *, Perception *)> callback_ = nullptr;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_OUTPUT_PLUGIN_H_
