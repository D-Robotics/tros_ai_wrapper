// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "plugin/output_plugin.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "base/common_def.h"
#include "glog/logging.h"
#include "plugin/input_plugin.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"

int OutputConsumerPlugin::Init(std::string config_file,
                               std::string config_string) {
  return BasePlugin::Init(config_file, config_string);
}

int OutputConsumerPlugin::Send(OutputMessage msg) {
  pc_queue_.put(msg);
  return 0;
}

void OutputConsumerPlugin::Run() {
  bool show_fps = false;
  if (getenv("SHOW_FPS_LOG")) {
    show_fps = true;
  }
  bool show_latency = false;
  if (getenv("SHOW_LATENCY_LOG")) {
    show_latency = true;
  }

  uint64_t statistic_cycle = 1000;
  char* stat_cycle = getenv("STAT_CYCLE");
  if (stat_cycle) {
    statistic_cycle = atoi(stat_cycle);
  }

  uint64_t max_pre_duration = 0U;
  uint64_t min_pre_duration = UINT64_MAX;
  uint64_t total_pre_duration = 0U;

  uint64_t max_pp_duration = 0U;
  uint64_t min_pp_duration = UINT64_MAX;
  uint64_t total_pp_duration = 0U;

  uint64_t max_infer_duration = 0;
  uint64_t min_infer_duration = UINT64_MAX;
  uint64_t total_infer_duration = 0U;

  uint64_t start_ts{0U};
  uint64_t end_ts{0U};

  while (!stop_) {
    OutputMessage msg;
    if (!pc_queue_.get(msg, 1000)) {
      if (stop_) {
        break;
      }
      continue;
    }

    auto perception = msg.second;

    frame_cnt_++;

    if (show_latency) {
      max_infer_duration =
          std::max(max_infer_duration, perception->infer_duration);
      min_infer_duration =
          std::min(min_infer_duration, perception->infer_duration);
      total_infer_duration += perception->infer_duration;

      max_pp_duration = std::max(max_pp_duration, perception->pp_duration);
      min_pp_duration = std::min(min_pp_duration, perception->pp_duration);
      total_pp_duration += perception->pp_duration;

      max_pre_duration = std::max(max_pre_duration, perception->pre_duration);
      min_pre_duration = std::min(min_pre_duration, perception->pre_duration);
      total_pre_duration += perception->pre_duration;

      statistic_cycle = statistic_cycle == 0 ? 1000 : statistic_cycle;
      if (frame_cnt_ % statistic_cycle == 0) {
        if (total_pre_duration > 0) {
          VLOG(EXAMPLE_REPORT)
              << std::fixed << std::setprecision(3) << RED_COMMENT_START
              << "Pre process latency: [avg:  "
              << (total_pre_duration - min_pre_duration - max_pre_duration) /
                     1000.0 / (statistic_cycle - 2)
              << "ms,  max:  " << max_pre_duration / 1000.0
              << "ms,  min:  " << min_pre_duration / 1000.0 << "ms],"
              << " Infer latency:  [avg:  "
              << (total_infer_duration - min_infer_duration -
                  max_infer_duration) /
                     1000.0 / (statistic_cycle - 2)
              << "ms,  max:  " << max_infer_duration / 1000.0
              << "ms,  min:  " << min_infer_duration / 1000.0 << "ms],"
              << " Post process latency: [avg:  "
              << (total_pp_duration - min_pp_duration - max_pp_duration) /
                     1000.0 / (statistic_cycle - 2)
              << "ms,  max:  " << max_pp_duration / 1000.0
              << "ms,  min:  " << min_pp_duration / 1000.0 << "ms]."
              << RED_COMMENT_END;
        } else {
          VLOG(EXAMPLE_REPORT)
              << std::fixed << std::setprecision(3) << RED_COMMENT_START
              << " Infer latency:  [avg:  "
              << (total_infer_duration - min_infer_duration -
                  max_infer_duration) /
                     1000.0 / (statistic_cycle - 2)
              << "ms,  max:  " << max_infer_duration / 1000.0
              << "ms,  min:  " << min_infer_duration / 1000.0 << "ms],"
              << " Post process latency: [avg:  "
              << (total_pp_duration - min_pp_duration - max_pp_duration) /
                     1000.0 / (statistic_cycle - 2)
              << "ms,  max:  " << max_pp_duration / 1000.0
              << "ms,  min:  " << min_pp_duration / 1000.0 << "ms]."
              << RED_COMMENT_END;
        }

        std::ostringstream json_output;
        json_output << std::fixed << std::setprecision(3)
                    << "{\"Pre process latency\": {\"avg\": "
                    << (total_pre_duration - min_pre_duration -
                        max_pre_duration) /
                           1000.0 / (statistic_cycle - 2)
                    << ", \"max\": " << max_pre_duration / 1000.0
                    << ", \"min\": " << min_pre_duration / 1000.0
                    << "}, \"Infer latency\": {\"avg\": "
                    << (total_infer_duration - min_infer_duration -
                        max_infer_duration) /
                           1000.0 / (statistic_cycle - 2)
                    << ", \"max\": " << max_infer_duration / 1000.0
                    << ", \"min\": " << min_infer_duration / 1000.0
                    << "}, \"Post process latency\": {\"avg\": "
                    << (total_pp_duration - min_pp_duration - max_pp_duration) /
                           1000.0 / (statistic_cycle - 2)
                    << ", \"max\": " << max_pp_duration / 1000.0
                    << ", \"min\": " << min_pp_duration / 1000.0 << "}}";

        const char* env_val = std::getenv("AI_BENCHMARK_LOG_TO_FILE");
        if (env_val && std::string(env_val) == "1") {
          std::ofstream latency_file("latency.txt", std::ios_base::app);
          if (latency_file.is_open()) {
            latency_file << json_output.str() << std::endl;
            latency_file.close();
          }
        }

        max_pp_duration = 0U;
        min_pp_duration = UINT64_MAX;
        total_pp_duration = 0U;

        max_infer_duration = 0U;
        min_infer_duration = UINT64_MAX;
        total_infer_duration = 0U;

        max_pre_duration = 0U;
        min_pre_duration = UINT64_MAX;
        total_pre_duration = 0U;
      }
    }

    if (show_fps) {
      if (frame_cnt_ % statistic_cycle == 0) {
        if (frame_cnt_ != statistic_cycle /*ignore first stat cycle*/) {
          end_ts = Stopwatch::CurrentTs();
          double fps = 1000000.0 * statistic_cycle / (end_ts - start_ts);
          VLOG(EXAMPLE_REPORT)
              << std::fixed << std::setprecision(2) << RED_COMMENT_START
              << "Throughput: " << fps << "fps" << RED_COMMENT_END;

          const char* env_val = std::getenv("AI_BENCHMARK_LOG_TO_FILE");
          if (env_val && std::string(env_val) == "1") {
            std::ofstream fps_file("fps.txt", std::ios_base::app);
            if (fps_file.is_open()) {
              fps_file << fps << std::endl;
              fps_file.close();
            }
          }
        }
        start_ts = Stopwatch::CurrentTs();
      }
    }

    if (in_order_) {
      // output in order (order by frame id, may block previous frames)
      auto frame_id = msg.first->frame_id;
      cache_[frame_id] = msg;
      while (cache_.count(next_frame)) {
        auto output_msg = cache_[next_frame];
        auto image_tensor = output_msg.first;
        auto perception = output_msg.second;
        output_module_->Write(image_tensor.get(), perception.get());
        InputProducerPlugin::GetInstance()->Release(image_tensor);
        cache_.erase(next_frame++);
      }
    } else {
      auto output_msg = msg;
      auto image_tensor = output_msg.first;
      auto perception = output_msg.second;
      if (is_temporal_multiframe_model_) {
        // Attention! The current temporal multi-frame inference model only supports QCNet.
        auto frames_per_sample_infer = msg.first->frames_per_sample_infer;
        if (frames_per_sample_infer == infer_fram_nums_ - 1) {
          output_module_->Write(image_tensor.get(), perception.get());
        }
        InputProducerPlugin::GetInstance()->Release(image_tensor);
      } else {
        output_module_->Write(image_tensor.get(), perception.get());
        InputProducerPlugin::GetInstance()->Release(image_tensor);
      }
    }
  }
  VLOG(EXAMPLE_DEBUG) << "OutputConsumerPlugin Send finished.";
}

int OutputConsumerPlugin::Start() {
  stop_ = false;
  consume_thread_ =
      std::make_shared<std::thread>(&OutputConsumerPlugin::Run, this);
  if (!consume_thread_) {
    VLOG(EXAMPLE_SYSTEM) << "Start thread failed.";
    return -1;
  }

  VLOG(EXAMPLE_DETAIL) << "OutputConsumerPlugin start.";
  return 0;
}

int OutputConsumerPlugin::Stop() {
  stop_ = true;
  consume_thread_->join();
  VLOG(EXAMPLE_DETAIL) << "OutputConsumerPlugin stop.";
  return 0;
}

int OutputConsumerPlugin::LoadConfig(std::string& config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }
  if (document.HasMember("output_type")) {
    std::string output_type = document["output_type"].GetString();
    output_module_ = OutputModule::GetImpl(output_type);
  } else {
    VLOG(EXAMPLE_SYSTEM)
        << "output config don not have parameter output_type! please check!";
    return -1;
  }
  if (document.HasMember("in_order")) {
    in_order_ = document["in_order"].GetBool();
  }
  if (document.HasMember("is_temporal_multiframe_model")) {
    is_temporal_multiframe_model_ =
        document["is_temporal_multiframe_model"].GetBool();
  }
  if (document.HasMember("infer_fram_nums")) {
    infer_fram_nums_ = document["infer_fram_nums"].GetInt();
  }
  return output_module_->Init("", config_string);
}

OutputConsumerPlugin::~OutputConsumerPlugin() {
  if (output_module_) {
    delete output_module_;
    output_module_ = nullptr;
  }
}

void OutputConsumerPlugin::SetCallback(std::function<void(ImageTensor *, Perception *)> callback) {
  VLOG(EXAMPLE_DETAIL) << "Set Callback\n";
  callback_ = callback;
}
