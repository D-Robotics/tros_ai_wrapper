// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>
#include <sys/stat.h>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

#include "rclcpp/rclcpp.hpp"

#include "ai_wrapper.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "input/data_iterator.h"
#include "plugin/input_plugin.h"
#include "plugin/output_plugin.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/writer.h"
#include "utils/image_utils.h"

#ifndef EMPTY
#define EMPTY ""
#endif

#ifndef glog_level
DEFINE_int32(glog_level, google::WARNING,
             "Logging level (INFO=0, WARNING=1, ERROR=2, FATAL=3)");
#endif

#ifndef config_file
DEFINE_string(config_file, EMPTY, "Json config file");
#endif

static std::shared_ptr<std::thread> task_thread_ = nullptr;

static InputProducerPlugin* input_plg = nullptr;
static WorkflowPlugin* workflow_plg = nullptr;
static OutputConsumerPlugin* output_plg = nullptr;

AIWrapper::AIWrapper() {
}

AIWrapper::~AIWrapper() {
  if (task_thread_ && task_thread_->joinable()) {
    task_thread_->join();
  }
}

void AIWrapper::Init(int argc, char **argv) {
  RCLCPP_WARN(rclcpp::get_logger("ai_wrapper"),
    "\n Set glog level in cmd line with '--glog_level=$num' \n\t EXAMPLE_SYSTEM = 0,  EXAMPLE_REPORT = 1,  EXAMPLE_DETAIL = 2,  EXAMPLE_DEBUG = 3");

  // Parsing command line arguments
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << gflags::GetArgv() << std::endl;

  // Init logging
  google::InitGoogleLogging("");
  google::SetStderrLogging(0);
  FLAGS_colorlogtostderr = true;
  google::SetVLOGLevel("*", FLAGS_glog_level);
  FLAGS_max_log_size = 200;
  FLAGS_logbufsecs = 0;
  FLAGS_logtostderr = true;

  VLOG(EXAMPLE_SYSTEM) << "EXAMPLE_SYSTEM";
  VLOG(EXAMPLE_REPORT) << "EXAMPLE_REPORT";
  VLOG(EXAMPLE_DETAIL) << "EXAMPLE_DETAIL";
  VLOG(EXAMPLE_DEBUG) << "EXAMPLE_DEBUG";

  // Parsing config
  std::string config_fname = FLAGS_config_file;
  VLOG(EXAMPLE_SYSTEM) << "config_fname: " << config_fname;
  std::ifstream ifs(config_fname);
  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document document;
  document.ParseStream(isw);
  LOG_IF(FATAL, document.HasParseError()) << "Parsing config file failed";
  LOG_IF(FATAL, !(document.HasMember("input_config") &&
                  document.HasMember("workflow") &&
                  document.HasMember("output_config")))
      << "Missing config for input/workflow/output";

  // Workflow plugin
  workflow_plg = WorkflowPlugin::GetInstance();
  int ret_code = workflow_plg->Init(EMPTY, json_to_string(document));
  LOG_IF(FATAL, ret_code != 0) << "Workflow init failed";

  // Input plugin
  input_plg = InputProducerPlugin::GetInstance();
  ret_code = input_plg->Init(EMPTY, json_to_string(document["input_config"]));
  LOG_IF(FATAL, ret_code != 0) << "Input plugin init failed";

  // Output plugin
  output_plg = OutputConsumerPlugin::GetInstance();
  ret_code = output_plg->Init(EMPTY, json_to_string(document["output_config"]));
  LOG_IF(FATAL, ret_code != 0) << "Output plugin init failed";
}

void AIWrapper::SetOutputCallback(
  std::function<void(const std::vector<cv::Mat>& imgs, std::shared_ptr<FrameInfo> frame)> callback) {
  output_plg->SetRenderImgsFunc([this, callback](const std::vector<cv::Mat>& imgs, ImageTensor *frame, Perception *perception){
    auto frame_info = std::make_shared<FrameInfo>();
    if (frame) {
      frame_info->frame_id = frame->frame_id;
    }
    if (perception) {
      frame_info->infer_duration = perception->infer_duration;
      frame_info->pp_duration = perception->pp_duration;
      frame_info->pre_duration = perception->pre_duration;
    }
    callback(imgs, frame_info);
  });
}

void AIWrapper::Start() {
  if (!is_running_) {
    is_running_ = true;

    // Start
    int ret_code = input_plg->Start();
    LOG_IF(FATAL, ret_code != 0) << "Input plugin start failed";
    ret_code = workflow_plg->Start();
    LOG_IF(FATAL, ret_code != 0) << "Workflow plugin start failed";
    ret_code = output_plg->Start();
    LOG_IF(FATAL, ret_code != 0) << "Output plugin start failed";
    
    if (!task_thread_) {
      task_thread_ = std::make_shared<std::thread>([&]() {
        while (is_running_) {
          if (!input_plg->IsRunning()) {
            VLOG(EXAMPLE_REPORT) << "finished!";
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      });
    }
  }

}

void AIWrapper::Stop() {
  VLOG(EXAMPLE_REPORT) << "stop start";
  is_running_ = false;
  // Stop
  input_plg->Stop();
  workflow_plg->Stop();
  output_plg->Stop();
  VLOG(EXAMPLE_REPORT) << "stop complete";
}
