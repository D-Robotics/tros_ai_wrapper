// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_BASE_PLUGIN_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_BASE_PLUGIN_H_

#include <memory>
#include <string>
#include <utility>

#include "base/perception_common.h"
#include "input/input_data.h"

typedef std::shared_ptr<ImageTensor> ImageTensorPtr;
typedef std::shared_ptr<Perception> PerceptionPtr;

typedef std::pair<ImageTensorPtr, PerceptionPtr> OutputMessage;

class BasePlugin {
 public:
  BasePlugin() = default;
  virtual int Init(std::string config_file, std::string config_string);

  virtual int Start() = 0;

  virtual int Stop() = 0;

  virtual ~BasePlugin() = default;

 protected:
  virtual int LoadConfig(std::string &config_string) { return 0; }

 private:
  int LoadConfigFile(std::string &config_file);
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_PLUGIN_BASE_PLUGIN_H_
