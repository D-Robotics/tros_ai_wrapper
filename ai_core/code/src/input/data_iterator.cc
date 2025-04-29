// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/data_iterator.h"

#include "base/common_def.h"
#include "glog/logging.h"

int DataIterator::Init(std::string config_string) {
  if (!config_string.empty()) {
    int ret_code = this->LoadConfig(config_string);
    if (ret_code != 0) {
      return ret_code;
    }
  }

  return 0;
}

DataIterator* DataIterator::GetImpl(const std::string& module_name) {
  return DataIteratorFactory::GetInstance()->GetDataIterator(
      module_name.data());
}
