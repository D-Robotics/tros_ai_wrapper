#!/bin/bash
# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e
SCRIPTS_DIR=$(readlink -f "$(dirname "$0")")
cd "$SCRIPTS_DIR" || exit 1

rm -rf dnn_tutorial_data_zoo.tar.gz
curl -O -u 'openexplorer:c5R,2!pG' ftp://vrftp.horizon.ai/ucp/v1.2.10/dnn_tutorial_data_zoo.tar.gz
tar zxf dnn_tutorial_data_zoo.tar.gz
echo "Extract dnn_tutorial_data_zoo success"

rm -rf ../s100/qat/mini_data
rm -rf ../s100/qat/mini_data
cp -r dnn_tutorial_data_zoo/ai_benchmark_mini_data ../s100/qat/mini_data
cp -r dnn_tutorial_data_zoo/ai_benchmark_mini_data ../s100/qat/mini_data
rm -rf dnn_tutorial_data_zoo*
rm -rf ../s100/qat/mini_data/cifar10
rm -rf ../s100/qat/mini_data/culane
rm -rf ../s100/qat/mini_data/flyingchairs
rm -rf ../s100/qat/mini_data/kitti3d
rm -rf ../s100/qat/mini_data/mot17
rm -rf ../s100/qat/mini_data/nuscenes*
rm -rf ../s100/qat/mini_data/argoverse1*
rm -rf ../s100/qat/mini_data/argoverse2_qcnet*
rm -rf ../s100/qat/mini_data/voc
rm -rf ../s100/qat/mini_data/flyingchairs
