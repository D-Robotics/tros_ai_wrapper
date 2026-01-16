#!/usr/bin/env bash
## Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
##
## The material in this file is confidential and contains trade secrets
## of Horizon Robotics Inc. This is proprietary information owned by
## Horizon Robotics Inc. No part of this work may be disclosed,
## reproduced, copied, transmitted, or used in any way for any purpose,
## without the express written permission of Horizon Robotics Inc.
# 一、基础配置（与原脚本一致，保留调试和安全退出机制）
set -x
set -e

# ******************** 默认配置 ********************
platform="s100"  # 默认平台：s100/s600
type="qat"
#********************
# 差异化配置（可根据实际build_s600补充）
declare -A PLATFORM_CONFIG=(
    ["s100:install_prefix"]="../../s100/${type}/script"
    ["s600:install_prefix"]="../../s600/${type}/script"
)


# ******************** 帮助函数 ********************
function show_usage() {
cat <<EOF

Usage: bash -e $0 <options>
available options:
  -p|--platform: set target platform ([s100|s600]), default is s100
  -t|--type: set type ([qat|qat]), default is qat
  -h|--help: show this help info
EOF
exit
}


# ******************** Linux/ARM构建（通用逻辑+平台差异化） ********************
function build_arm() {
    echo "===== Build for Linux ARM64 ====="
    local DIR=$(cd "$(dirname "$0")";pwd)
    local install_prefix=${PLATFORM_CONFIG[${platform}:install_prefix]}

    # 编译器配置（与原脚本一致，保留注释的交叉编译配置，调试用系统 gcc）
    if [ ${LINARO_GCC_ROOT} ]; then
      LINARO_GCC_ROOT=${LINARO_GCC_ROOT}
    else
      LINARO_GCC_ROOT=/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu
    fi

    #export CC="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-gcc"
    #export CXX="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-g++"
    
    export CC=gcc
    export CXX=g++
    
    cd ${DIR}
    # rm -rf arm_build
    mkdir -p arm_build
    cd arm_build

    cmake -DDSP_ON=${dsp} -DCMAKE_INSTALL_PREFIX=${install_prefix} ..
    make -j2
    make install

    cd ..
    # 可选：保留build目录便于调试，注释rm -rf arm_build
}


# ******************** 参数解析 ********************
# 定义合法参数列表
PLATFORM_OPTS=(s100 s600)
TYPE_OPTS=(qat qat)

# 解析命令行参数
GETOPT_ARGS=`getopt -o p:t:h -al platform:,type:,help -- "$@"`
eval set -- "${GETOPT_ARGS}"

while [ -n "$1" ]
do
  case "$1" in
    -p|--platform)
      platform=$2
      shift 2
      if [[ ! "${PLATFORM_OPTS[*]}" =~ $platform ]] ; then
        echo "invalid platform: $platform, only support ${PLATFORM_OPTS[*]}"
        show_usage
      fi
      ;;
    -t|--type)
      type=$2
      shift 2
      if [[ ! "${TYPE_OPTS[*]}" =~ $type ]] ; then
        echo "invalid type: $type, only support ${TYPE_OPTS[*]}"
        show_usage
      fi
      ;;
    -h|--help) 
      show_usage
      break
      ;;
    --) 
      break 
      ;;
    *) 
      echo "invalid option: $1"
      show_usage
      break
      ;;
  esac
done

# ******************** 主构建流程 ********************
# 移除：dsp/image相关判断逻辑、system分支判断
# 直接执行默认的Linux ARM构建（原system=linux对应的逻辑）
build_arm

set +x
echo "Build completed for platform: ${platform}, system: ${system}, type: ${type}"