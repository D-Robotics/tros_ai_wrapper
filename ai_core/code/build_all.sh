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
dsp="OFF"
image="OFF"
system="linux"
#********************
# 差异化配置（可根据实际build_s600补充）
declare -A PLATFORM_CONFIG=(
    ["s100:install_prefix"]="../../s100/${type}/script"
    ["s100:dsp_path"]="../s100/qat/script/detection/centerpoint_pointpillar_nuscenes"
    ["s600:install_prefix"]="../../s600/${type}/script"
    ["s600:dsp_path"]="../s600/qat/script/detection/centerpoint_pointpillar_nuscenes"
)

# ******************** 帮助函数 ********************
function show_usage() {
cat <<EOF

Usage: bash -e $0 <options>
available options:
  -p|--platform: set target platform ([s100|s600]), default is s100
  -s|--system: set target operating system ([linux|qnx]), default is linux
  -t|--type: set type ([qat|qat]), default is qat
  -d|--dsp: set dsp flag ([ON|OFF]), default is OFF
  -i|--image: set build vdsp image flag ([ON|OFF]), default is OFF
  -h|--help: show this help info
EOF
exit
}


# ******************** DSP构建（平台差异化） ********************
function build_dsp_sample() {
    echo "===== Build DSP Sample Image ====="
    cd ../../../custom_operator/dsp_sample/dsp_code
    bash build_dsp.sh -a aarch64
    cd -

    local dsp_path=${PLATFORM_CONFIG[${platform}:dsp_path]}
    if [ ! -d "${dsp_path}/dsp_image" ]; then
      mkdir -p ${dsp_path}/dsp_image
    fi

    cp ../../../deps_aarch64/ucp/bin/image/vdsp_image_launch.sh ${dsp_path}/dsp_image/dsp_deploy.sh
    cp ../../../custom_operator/dsp_sample/script/image/vdsp0 ${dsp_path}/dsp_image/
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


# ******************** QNX构建（通用逻辑+平台差异化） ********************
function build_qnx() {
    local DIR=$(cd "$(dirname "$0")";pwd)
    local install_prefix=${PLATFORM_CONFIG[${platform}:install_prefix]}

    # QNX环境检查
    if [ ! $QNX_HOST ]; then
      if [ ! -f /opt/qnx800/qnxsdp-env.sh ]; then
        echo "Please set environment QNX_HOST correctly"
        exit
      else
        source /opt/qnx800/qnxsdp-env.sh
      fi
    fi
    export CC=${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx8.0.0-gcc
    export CXX=${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx8.0.0-g++

    cd ${DIR}
    rm -rf qnx_build
    mkdir qnx_build
    cd qnx_build

    cmake -DDSP_ON=${dsp} -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCMAKE_SYSTEM_NAME=QNX ..

    make install
    make -j8
    
    cd ..
    rm -rf qnx_build
}



# ******************** 参数解析 ********************
# 定义合法参数列表
PLATFORM_OPTS=(s100 s600)
SYSTEM_OPTS=(linux qnx)
TYPE_OPTS=(qat qat)
DSP_OPTS=(ON OFF)
IMAGE_OPTS=(ON OFF)

# 解析命令行参数
GETOPT_ARGS=`getopt -o p:s:t:d:i:h -al platform:,system:,type:,dsp:,image:,help -- "$@"`
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
    -s|--system)
      system=$2
      shift 2
      if [[ ! "${SYSTEM_OPTS[*]}" =~ $system ]] ; then
        echo "invalid system: $system, only support ${SYSTEM_OPTS[*]}"
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
    -d|--dsp)
      dsp=$2
      shift 2
      if [[ ! "${DSP_OPTS[*]}" =~ $dsp ]] ; then
        echo "invalid dsp opt: $dsp, only support ${DSP_OPTS[*]}"
        show_usage
      fi
      ;;
    -i|--image)
      image=$2
      shift 2
      if [[ ! "${IMAGE_OPTS[*]}" =~ $image ]] ; then
        echo "invalid image opt: $image, only support ${IMAGE_OPTS[*]}"
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
if [[ ${dsp} == "ON" && ${image} == "ON" ]] ; then
    build_dsp_sample
fi

# 按系统类型选择构建逻辑
if [[ ${system} == "linux" ]]; then
    build_arm
elif [[ ${system} == "qnx" ]]; then
    build_qnx
else
    echo "unsupported system: ${system}"
    show_usage
fi

set +x
echo "Build completed for platform: ${platform}, system: ${system}, type: ${type}"
