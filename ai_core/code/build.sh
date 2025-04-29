#!/usr/bin/env bash
## Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
##
## The material in this file is confidential and contains trade secrets
## of Horizon Robotics Inc. This is proprietary information owned by
## Horizon Robotics Inc. No part of this work may be disclosed,
## reproduced, copied, transmitted, or used in any way for any purpose,
## without the express written permission of Horizon Robotics Inc.
set -x
set -e

#********************
type=qat
dsp=OFF
image=OFF
system=linux
#********************

function show_usage() {
cat <<EOF

Usage: bash -e $0 <options>
available options:
  -s|--system: set target operating system ([linux|qnx]), default is linux
  -t|--type: set type ([qat|qat]), default is qat
  -d|--dsp: set dsp flag ([ON|OFF]), default is OFF
  -i|--image: set build vdsp image flag ([ON|OFF]), default is OFF
  -h|--help
EOF
exit
}

function build_dsp_sample() {
    cd ../../../custom_operator/dsp_sample/dsp_code
    bash build_dsp.sh -a aarch64
    cd -

    centerpoint_path=../s100/qat/script/detection/centerpoint_pointpillar_nuscenes
    if [ ! -d "$centerpoint_path/dsp_image" ]; then
      mkdir $centerpoint_path/dsp_image
    fi

    cp ../../../deps_aarch64/ucp/bin/image/vdsp_image_launch.sh $centerpoint_path/dsp_image/dsp_deploy.sh
    cp ../../../custom_operator/dsp_sample/script/image/vdsp0 $centerpoint_path/dsp_image/
}

function build_arm() {
    DIR=$(cd "$(dirname "$0")";pwd)

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

    cmake -DDSP_ON=${dsp} -DCMAKE_INSTALL_PREFIX=../../s100/${type}/script ..
    make -j8
    make install

    cd ..
    # rm -rf arm_build
}

function build_qnx() {
  DIR=$(cd "$(dirname "$0")";pwd)

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

	cmake -DDSP_ON=${dsp} -DCMAKE_INSTALL_PREFIX=../../s100/${type}/script -DCMAKE_SYSTEM_NAME=QNX ..

	make -j8
	make install

	cd ..
	rm -rf qnx_build
}

SYSTEM_OPTS=(linux qnx)
TYPE_OPTS=(qat qat)
DSP_OPTS=(ON OFF)
IMAGE_OPTS=(ON OFF)
GETOPT_ARGS=`getopt -o s:t:d:i:h -al system:,type:,dsp:,image:,help -- "$@"`

while [ -n "$1" ]
do
  case "$1" in
    -s|--system)
      system=$2
      shift 2
      if [[ ! "${SYSTEM_OPTS[*]}" =~ $system ]] ; then
        echo "invalid system: $system"
        show_usage
      fi
      ;;
    -t|--type)
      type=$2
      shift 2
      if [[ ! "${TYPE_OPTS[*]}" =~ $type ]] ; then
        echo "invalid type: $type"
        show_usage
      fi
      ;;
    -d|--dsp)
      dsp=$2
      shift 2
      if [[ ! "${DSP_OPTS[*]}" =~ $dsp ]] ; then
        echo "invalid dsp opt: $dsp"
        show_usage
      fi
      ;;
    -i|--image)
      image=$2
      shift 2
      if [[ ! "${IMAGE_OPTS[*]}" =~ $image ]] ; then
        echo "invalid image opt: $image"
        show_usage
      fi
      ;;
    -h|--help) show_usage; break;;
    --) break ;;
    *) echo $1,$2 show_usage; break;;
  esac
done

if [[ ${dsp} == "ON" && ${image} == "ON" ]] ; then
    build_dsp_sample
fi

if [[ ${system} == "linux" ]]; then
    build_arm
else
    build_qnx
fi
