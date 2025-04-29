cmake_minimum_required(VERSION 2.8)
project(ai_benchmarks)

set(RELEASE_ROOT ${CMAKE_INSTALL_PREFIX})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(DEPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../deps_aarch64)
  set(HOST_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../deps_aarch64)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
  set(DEPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../deps_qnx)
  set(HOST_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../deps_qnx)
  add_definitions(-D_QNX_SOURCE)
endif()

message(STATUS "DEPS_ROOT: ${DEPS_ROOT}")
message(STATUS "RELEASE_ROOT: ${RELEASE_ROOT}")
message(STATUS "HOST_PACKAGE_DIR: ${HOST_PACKAGE_DIR}")

set(RELEASE_LIB_DIR ${RELEASE_ROOT}/aarch64/lib/)
set(RELEASE_BIN_DIR ${RELEASE_ROOT}/aarch64/bin/)

# libdnn.so depends on system software dynamic link library, use -Wl,-unresolved-symbols=ignore-in-shared-libs to shield during compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wl,-unresolved-symbols=ignore-in-shared-libs")

set(UCP_INCLUDE ${HOST_PACKAGE_DIR}/ucp/include/)
set(UCP_LIB ${HOST_PACKAGE_DIR}/ucp/lib)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${UCP_INCLUDE}
        ${DEPS_ROOT}/glog/include
        ${DEPS_ROOT}/gflags/include
        ${DEPS_ROOT}/rapidjson
        ${DEPS_ROOT}/opencv/include
)

link_directories(
        ${UCP_LIB}
        ${DEPS_ROOT}/glog/lib
        ${DEPS_ROOT}/gflags/lib
        ${DEPS_ROOT}/opencv/lib
)


link_directories(
        /usr/hobot/lib/
	../../../deps_aarch64/
	../../../deps_aarch64/opencv/lib/
        /usr/hobot/lib/
)

if (DSP_ON)
  add_definitions(-DHAVE_DSP)
endif()


if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(os_base_lib pthread rt dl)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
  link_directories(${DEPS_ROOT}/appsdk/appuser/lib)
  set(os_base_lib bpu)
endif()

SET(Link_libs
        dnn
        gflags
        glog
        opencv_world
        hbucp
        ${os_base_lib})


file(GLOB INPUT_SOURCE src/input/*.cc)
file(GLOB OUTPUT_SOURCE src/output/*.cc)
file(GLOB METHOD_SOURCE src/method/*.cc)
file(GLOB PLUGIN_SOURCE src/plugin/*.cc)

set(UTILS_SOURCE
        src/utils/image_utils.cc
        src/utils/data_transformer.cc
        src/utils/nms.cc
        src/utils/stop_watch.cc
        src/utils/tensor_utils.cc
        src/utils/utils.cc
        src/utils/box3d_nms.cc
        )

add_executable(example
        src/simple_example.cc
        ${INPUT_SOURCE}
        ${OUTPUT_SOURCE}
        ${UTILS_SOURCE}
        ${METHOD_SOURCE}
        ${PLUGIN_SOURCE}
        )

if(DSP_ON)
  target_link_libraries(example ${Link_libs} hbdsp_plugin)
else()
  target_link_libraries(example ${Link_libs})
endif()

install(TARGETS example
        DESTINATION ${RELEASE_BIN_DIR}/)

# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhbucp.so
#         DESTINATION ${RELEASE_LIB_DIR})
# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhlog_wrapper.so
#         DESTINATION ${RELEASE_LIB_DIR})
# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libdnn.so
#         DESTINATION ${RELEASE_LIB_DIR})
install(FILES ${DEPS_ROOT}/opencv/lib/libopencv_world.so.3.4
        DESTINATION ${RELEASE_LIB_DIR})
# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhbrt4.so
#         DESTINATION ${RELEASE_LIB_DIR})
# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhbtl.so
#         DESTINATION ${RELEASE_LIB_DIR})
# install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhbtl_ext_dnn.so
#         DESTINATION ${RELEASE_LIB_DIR})
# file(GLOB HLOG_LIBS "${HOST_PACKAGE_DIR}/hlog/lib/libhlog.so*")
# install(FILES ${HLOG_LIBS} DESTINATION ${RELEASE_LIB_DIR})

# if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#   install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhb_arm_rpc.so
#           DESTINATION ${RELEASE_LIB_DIR})
#   install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libperfetto_sdk.so
#           DESTINATION ${RELEASE_LIB_DIR})
# endif()

# if (DSP_ON)
#   # install dsp related files to release directory
#   install(FILES ${HOST_PACKAGE_DIR}/ucp/lib/libhbdsp_plugin.so
#           DESTINATION ${RELEASE_LIB_DIR})
# endif()
