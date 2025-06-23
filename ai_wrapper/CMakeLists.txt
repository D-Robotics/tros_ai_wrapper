cmake_minimum_required(VERSION 3.8)
project(tros_ai_wrapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -ggdb)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4 REQUIRED)

set(DEPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../ai_core/deps_aarch64)
set (SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../ai_core/code/)

message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("DEPS_ROOT: ${DEPS_ROOT}")

# libdnn.so depends on system software dynamic link library, use -Wl,-unresolved-symbols=ignore-in-shared-libs to shield during compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wl,-unresolved-symbols=ignore-in-shared-libs")

include_directories(
  ${SRC_PATH}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${DEPS_ROOT}/gflags/include
  ${DEPS_ROOT}/glog/include
)

link_directories(
  /usr/hobot/lib/
  ${DEPS_ROOT}/gflags/lib
  ${DEPS_ROOT}/glog/lib
)

if (DSP_ON)
  add_definitions(-DHAVE_DSP)
endif()

set(os_base_lib pthread rt dl)

SET(Link_libs
        dnn
        gflags
        glog
        hbucp
        ${os_base_lib})

file(GLOB INPUT_SOURCE ${SRC_PATH}/src/input/*.cc)
file(GLOB OUTPUT_SOURCE ${SRC_PATH}/src/output/*.cc)
file(GLOB METHOD_SOURCE ${SRC_PATH}/src/method/*.cc)
file(GLOB PLUGIN_SOURCE ${SRC_PATH}/src/plugin/*.cc)

set(UTILS_SOURCE
        ${SRC_PATH}/src/utils/image_utils.cc
        ${SRC_PATH}/src/utils/data_transformer.cc
        ${SRC_PATH}/src/utils/nms.cc
        ${SRC_PATH}/src/utils/stop_watch.cc
        ${SRC_PATH}/src/utils/tensor_utils.cc
        ${SRC_PATH}/src/utils/utils.cc
        ${SRC_PATH}/src/utils/box3d_nms.cc
        )

add_library(${PROJECT_NAME} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_wrapper.cpp
        ${INPUT_SOURCE}
        ${OUTPUT_SOURCE}
        ${UTILS_SOURCE}
        ${METHOD_SOURCE}
        ${PLUGIN_SOURCE}
        )

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  OpenCV
)

if(DSP_ON)
  target_link_libraries(${PROJECT_NAME} ${Link_libs} hbdsp_plugin)
else()
  target_link_libraries(${PROJECT_NAME} ${Link_libs})
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ai_wrapper.h
  DESTINATION include/${PROJECT_NAME}
)


install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# install(DIRECTORY
#   ${PROJECT_SOURCE_DIR}/../ai_core/s100/qat
#   DESTINATION lib/${PROJECT_NAME}/
# )

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/../ai_core/s100/qat/script/
  DESTINATION lib/qat/script/
)
install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/../ai_core/s100/qat/model/runtime/bev_gkt_mixvargenet_multitask_nuscenes/
  DESTINATION lib/qat/model/runtime/bev_gkt_mixvargenet_multitask_nuscenes/
)
install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/../ai_core/s100/qat/model/runtime/centerpoint_pointpillar_nuscenes/
  DESTINATION lib/qat/model/runtime/centerpoint_pointpillar_nuscenes/
)
install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/../ai_core/s100/qat/mini_data/
  DESTINATION lib/qat/mini_data/
)

# specific order: dependents before dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME})


ament_package()
