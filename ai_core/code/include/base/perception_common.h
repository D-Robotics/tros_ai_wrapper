// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_BASE_PERCEPTION_COMMON_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_BASE_PERCEPTION_COMMON_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * define a quaternion
 * angle, axis x, axis y, axis z
 */
struct Quaternion {
  float w;
  float x;
  float y;
  float z;

  Quaternion(float angle, float axisX, float axisY, float axisZ) {
    float halfAngle = angle * 0.5f;
    float sinHalfAngle = std::sin(halfAngle);
    float cosHalfAngle = std::cos(halfAngle);

    w = cosHalfAngle;
    x = axisX * sinHalfAngle;
    y = axisY * sinHalfAngle;
    z = axisZ * sinHalfAngle;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const Quaternion &quaternion) {
    os << quaternion.w << " + " << quaternion.x << "i + " << quaternion.y
       << "j + " << quaternion.z << "k";
    return os;
  }

  std::vector<std::vector<float>> RotationMatrix() const {
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float xw = x * w;
    float yy = y * y;
    float yz = y * z;
    float yw = y * w;
    float zz = z * z;
    float zw = z * w;

    std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
    rotation[0][0] = 1.0f - 2.0f * (yy + zz);
    rotation[0][1] = 2.0f * (xy - zw);
    rotation[0][2] = 2.0f * (xz + yw);
    rotation[1][0] = 2.0f * (xy + zw);
    rotation[1][1] = 1.0f - 2.0f * (xx + zz);
    rotation[1][2] = 2.0f * (yz - xw);
    rotation[2][0] = 2.0f * (xz - yw);
    rotation[2][1] = 2.0f * (yz + xw);
    rotation[2][2] = 1.0f - 2.0f * (xx + yy);

    return rotation;
  }
};

typedef struct Anchor {
  float cx{0.0};
  float cy{0.0};
  float w{0.0};
  float h{0.0};
  Anchor(float cx, float cy, float w, float h) : cx(cx), cy(cy), w(w), h(h) {}

  friend std::ostream &operator<<(std::ostream &os, const Anchor &anchor) {
    os << "[" << anchor.cx << "," << anchor.cy << "," << anchor.w << ","
       << anchor.h << "]";
    return os;
  }
} Anchor;

/**
 * Bounding box definition
 */
typedef struct Bbox {
  float xmin{0.0};
  float ymin{0.0};
  float xmax{0.0};
  float ymax{0.0};

  Bbox() {}

  Bbox(float xmin, float ymin, float xmax, float ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.xmin << ","
       << bbox.ymin << "," << bbox.xmax << "," << bbox.ymax << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Bbox() {}
} Bbox;

typedef struct Detection {
  int id{0};
  float score{0.0};
  Bbox bbox;
  const char *class_name{nullptr};
  Detection() {}

  Detection(int id, float score, Bbox bbox)
      : id(id), score(score), bbox(bbox) {}

  Detection(int id, float score, Bbox bbox, const char *class_name)
      : id(id), score(score), bbox(bbox), class_name(class_name) {}

  friend bool operator>(const Detection &lhs, const Detection &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection &det) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("label")"
       << ":" << det.id << ","
       << R"("class_name")"
       << ":\"" << det.class_name << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Detection() {}
} Detection;

static bool greater(Detection det1, Detection det2) {
  return (det1.score >= det2.score);
}

typedef struct Lidar3D {
  float xs{0.0};
  float ys{0.0};
  float height{0.0};
  float dim_0{0.0};
  float dim_1{0.0};
  float dim_2{0.0};
  float rot{0.0};
  float vel_0{0.0};
  float vel_1{0.0};

  Lidar3D() {}

  Lidar3D(float x, float y, float h, float d_0, float d_1, float d_2, float r,
          float v_0, float v_1)
      : xs(x),
        ys(y),
        height(h),
        dim_0(d_0),
        dim_1(d_1),
        dim_2(d_2),
        rot(r),
        vel_0(v_0),
        vel_1(v_1) {}

  friend std::ostream &operator<<(std::ostream &os, const Lidar3D &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.xs << "," << bbox.ys
       << "," << bbox.height << "," << bbox.dim_0 << "," << bbox.dim_1 << ","
       << bbox.dim_2 << "," << bbox.rot << "," << bbox.vel_0 << ","
       << bbox.vel_1 << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Lidar3D() {}
} Lidar3D;

typedef struct LidarDetection3D {
  Lidar3D bbox;
  float score{0.0};
  int label;  // classification lable
  LidarDetection3D() {}

  LidarDetection3D(Lidar3D bbox, float score, int label)
      : bbox(bbox), score(score), label(label) {}

  friend bool operator>(const LidarDetection3D &lhs,
                        const LidarDetection3D &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const LidarDetection3D &det) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("score")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("label")"
       << ":" << det.label << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~LidarDetection3D() {}
} LidarDetection3D;

typedef struct EgoBbox {
  float xs{1.0f};
  float ys{1.0f};
  float height{1.0f};
  float dim_0{1.0f};
  float dim_1{1.0f};
  float dim_2{1.0f};
  float rot{1.0f};
  float vel_0{1.0f};
  float vel_1{1.0f};

  EgoBbox() {}

  EgoBbox(float x, float y, float h, float d_0, float d_1, float d_2, float r,
          float v_0, float v_1)
      : xs(x),
        ys(y),
        height(h),
        dim_0(d_0),
        dim_1(d_1),
        dim_2(d_2),
        rot(r),
        vel_0(v_0),
        vel_1(v_1) {}

  friend std::ostream &operator<<(std::ostream &os, const EgoBbox &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.xs << "," << bbox.ys
       << "," << bbox.height << "," << bbox.dim_0 << "," << bbox.dim_1 << ","
       << bbox.dim_2 << "," << bbox.rot << "," << bbox.vel_0 << ","
       << bbox.vel_1 << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~EgoBbox() {}
} EgoBbox;

typedef struct BevDetection3D {
  EgoBbox bbox;
  float score{0.0};
  int label;  // classification lable
  BevDetection3D() {}

  BevDetection3D(EgoBbox bbox, float score, int label)
      : bbox(bbox), score(score), label(label) {}

  friend bool operator>(const BevDetection3D &lhs, const BevDetection3D &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const BevDetection3D &bev) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << bev.bbox << ","
       << R"("score")"
       << ":" << std::fixed << std::setprecision(6) << bev.score << ","
       << R"("label")"
       << ":" << bev.label << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~BevDetection3D() {}
} BevDetection3D;

typedef struct Bbox3D {
  // geometric center coordinates: (x, y, z)
  float x{0.0};
  float y{0.0};
  float z{0.0};
  // width
  float w{0.0};
  // length
  float l{0.0};
  // height
  float h{0.0};
  // yaw angle
  float r{0.0};

  Bbox3D() {}

  Bbox3D(float x, float y, float z, float w, float l, float h, float r)
      : x(x), y(y), z(z), w(w), l(l), h(h), r(r) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox3D &bbox) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << bbox.x << "," << bbox.y
       << "," << bbox.z << "," << bbox.w << bbox.l << "," << bbox.h << ","
       << bbox.r << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Bbox3D() {}
} Bbox3D;

typedef struct Detection3D {
  Bbox3D bbox;
  float score{0.0};  // score
  int dir_label{0};  // direction label
  int top_label;     // classification lable
  Detection3D() {}

  Detection3D(Bbox3D bbox, float score, int dir_label, int top_label)
      : bbox(bbox), score(score), dir_label(dir_label), top_label(top_label) {}

  friend bool operator>(const Detection3D &lhs, const Detection3D &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection3D &det) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("score")"
       << ":" << std::fixed << std::setprecision(6) << det.score << ","
       << R"("dir_label")"
       << ":" << det.dir_label << R"("top_label")"
       << ":" << det.top_label << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~Detection3D() {}
} Detection3D;

typedef struct TrajPred {
  std::vector<float> traj;
  float score{0.0};
  std::string single_bacth_name;
  TrajPred() {}

  TrajPred(std::vector<float> traj, float score) : traj(traj), score(score) {}

  friend bool operator>(const TrajPred &lhs, const TrajPred &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const TrajPred &traj_pred) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("traj")"
       << ":";
    for (int i = 0; i < traj_pred.traj.size(); i++) {
      os << std::fixed << std::setprecision(6) << traj_pred.traj[i] << ",";
    }
    os << R"("score")"
       << ":" << std::fixed << std::setprecision(6) << traj_pred.score
       << "\"} ";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~TrajPred() {}
} TrajPred;

typedef struct TRAJPRED_QCNET {
  std::vector<std::vector<float>> traj;
  float score{0.0};
  float x_a_emb{0};  // int8_t
  float x_a_enc{0};  // int8_t
  TRAJPRED_QCNET() {}

  TRAJPRED_QCNET(std::vector<std::vector<float>> traj, float score)
      : traj(traj), score(score) {}
  friend bool operator>(const TRAJPRED_QCNET &lhs, const TRAJPRED_QCNET &rhs) {
    return (lhs.score > rhs.score);
  }
  friend std::ostream &operator<<(std::ostream &os,
                                  const TRAJPRED_QCNET &qcnet) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "precision.get";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~TRAJPRED_QCNET() {}
} TRAJPRED_QCNET;
typedef struct Campoints {
  std::vector<std::vector<float>> ca2img_v{3, std::vector<float>(3, 0)};
  std::vector<std::vector<float>> bboxes;
  bool bboxes_empty{false};

  Campoints() {}

  Campoints(std::vector<std::vector<float>> &ca2img,
            std::vector<std::vector<float>> &bboxes_res)
      : ca2img_v(ca2img), bboxes(bboxes_res) {}

  ~Campoints() {}
} Campoints;

// use save cambox3d Boxcenter and Boxdim
typedef struct BoxOutParameter {
  float dim1_{0.0};
  float dim2_{0.0};
  float dim3_{0.0};

  BoxOutParameter() {}

  BoxOutParameter(float dim1, float dim2, float dim3)
      : dim1_(dim1), dim2_(dim2), dim3_(dim3) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const BoxOutParameter &boxcenter) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << boxcenter.dim1_ << ","
       << boxcenter.dim2_ << "," << boxcenter.dim3_ << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~BoxOutParameter() {}
} BoxOutParameter;

typedef struct Velocity {
  float dim1_{0.0};
  float dim2_{0.0};

  Velocity() {}

  Velocity(float dim1, float dim2) : dim1_(dim1), dim2_(dim2) {}

  friend std::ostream &operator<<(std::ostream &os, const Velocity &velocity) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "[" << std::fixed << std::setprecision(6) << velocity.dim1_ << ","
       << velocity.dim2_ << "]";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Velocity() {}
} Velocity;

typedef struct DetectionCam3D {
  BoxOutParameter box_center_;
  BoxOutParameter box_dim_;
  float box_yaw_;
  Velocity velocity_;
  float score_;
  int label_;
  int attr_;
  Campoints bboxes_;
  DetectionCam3D() {}

  DetectionCam3D(BoxOutParameter box_center, BoxOutParameter box_dim,
                 float box_yaw, Velocity velocity, float score, int label,
                 int attr, Campoints bboxes)
      : box_center_(box_center),
        box_dim_(box_dim),
        box_yaw_(box_yaw),
        velocity_(velocity),
        score_(score),
        label_(label),
        attr_(attr),
        bboxes_(bboxes) {}

  friend bool operator>(const DetectionCam3D &lhs, const DetectionCam3D &rhs) {
    return (lhs.score_ > rhs.score_);
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const DetectionCam3D &detcam3d) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("box_center")"
       << ":" << detcam3d.box_center_ << ","
       << R"("box_dim")"
       << ":" << detcam3d.box_dim_ << ","
       << R"("box_yaw")"
       << ":" << detcam3d.box_yaw_ << R"("velocity")"
       << ":" << detcam3d.velocity_ << R"("score")"
       << ":" << detcam3d.score_ << R"("label")"
       << ":" << detcam3d.label_ << R"("attr")"
       << ":" << detcam3d.attr_ << "\"}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }
  ~DetectionCam3D() {}
} DetectionCam3D;

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0) {}

  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Classification &cls) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(5) << cls.score << ","
       << R"("label")"
       << ":" << cls.id << ","
       << R"("class_name")"
       << ":"
       << "\"" << cls.class_name << "\""
       << "}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Classification() {}
} Classification;

template <typename T>
struct Parsing {
  std::vector<T> seg;
  int32_t num_classes = 0;
  int32_t width = 0;
  int32_t height = 0;
};

template <typename T>
struct Parsing3d {
  std::vector<T> seg;
  int32_t num_classes = 0;
  int32_t h = 0;
  int32_t w = 0;
  int32_t z = 0;
};

struct Point {
  std::vector<float> point;
  int32_t width = 0;
  int32_t height = 0;
};

struct MaskResultInfo {
  int32_t width = 0;
  int32_t height = 0;
  float h_base = 0;
  float w_base = 0;
  std::vector<Detection> det_info;
  std::vector<float> mask_info;
};

struct KeyPoint {
  std::vector<std::vector<std::pair<float, float>>> points;
  int32_t width = 0;
  int32_t height = 0;
  int32_t groups_size = 0;
  bool is_keypoints{false};
};

typedef struct MapDetection {
  std::vector<std::pair<float, float>> pts;
  float score{0.0};
  int label;  // classification lable
  MapDetection() {}

  MapDetection(std::vector<std::pair<float, float>> &pts, float score,
               int label)
      : pts(pts), score(score), label(label) {}

  friend bool operator>(const MapDetection &lhs, const MapDetection &rhs) {
    return (lhs.score > rhs.score);
  }

  ~MapDetection() {}
} MapDetection;

struct Perception {
  // Perception data
  std::vector<std::vector<TrajPred>> trajPred;  // multi batch
  std::vector<std::vector<TRAJPRED_QCNET>> qcnet;
  std::vector<LidarDetection3D> lidar3d;
  std::vector<Detection> det;
  std::vector<Detection3D> det3d;
  std::vector<DetectionCam3D> detcam3d;
  std::vector<Classification> cls;
  std::vector<BevDetection3D> bevDet3d;
  std::vector<MapDetection> mapDet;
  Parsing<uint32_t> bevSeg;
  Parsing<uint8_t> lidarSeg;
  Parsing<uint8_t> seg;
  Parsing3d<uint32_t> seg3d;
  Point pt;
  MaskResultInfo mask;
  KeyPoint kpt;
  float h_base = 1;
  float w_base = 1;

  // TODO(@horizon.ai): remove from here
  uint64_t infer_duration;
  uint64_t pp_duration;
  uint64_t pre_duration;

  // Perception type
  enum {
    DET = (1 << 0),
    CLS = (1 << 1),
    SEG = (1 << 2),
    MASK = (1 << 3),
    POINT = (1 << 4),
    DET3D = (1 << 5),
    KEYPOINT = (1 << 6),
    DETCAM3D = (1 << 7),
    LIDAR3D = (1 << 8),
    BEV = (1 << 9),
    LIDARMULTITASK = (1 << 10),
    TRAJPRED = (1 << 11),
    DEPTH = (1 << 12),
    SEG3D = (1 << 13),
    MAP = (1 << 14),
    TRAJPRED_qcnet = (1 << 15)
  } type;

  friend std::ostream &operator<<(std::ostream &os, Perception &perception) {
    os << "[";
    if (perception.type == Perception::DET) {
      auto &detection = perception.det;
      for (int i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }

    } else if (perception.type == Perception::CLS) {
      auto &cls = perception.cls;
      for (int i = 0; i < cls.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << cls[i];
      }
    } else if (perception.type == Perception::SEG ||
               perception.type == Perception::SEG3D) {
      auto &seg = perception.seg;
      for (int i = 0; i < seg.seg.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << static_cast<int>(seg.seg[i]);
      }
    } else if (perception.type == Perception::MASK) {
      auto &detection = perception.mask.det_info;
      for (int i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }
    } else if (perception.type == Perception::POINT) {
      auto &points = perception.pt.point;
      for (int i = 0; i < points.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << points[i];
      }
    } else if (perception.type == Perception::DET3D) {
      auto &rlts = perception.det3d;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << rlts[i];
      }
    } else if (perception.type == Perception::KEYPOINT) {
      auto &kpts = perception.kpt.points;
      for (int i = 0; i < kpts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        for (int j = 0; j < kpts[i].size(); j++) {
          if (j != 0) {
            os << ",";
          }
          os << kpts[i][j].first << "," << kpts[i][j].second;
        }
      }
    } else if (perception.type == Perception::DETCAM3D) {
      auto &rlts = perception.detcam3d;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << rlts[i];
      }
    } else if (perception.type == Perception::LIDAR3D) {
      auto &rlts = perception.lidar3d;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << rlts[i];
      }
    } else if (perception.type == Perception::TRAJPRED) {
      auto &rlts = perception.trajPred;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        for (int j = 0; j < rlts[i].size(); j++) {
          os << rlts[i][j].single_bacth_name << ": " << rlts[i][j];
        }
      }
    } else if (perception.type == Perception::DEPTH) {
      auto &points = perception.pt.point;
      for (int i = 0; i < points.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << points[i];
      }
    } else if (perception.type == Perception::TRAJPRED_qcnet) {
      auto &rlts = perception.qcnet;
      for (int i = 0; i < rlts.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        for (int j = 0; j < rlts[i].size(); j++) {
          os << ": " << rlts[i][j];
        }
      }
    }
    os << "]";
    return os;
  }
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_BASE_PERCEPTION_COMMON_H_
