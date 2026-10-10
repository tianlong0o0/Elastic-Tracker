#pragma once

#include <mapping/mapping.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <vector>
#include <random>

namespace prediction {

struct Primitive {
  std::vector<Eigen::Vector3d> pts;
  double cost;
  bool valid;
};
class Predict {
private:
  double dt;
  double pre_dur;
  double rho_a;
  double vmax;
  int primitive_num;
  mapping::OccGridMap map;

  inline bool isValid(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return (v.norm() < vmax) && (!map.isOccupied(p));
  }

  Primitive generatePrimitive(const Eigen::Vector3d& p0,
                              const Eigen::Vector3d& v0,
                              const Eigen::Vector3d& pf) {
    Primitive prim;
    prim.cost = 0;
    prim.valid = true;
    double T = pre_dur;

    // cubic Bernstein
    Eigen::Vector3d c0 = p0;
    Eigen::Vector3d c1 = p0 + v0 * T / 3.0;
    Eigen::Vector3d c2 = 0.5 * p0 + 0.5 * pf + v0 * T / 6.0;
    Eigen::Vector3d c3 = pf;
    int N = static_cast<int>(pre_dur / dt);

    for (int i=0;i<=N;++i) {
      double s = static_cast<double>(i) / N;
      double b0 = pow(1-s, 3);
      double b1 = 3 * s * pow(1-s, 2);
      double b2 = 3 * (1-s) * pow(s, 3);
      double b3 = pow(s, 3);
      Eigen::Vector3d p = b0 * c0 + b1 * c1 + b2 * c2 + b3 * c3;

      if (!isValid(p, v0)) {
        prim.valid = false;
        return prim;
      }

      prim.pts.push_back(p);
      prim.cost += (p-pf).squaredNorm();
    }

    return prim;
  }

public:
  // Predict(ros::NodeHandle& nh) {
  //   nh.getParam("tracking_dur", pre_dur);
  //   nh.getParam("tracking_dt", dt);
  //   nh.getParam("prediction/rho_a", rho_a);
  //   nh.getParam("prediction/vmax", vmax);
  //   primitive_num = 100;
  // }

  inline void setMap(const mapping::OccGridMap& _map) {
    map = _map;
    // map.inflate_last();
  }

  bool predict(const Eigen::Vector3d& target_p,
               const Eigen::Vector3d& target_v,
               std::vector<Eigen::Vector3d>& target_predict,
               const double& max_time = 0.1) {
    std::vector<Primitive> candidates;
    ros::Time t_start = ros::Time::now();
    
    // Sample(Nearly Constant Velocity Model)
    std::default_random_engine gen;
    std::normal_distribution<double> noise(0, 0.5);
    
    for (int i=0;i<primitive_num;++i) {
      double t_cost = (ros::Time::now() - t_start).toSec();
      if (t_cost > max_time) {
        std::cout << "[prediction] too slow!" << std::endl;
        return false;
      }
      
      Eigen::Vector3d end = target_p + target_v * pre_dur;
      end.x() += noise(gen);
      end.y() += noise(gen);
      end.z() += noise(gen);

      auto prim = generatePrimitive(target_p, target_v, end);
      if (prim.valid) candidates.push_back(prim);
    }

    if (candidates.empty()) return false;

    // Selection
    double best = 1e9;
    int index = -1;

    for (int i=0;i<candidates.size();i++) {
      double t_cost = (ros::Time::now() - t_start).toSec();
      if (t_cost > max_time) {
        std::cout << "[prediction] too slow!" << std::endl;
        return false;
      }
      
      if (candidates[i].cost < best) {
        best = candidates[i].cost;
        index = i;
      }
    }

    target_predict = candidates[index].pts;
    return true;
  }
};

}
