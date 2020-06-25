#pragma once

#include "../subdiv/evaluator.h"

#include <Eigen/Eigen>

#include <vector>

class ModelBase
{
public:
    ModelBase() {}

    virtual const int n_verts() const = 0;
    virtual const int n_triangles() const = 0;
    virtual const std::vector<Eigen::Vector3i>& triangles() const = 0;
    virtual const std::vector<Eigen::Vector3i>& face_adjs() const = 0;
    virtual const int n_params() const = 0;
    virtual const std::vector<Eigen::Vector3d> M(const Eigen::VectorXd &params) const = 0;
    virtual const SurfaceFeatures S(const std::vector<SurfacePoint> &sps, const std::vector<Eigen::Vector3d> &V) const = 0;
    virtual const std::vector<Eigen::VectorXd> compute_dVdp(const Eigen::VectorXd &params) const = 0;
    // virtual void compute_dVdp() = 0;    // sets dVdp
    virtual const SubdivEvaluator& get_evaluator() const = 0;

    virtual const std::vector<Eigen::Vector3d> generate_points_on_surface(const Eigen::VectorXd &params, const std::vector<SurfacePoint> &sps) const = 0;
    
    // const SurfaceFeatures S(const std::vector<SurfacePoint> &sps, const std::vector<Eigen::Vector3d> &V) { return evaluator.compute_surface_features(V, sps, true); } 
    // const std::vector<Eigen::Vector3i>& face_adjs() const { return mesh.get_face_adjs(); }
    // const std::vector<Eigen::VectorXd>& get_dVdp() const { return dVdp; }
    // const SubdivEvaluator& get_evaluator() const { return evaluator; }

// private:
    // Mesh mesh;                          // template
    // std::vector<Eigen::VectorXd> dVdp;
    // SubdivEvaluator evaluator;
};