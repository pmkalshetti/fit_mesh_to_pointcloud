#pragma once

#include "../model/model_base.h"

#include <Eigen/Eigen>
#include <Open3D/Open3D.h>

#include <filesystem>
#include <vector>
#include <fstream>

class Mano
{
public:
    using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixX3dR = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Matrix3dR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    using TriangleMesh = open3d::geometry::TriangleMesh;
    
    Mano(const std::filesystem::path &path) : trimesh{std::make_shared<TriangleMesh>()} { read_bin(path); }
    const int n_vertices() const { return trimesh->vertices_.size(); }
    const int n_triangles() const { return trimesh->triangles_.size(); }
    const std::vector<Eigen::Vector3d>& vertices() const { return trimesh->vertices_; }
    const std::vector<Eigen::Vector3i>& triangles() const { return trimesh->triangles_; }
    const int n_joints() const { return m_J.rows(); }

    const std::vector<Eigen::Vector3d> W(const Eigen::VectorXd &beta, const Eigen::VectorXd &theta) const;
    const std::vector<Eigen::VectorXd> compute_dVdp(const Eigen::VectorXd &beta, const Eigen::VectorXd &theta) const;

private:
    const std::vector<Transform> compute_Ts(const MatrixXdR &J) const;
    const std::vector<Transform> compute_Hs(const std::vector<Transform> &local_to_parent) const;
    const std::vector<Transform> compute_Rs(const Eigen::VectorXd &theta) const;
    const std::vector<Transform> compute_Gs(const std::vector<Transform> &local_to_parent, const std::vector<Transform> &local_rotation) const;
    const std::vector<Eigen::Vector3d> lbs(const std::vector<Transform> &local_to_world, const std::vector<Transform> &local_rest_to_world, const MatrixX3dR &T_P) const;

    std::vector<std::vector<Matrix3dR>> compute_dR_dts(const Eigen::VectorXd &theta) const;
    std::vector<std::vector<Transform>> compute_dG_dts(const Eigen::VectorXd &theta, const std::vector<std::vector<Matrix3dR>> &dR_dts, const std::vector<Transform> &Ts, const std::vector<Transform> &Rs, const std::vector<Transform> &Gs) const;
    std::vector<std::vector<Eigen::Vector3d>> compute_dvp_dts(const Eigen::VectorXd &theta, const std::vector<std::vector<Matrix3dR>> &dR_dts) const;
    

    const Eigen::Map<MatrixX3dR> T() const { return Eigen::Map<MatrixX3dR>(&trimesh->vertices_[0](0), trimesh->vertices_.size(), 3); }
    const MatrixX3dR B_S(const Eigen::VectorXd &beta) const;
    const MatrixX3dR B_P(const std::vector<Transform> &local_rotations) const;

    void read_bin(const std::filesystem::path &path);
    void read_trimesh(std::ifstream &file);
    void read_S(std::ifstream &file);
    void read_P(std::ifstream &file);
    void read_J(std::ifstream &file);
    void read_W(std::ifstream &file);
    void read_A(std::ifstream &file);
    void read_pose_pca_basis(std::ifstream &file);
    void read_pose_pca_mean(std::ifstream &file);

private:
    std::shared_ptr<TriangleMesh> trimesh;
    MatrixXdR m_S;                   // (778*3, 10)
    MatrixXdR m_P;                   // (778*3, 135)
    Eigen::SparseMatrix<double> m_J; // (16, 778)
    MatrixXdR m_W;                   // (778, 16)
    std::vector<int> m_A;            // (16,)
    MatrixXdR pose_pca_basis;        // (45, 45)
    Eigen::VectorXd pose_pca_mean;   // (45,)

};