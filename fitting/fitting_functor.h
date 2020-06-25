#pragma once

#include "../subdiv/evaluator.h"
#include "../model/model_base.h"
#include "../utility/visualizer.h"

// required by LevenbergMarquardt file
#include <iostream>
using Scalar = double;

#include <Eigen/Eigen>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/src/SparseExtra/BlockSparseQR.h>
#include <unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h>
#include <Open3D/Open3D.h>

#include <filesystem>

struct FittingFunctor : Eigen::SparseFunctor<double>
{
    using LineSet = open3d::geometry::LineSet;

public:
    // optimization variables
    struct InputType
    {
        Eigen::VectorXd params;
        std::vector<SurfacePoint> sps;

        InputType() {}

        InputType(const Eigen::VectorXd &params, const std::vector<SurfacePoint> &sps) 
        : params{params}, sps{sps}
        {
        }
    };

    FittingFunctor(const std::shared_ptr<std::vector<Eigen::Vector3d>> &X, const std::shared_ptr<ModelBase> &model, std::shared_ptr<Visualizer> &vis, std::shared_ptr<LineSet> &V_lineset, const std::filesystem::path dir_log = "") 
    : SparseFunctor<double>(
        model->n_params() + X->size()*2,  // total parameters
        X->size()*3                  // number of residuals
    ), 
    X{X}, model{model}, sf(X->size()), vis{vis}, V_lineset{V_lineset}, dir_log{dir_log}, iteration{0}
    {
        if (!dir_log.empty())
        {
            if (std::filesystem::exists(dir_log)) std::filesystem::remove_all(dir_log);
            std::filesystem::create_directories(dir_log);
        }
    }

    static int update_surface_point(const std::vector<Eigen::Vector3i> &face_adjs, const std::vector<Eigen::Vector3d> &V, const SubdivEvaluator &evaluator, SurfacePoint &sp, const Eigen::Vector2d &dU);

private:
    std::shared_ptr<ModelBase> model;                   // model
    std::shared_ptr<std::vector<Eigen::Vector3d>> X;    // data
    std::vector<Eigen::Vector3d> V;                     // vertices at current iteration
    SurfaceFeatures sf;                                 // surface features at current iteration
    std::shared_ptr<Visualizer> vis;
    std::shared_ptr<open3d::geometry::LineSet> V_lineset;
    int iteration;
    std::filesystem::path dir_log;

public:
    // functor functions
    int operator()(const InputType &x, ValueType &fvec);
    int df(const InputType &x, JacobianType &fjac);
    void increment_in_place(InputType *x, const StepType &p);
    double estimateNorm(const InputType &x, const StepType &diag);

    using DenseQRSolver3x2 = Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 3, 2>>;
    using LeftSuperBlockSolver = Eigen::BlockDiagonalSparseQR<JacobianType, DenseQRSolver3x2>;
    using RightSuperBlockSolver = Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;
    using SchurLikeQRSolver = Eigen::BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver>;
    using QRSolver = SchurLikeQRSolver;
    void initQRSolver(QRSolver &qr)
    {
        qr.setBlockParams(X->size() * 2);
        qr.getLeftSolver().setSparseBlockParams(3, 2);
    }

};