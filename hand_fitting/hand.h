#pragma once

#include "mano.h"
#include "../model/model_base.h"
#include "../subdiv/evaluator.h"
#include "../mesh/mesh.h"

class Hand : public ModelBase
{
public:
    Hand(const std::shared_ptr<Mano> &mano) : mano{mano}, evaluator{mano->n_vertices(), mano->triangles(), true}, mesh{mano->vertices(), mano->triangles()}
    {
    }

    virtual const int n_verts() const override { return mano->n_vertices(); }
    virtual const int n_triangles() const override { return mano->n_triangles(); }
    virtual const std::vector<Eigen::Vector3i>& triangles() const override { return mano->triangles(); }
    virtual const std::vector<Eigen::Vector3i>& face_adjs() const override { return mesh.get_face_adjs(); }
    virtual const int n_params() const override { return 10+23; }
    virtual const std::vector<Eigen::Vector3d> M(const Eigen::VectorXd &params) const override { return mano->W(params.head(10), params.tail(23)); }
    virtual const std::vector<Eigen::VectorXd> compute_dVdp(const Eigen::VectorXd &params) const override { return mano->compute_dVdp(params.head(10), params.tail(23)); }

    virtual const SurfaceFeatures S(const std::vector<SurfacePoint> &sps, const std::vector<Eigen::Vector3d> &V) const override { return evaluator.compute_surface_features(V, sps, true); }
    virtual const SubdivEvaluator& get_evaluator() const override { return evaluator; }

    virtual const std::vector<Eigen::Vector3d> generate_points_on_surface(const Eigen::VectorXd &params, const std::vector<SurfacePoint> &sps) const override
    {
        std::vector<Eigen::Vector3d> V = M(params);
        SurfaceFeatures sf = S(sps, V);
        return sf.get_S();
    }

private:
    std::shared_ptr<Mano> mano;
    Mesh mesh;
    SubdivEvaluator evaluator;
};