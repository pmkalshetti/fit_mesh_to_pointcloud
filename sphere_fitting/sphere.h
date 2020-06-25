#pragma once

#include "../model/model_base.h"
#include "../subdiv/evaluator.h"
#include "../mesh/mesh.h"

#include <Open3D/Open3D.h>
#include <Eigen/Eigen>

class Sphere : public ModelBase
{
    using TriangleMesh = open3d::geometry::TriangleMesh;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    using Translation = Eigen::Translation<double, 3>;

public:
    Sphere(const std::shared_ptr<TriangleMesh> &trimesh) : trimesh{trimesh}, evaluator(trimesh->vertices_.size(), trimesh->triangles_, true), mesh{trimesh->vertices_, trimesh->triangles_}
    {
    }

    virtual const int n_verts() const override { return trimesh->vertices_.size(); }
    virtual const int n_triangles() const override { return trimesh->triangles_.size(); }
    virtual const std::vector<Eigen::Vector3i>& triangles() const override { return trimesh->triangles_; };
    virtual const std::vector<Eigen::Vector3i>& face_adjs() const override { return mesh.get_face_adjs(); };
    virtual const int n_params() const override { return 3+3; }

    virtual const std::vector<Eigen::Vector3d> M(const Eigen::VectorXd &params) const override
    {
        Transform T;
        T.linear().setIdentity();

        // set scale
        for (int i{0}; i < 3; ++i)
            T(i, i) = params(i);
        
        // set translation
        T.translation() = params.tail(3);

        // apply transformation
        std::vector<Eigen::Vector3d> V(trimesh->vertices_.size());
        for (int id_vert{0}; id_vert < V.size(); ++id_vert)
        {
            V[id_vert] = T * trimesh->vertices_[id_vert];
        }

        return V;
    }

    virtual const std::vector<Eigen::VectorXd> compute_dVdp(const Eigen::VectorXd &params) const override
    {
        const std::vector<Eigen::Vector3d> V = M(params);

        std::vector<Eigen::VectorXd> dVdp(n_verts()*3, Eigen::VectorXd::Zero(params.size()));
        for (int id_vert{0}; id_vert < n_verts(); ++id_vert)
        {
            // wrt scale
            for (int id_scale{0}; id_scale < 3; ++id_scale)
            {
                for (int id_coord{0}; id_coord < 3; ++id_coord)
                {   
                    // scales act independently across axes
                    if (id_scale == id_coord)
                    {
                        dVdp[id_vert*3 + id_coord](id_scale) = V[id_vert](id_coord);
                    }
                }
            }

            // wrt translation
            for (int id_translation{0}; id_translation < 3; ++id_translation)
            {
                for (int id_coord{0}; id_coord < 3; ++id_coord)
                {
                    if (id_translation == id_coord)
                    {
                        dVdp[id_vert*3 + id_coord](3 + id_translation) = 1;
                    }
                }
            }
        }

        return dVdp;
    }

    virtual const SurfaceFeatures S(const std::vector<SurfacePoint> &sps, const std::vector<Eigen::Vector3d> &V) const override { return evaluator.compute_surface_features(V, sps, true); };
    virtual const SubdivEvaluator& get_evaluator() const override { return evaluator; };

    virtual const std::vector<Eigen::Vector3d> generate_points_on_surface(const Eigen::VectorXd &params, const std::vector<SurfacePoint> &sps) const override
    {
        std::vector<Eigen::Vector3d> V = M(params);
        SurfaceFeatures sf = S(sps, V);
        return sf.get_S();
    }

private:
    std::shared_ptr<TriangleMesh> trimesh;
    Mesh mesh;
    SubdivEvaluator evaluator;

};