#pragma once

#include "../mesh/mesh.h"

#include <Eigen/Eigen>
#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>

#include <vector>


struct OSDVertex
{
    OSDVertex() {}

    void Clear(void * = 0)
    {
        point.setZero();
    }

    void AddWithWeight(const OSDVertex &src, float weight)
    {
        point += weight * src.point;
    }

    void SetPosition(float x, float y, float z)
    {
        point << x, y, z;
    }

    Eigen::Vector3d point;
};


class SurfacePoint
{
public:
    int face;
    Eigen::Vector2d U;

    static std::vector<SurfacePoint> generate(const int n_points, const int n_faces);
};


class SurfaceFeatures
{
public:
    SurfaceFeatures(int n_surface_points)
    : S(n_surface_points, Eigen::Vector3d::Zero()), // position of evaluated point
    N(n_surface_points, Eigen::Vector3d::Zero()),   // normal at evaluated point
    dSdu(n_surface_points, Eigen::Vector3d::Zero()), dSdv(n_surface_points, Eigen::Vector3d::Zero()),   // derivative wrt  u,v
    dSdV(max_n_weights * n_surface_points),    // derivative wrt vertices
    p_weights_vec(max_n_weights, 0), du_weights_vec(max_n_weights, 0), dv_weights_vec(max_n_weights, 0)
    {
    }

    void update_S_dSdU(Eigen::Vector3d& vert, int idx_pt, int idx_cv)
    {
        S[idx_pt] += vert * p_weights_vec[idx_cv];
        dSdu[idx_pt] += vert * du_weights_vec[idx_cv];
        dSdv[idx_pt] += vert * dv_weights_vec[idx_cv];
    }

    void update_dSdV(const int id_pt, const int id_cv, const double value)
    {
        
        dSdV.add(id_pt, id_cv, value);
    }

    void compute_normal()
    {
        for (int i{0}; i < dSdu.size(); ++i)
        {
            N[i] = dSdu[i].cross(dSdv[i]);
        }
    }

    const std::vector<Eigen::Vector3d>& get_S() const { return S; }
    const std::vector<Eigen::Vector3d>& get_dSdu() const { return dSdu; }
    const std::vector<Eigen::Vector3d>& get_dSdv() const { return dSdv; }
    const Eigen::SparseMatrix<double> get_dSdV(const int n_verts) const
    { 
        Eigen::SparseMatrix<double> dSdV_mat(S.size(), n_verts);
        dSdV_mat.setFromTriplets(dSdV.begin(), dSdV.end());

        return dSdV_mat;
    }

    float* p_weights() { return &(p_weights_vec[0]); }
    float* du_weights() { return &du_weights_vec[0]; }
    float* dv_weights() { return &dv_weights_vec[0]; }

private:

    // evaluated point
    std::vector<Eigen::Vector3d> S; // position
    std::vector<Eigen::Vector3d> N; // normal

    // derivative of surface function wrt correspondences
    std::vector<Eigen::Vector3d> dSdu;
    std::vector<Eigen::Vector3d> dSdv;
    
    // derivative of surface function wrt control vertices
    Eigen::TripletArray<double> dSdV;

    // weights
    static constexpr int max_n_weights{16};    // 16 since using ENDCAP_BSPLINE_BASIS
    std::vector<float> p_weights_vec;   // position
    std::vector<float> du_weights_vec;  // derivative wrt u
    std::vector<float> dv_weights_vec;  // derivative wrt v
};


class SubdivEvaluator
{
public:
    SubdivEvaluator(const int n_vertices, const std::vector<Eigen::Vector3i> &triangles, const bool has_boundary);

    SurfaceFeatures compute_surface_features(const std::vector<Eigen::Vector3d> &V, const std::vector<SurfacePoint> &sps, const bool compute_dSdV) const;
    Mesh generate_refined_mesh(const std::vector<Eigen::Vector3d> &V, const int level) const;
    
    ~SubdivEvaluator()
    {
        delete patch_table;
        delete refiner;
    }

private:
    int n_refiner_vertices;
    OpenSubdiv::Far::TopologyRefiner *refiner;

    int n_local_points;
    OpenSubdiv::Far::PatchTable *patch_table;

// public:
//     int n_vertices;
//     int n_refiner_vertices;

//     mutable std::vector<OSDVertex> evaluation_verts_buffer;
//     static const int maxlevel{3};
//     OpenSubdiv::Far::TopologyRefiner *refiner;

//     size_t n_local_points;
//     OpenSubdiv::Far::PatchTable *patch_table;

//     SubdivEvaluator(const Mesh &mesh);
//     SubdivEvaluator(const SubdivEvaluator &that) { *this = that; }
//     SubdivEvaluator &operator=(const SubdivEvaluator &that)
//     {
//         n_vertices = that.n_vertices;
//         n_refiner_vertices = that.n_refiner_vertices;
//         evaluation_verts_buffer = that.evaluation_verts_buffer;
//         n_local_points = that.n_local_points;
//         patch_table = new OpenSubdiv::Far::PatchTable(*that.patch_table);

//         return *this;
//     }

//     std::shared_ptr<Mesh> generate_refined_mesh(const std::vector<Eigen::Vector3d> &verts_in, int level);
//     void evaluate_subdiv_surface(const std::vector<Eigen::Vector3d> &coarse_verts, const std::vector<SurfacePoint> &uv, SurfaceFeatures &sf, const bool compute_dX = true) const;

    
};

