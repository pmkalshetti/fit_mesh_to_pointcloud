#pragma once

#include <Open3D/Open3D.h>
#include <Eigen/Eigen>

#include <vector>

class Mesh
{
public:
    Mesh(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3i> &triangles)
    : tri_mesh(std::make_shared<open3d::geometry::TriangleMesh>(vertices, triangles))
    {
        update_adjacencies();
    }

    void update_adjacencies();

    const int n_vertices() const { return tri_mesh->vertices_.size(); }
    const int n_triangles() const { return tri_mesh->triangles_.size(); }
    const std::vector<Eigen::Vector3d> &vertices() const { return tri_mesh->vertices_; }
    const std::vector<Eigen::Vector3i> &triangles() const { return tri_mesh->triangles_; }
    const std::vector<Eigen::Vector3i> &get_face_adjs() const { return face_adjs; }
    int adj_face(const int id_face, const int id_side) const { return face_adjs[id_face](id_side); }

private:
    std::shared_ptr<open3d::geometry::TriangleMesh> tri_mesh;
    std::vector<Eigen::Vector3i> face_adjs;
};