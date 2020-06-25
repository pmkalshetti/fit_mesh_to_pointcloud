#pragma once

#include <Open3D/Open3D.h>
#include <Eigen/Eigen>

#include <vector>


class Visualizer
{
public:
    Visualizer(const std::string &window_name="Open3D", int width=1920, int height=1080)
    {
        vis.CreateVisualizerWindow(window_name, width, height);
        vis.GetRenderOption().ToggleMeshShowWireframe();
    }

    void add_point_cloud(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &color = {0, 0, 0})
    {
        open3d::geometry::PointCloud pcd(points);
        pcd.PaintUniformColor(color);
        vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pcd));
    }

    void add_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, const Eigen::Vector3d &color = {0, 0, 0})
    {
        pcd->PaintUniformColor(color);
        vis.AddGeometry(pcd);
    }

    void add_mesh(const std::shared_ptr<open3d::geometry::TriangleMesh> &tri_mesh)
    {
        tri_mesh->ComputeVertexNormals();
        vis.AddGeometry(tri_mesh);
    }

    void add_lineset(const std::shared_ptr<open3d::geometry::LineSet> &lineset)
    {
        vis.AddGeometry(lineset);
    }

    void update(std::shared_ptr<open3d::geometry::TriangleMesh> &tri_mesh)
    {
        tri_mesh->ComputeTriangleNormals();
        vis.UpdateGeometry(tri_mesh);
    }

    void update(std::shared_ptr<open3d::geometry::LineSet> &lineset)
    {
        vis.UpdateGeometry(lineset);
    }

    bool show()
    {
        return vis.PollEvents();
    }

    void save_image(const std::string& filename)
    {
        vis.CaptureScreenImage(filename);
    }

    void run()
    {
        vis.Run();
        vis.DestroyVisualizerWindow();
    }

private:
    open3d::visualization::Visualizer vis;
};