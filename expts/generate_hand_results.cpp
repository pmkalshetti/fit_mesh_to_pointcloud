#include "../hand_fitting/hand.h"
#include "../hand_fitting/mano.h"
#include "../fitting/fitting_functor.h"
#include "../utility/timer.h"
#include "../utility/print.h"
#include "../utility/json_io.h"
#include "../utility/visualizer.h"
#include "../sampling/graph_filter.h"

#include <Open3D/Open3D.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

using namespace Eigen;
using LineSet = open3d::geometry::LineSet;
using TriangleMesh = open3d::geometry::TriangleMesh;
using PointCloud = open3d::geometry::PointCloud;

std::vector<SurfacePoint> offset_sps(const std::vector<Eigen::Vector3i> &face_adjs, const std::vector<Eigen::Vector3d> &V, const SubdivEvaluator &evaluator, const std::vector<SurfacePoint> &sps)
{
    std::vector<SurfacePoint> sps_offset{sps};

    std::mt19937 random_generator(1);
    std::normal_distribution<double> dist_offset(0.0, 0.);
    for (auto &sp_offset : sps_offset)
    {
        Vector2d dU = {dist_offset(random_generator), dist_offset(random_generator)};
        FittingFunctor::update_surface_point(face_adjs, V, evaluator, sp_offset, dU);
    }

    return sps_offset;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: ./execeutable path/to/mano.bin path/to/params_gt.jsonc path/to/params_init.jsonc path/to/log/dir/plys_imgs_dir path/to/log/dir/fitting_iterations \n";
        std::exit(0);
    }

    std::shared_ptr<Visualizer> vis = std::make_shared<Visualizer>("Mesh Fitting");

    // load mano
    const std::filesystem::path path_mano{argv[1]};
    std::shared_ptr<Mano> mano = std::make_shared<Mano>(path_mano);
    std::shared_ptr<Hand> hand = std::make_shared<Hand>(mano);

    // groundtruth parameters
    VectorXd beta_gt = read_vec_from_json(argv[2], "beta");
    VectorXd theta_gt = read_vec_from_json(argv[2], "theta");
    theta_gt = theta_gt.array() * M_PI/180;
    VectorXd params_gt(beta_gt.size() + theta_gt.size());
    params_gt << beta_gt, theta_gt;

    // generate data
    constexpr int n_data{10000};
    const std::vector<SurfacePoint> sps_gt = SurfacePoint::generate(n_data, hand->n_triangles());
    const std::shared_ptr<std::vector<Vector3d>> points_observed = std::make_shared<std::vector<Vector3d>>(hand->generate_points_on_surface(params_gt, sps_gt));
    
    // const std::filesystem::path path_log = argv[4];
    // open3d::io::WritePointCloudToPLY(path_log/"plys/points_observed.ply", PointCloud(*points_observed));
    // vis->add_point_cloud(*points_observed, {1, 0, 0});
    // vis->save_image(path_log/"imgs/points_observed.png");
    // vis->run();
    // std::exit(0);
    
    // sample data
    constexpr int n_samples{200};
    std::vector<int> sampled_ids = sample_ids(n_samples, *points_observed, "high", 100, 100);
    std::vector<SurfacePoint> sps_gt_sampled(n_samples);
    std::shared_ptr<std::vector<Vector3d>> points_observed_sampled = std::make_shared<std::vector<Vector3d>>(n_samples);
    for (int i{0}; i < n_samples; ++i)
    {
        sps_gt_sampled[i] = sps_gt[sampled_ids[i]];
        points_observed_sampled->at(i) = points_observed->at(sampled_ids[i]);
    }
    
    // const std::filesystem::path path_log = argv[4];
    // open3d::io::WritePointCloudToPLY(path_log/"plys/points_observed_sampled.ply", PointCloud(*points_observed_sampled));
    vis->add_point_cloud(*points_observed_sampled, {1, 0, 0});
    // vis->save_image(path_log/"imgs/points_observed_sampled.png");
    // vis->run();
    // std::exit(0);

    // initial parameters
    VectorXd beta_init = read_vec_from_json(argv[3], "beta");
    VectorXd theta_init = read_vec_from_json(argv[3], "theta");
    theta_init = theta_init.array() * M_PI/180;
    VectorXd params_init(beta_init.size() + theta_init.size());
    params_init << beta_init, theta_init;

    // initial correspondences
    const std::vector<Vector3d> V_init = hand->M(params_init);
    const TriangleMesh V_mesh_init(V_init, hand->triangles());
    std::shared_ptr<LineSet> V_lineset_init = LineSet::CreateFromTriangleMesh(V_mesh_init);
    vis->add_lineset(V_lineset_init);
    std::vector<SurfacePoint> sps_init = offset_sps(hand->face_adjs(), V_init, hand->get_evaluator(), sps_gt_sampled);

    // const std::filesystem::path path_log = argv[4];
    // open3d::io::WriteTriangleMeshToPLY(path_log/"plys/mesh_init.ply", V_mesh_init, true, false, false, false, false, false);
    // vis->save_image(path_log/"imgs/mesh_init.png");
    // vis->run();
    // std::exit(0);


    // optimization
    FittingFunctor::InputType params(params_init, sps_init);
    FittingFunctor fitting_functor(points_observed_sampled, hand, vis, V_lineset_init, argv[5]);
    Eigen::LevenbergMarquardt<FittingFunctor> lm(fitting_functor);
    lm.setVerbose(true);
    lm.setMaxfev(50);
    utility::Timer timer;
    Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
    std::cout << "\nOptimization took " << timer.elapsed() << "s\n";

    // const std::filesystem::path path_log = argv[4];
    // const TriangleMesh V_mesh_fitted(hand->M(params.params), hand->triangles());
    // open3d::io::WriteTriangleMeshToPLY(path_log/"plys/mesh_fitted.ply", V_mesh_fitted, true, false, false, false, false, false);
    // vis->save_image(path_log/"imgs/mesh_fitted.png");
    // vis->run();
    // std::exit(0);

    std::shared_ptr<Visualizer> vis_surface = std::make_shared<Visualizer>("Mesh Surface");
    const std::filesystem::path path_log = argv[4];
    const TriangleMesh V_mesh_fitted(hand->M(params.params), hand->triangles());
    std::shared_ptr<TriangleMesh> V_mesh_surface = V_mesh_fitted.SubdivideLoop(3);
    vis_surface->add_mesh(V_mesh_surface);
    open3d::io::WriteTriangleMeshToPLY(path_log/"plys/mesh_surface.ply", *V_mesh_surface, true, false, false, false, false, false);
    vis_surface->save_image(path_log/"imgs/mesh_surface.png");
    vis_surface->run();
    std::exit(0);


    // log
    std::cout << "\nGroundtruth\n";
    print_vector(params_gt);

    std::cout << "\nInitial\n";
    print_vector(params_init);

    std::cout << "\nOptimized\n";
    print_vector(params.params);

    std::cout << "\nError: " << (params_gt - params.params).norm() << "\n";

    // hold visualizer
    std::cout << "\nPress `q` to close visualizer window\n";
    vis->run();

    return 0;
}