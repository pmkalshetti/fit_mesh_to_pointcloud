#include "mano.h"
#include "../utility/rotation.h"

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixX3dR = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Matrix3dR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Matrix4d = Eigen::Matrix4d;
using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;
using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
using Translation = Eigen::Translation<double, 3>;
using TriangleMesh = open3d::geometry::TriangleMesh;


const std::vector<Vector3d> Mano::W(const VectorXd &beta, const VectorXd &theta) const
{
    MatrixX3dR T_S = T() + B_S(beta);
    MatrixXdR J = m_J * T_S;
    std::vector<Transform> Ts = compute_Ts(J);
    std::vector<Transform> Hs = compute_Hs(Ts);
    std::vector<Transform> Rs = compute_Rs(theta);
    std::vector<Transform> Gs = compute_Gs(Ts, Rs);
    MatrixX3dR T_P = T_S + B_P(Rs);
    std::vector<Vector3d> T_prime = lbs(Gs, Hs, T_P);

    return T_prime;
}

const std::vector<Eigen::VectorXd> Mano::compute_dVdp(const VectorXd &beta, const VectorXd &theta) const
{
    MatrixX3dR T_S = T() + B_S(beta);
    MatrixXdR J = m_J * T_S;
    std::vector<Transform> Ts = compute_Ts(J);
    std::vector<Transform> Hs = compute_Hs(Ts);
    std::vector<Transform> Rs = compute_Rs(theta);
    std::vector<Transform> Gs = compute_Gs(Ts, Rs);
    MatrixX3dR T_P = T_S + B_P(Rs);

    // precompute H^-1
    std::vector<Transform> Hs_inv(Hs.size());
    for (int id_joint{0}; id_joint < n_joints(); ++id_joint)
    {
        Hs_inv[id_joint] = Hs[id_joint].inverse();
    }

    // intermediate derivatives using chain rule
    std::vector<std::vector<Matrix3dR>> dR_dts = compute_dR_dts(theta);
    std::vector<std::vector<Transform>> dG_dts = compute_dG_dts(theta, dR_dts, Ts, Rs, Gs);
    std::vector<std::vector<Vector3d>> dvp_dts = compute_dvp_dts(theta, dR_dts);

    std::vector<VectorXd> dVdp(n_vertices()*3, VectorXd::Zero(beta.size()+theta.size()));
    for (int id_vert{0}; id_vert < n_vertices(); ++id_vert)
    {      
        // wrt beta
        for (int id_beta{0}; id_beta < 0; ++id_beta)
        {
            // derivative of vertex wrt each beta
            Vector3d dVi_dbm = Vector3d::Zero();

            // TODO: use terms from skinning
            dVi_dbm = m_S.col(id_beta).segment<3>(id_vert*3);

            for (int id_coord{0}; id_coord < 3; ++id_coord)
            {
                dVdp[id_vert*3 + id_coord](id_beta) = dVi_dbm(id_coord);
            }
        }

        // wrt theta
        for  (int id_theta{0}; id_theta < theta.size(); ++id_theta)
        {
            // derivative of vertex wrt each theta
            Vector3d dVi_dtp = Vector3d::Zero();
            for (int id_joint{0}; id_joint < n_joints(); ++id_joint)
            {
                dVi_dtp += m_W(id_vert, id_joint) * (dG_dts[id_theta][id_joint]*Hs_inv[id_joint]*T_P.row(id_vert).transpose() + Gs[id_joint]*Hs_inv[id_joint]*dvp_dts[id_theta][id_vert]);
            }

            for (int id_coord{0}; id_coord < 3; ++id_coord)
            {
                dVdp[id_vert*3 + id_coord](beta.size() + id_theta) = dVi_dtp(id_coord);
            }
        }
    
    }

    return dVdp;
}


const MatrixX3dR Mano::B_S(const VectorXd &beta) const
{
    MatrixX3dR b_s(n_vertices(), 3);
    Eigen::Map<VectorXd> b_s_vec(b_s.data(), n_vertices() * 3);
    b_s_vec = m_S * beta;
    return b_s;
}

const MatrixX3dR Mano::B_P(const std::vector<Transform> &local_rotation) const
{
    // (16, 3,3) -> 15*3*3 (skip global and flatten)
    std::vector<Matrix3dR> R_diff(local_rotation.size() - 1);
    MatrixXdR I = MatrixXdR::Identity(3, 3);
    for (int id_joint{0}; id_joint < R_diff.size(); ++id_joint)
    {
        R_diff[id_joint] = local_rotation[id_joint + 1].linear() - I;
    }

    // flat view
    Eigen::Map<VectorXd> rot_mats_flattened(&R_diff[0](0, 0), 15 * 3 * 3);

    MatrixX3dR b_p(n_vertices(), 3);
    Eigen::Map<VectorXd> b_p_vec(b_p.data(), n_vertices() * 3);
    b_p_vec = m_P * rot_mats_flattened;
    return b_p;
}

const std::vector<Transform> Mano::compute_Ts(const MatrixXdR &J) const
{
    // container for local to parent transforms
    std::vector<Transform> local_to_parent;
    local_to_parent.reserve(J.rows());

    // root joint to MANO origin
    Transform T_o_root(Translation(J.row(0)));
    local_to_parent.push_back(T_o_root);

    for (int id_finger{0}; id_finger < 5; ++id_finger)
    {
        int id_joint_base = 3 * id_finger + 1;

        // finger base (f0) to root
        Vector3d v_f0_f1 = (J.row(id_joint_base + 1) - J.row(id_joint_base)).normalized(); // vector from joint f0 to joint f1
        double a_f0 = atan2(v_f0_f1(2), -v_f0_f1(0));                                      // angle between v_f0_f1 and -X axis
        Eigen::Matrix3d R_root_f0 = utility::RotMatY(M_PI + a_f0);
        if (id_finger == 4)
            R_root_f0 *= utility::RotMatX(-60 * M_PI / 180);
        Transform T_root_f0;
        T_root_f0.linear() = R_root_f0;
        T_root_f0.translation() = J.row(id_joint_base) - J.row(0);
        local_to_parent.push_back(T_root_f0);

        // finger middle (f1) to f0
        Transform T_f0_f1;
        T_f0_f1.linear().setIdentity();
        T_f0_f1.translation() = R_root_f0.transpose() * (J.row(id_joint_base + 1) - J.row(id_joint_base)).transpose();
        local_to_parent.push_back(T_f0_f1);

        // finger distal (f2) to f1
        Transform T_f1_f2;
        T_f1_f2.linear().setIdentity();
        T_f1_f2.translation() = R_root_f0.transpose() * (J.row(id_joint_base + 2) - J.row(id_joint_base + 1)).transpose();
        local_to_parent.push_back(T_f1_f2);
    }

    return local_to_parent;
}

const std::vector<Transform> Mano::compute_Hs(const std::vector<Transform> &local_to_parent) const
{
    std::vector<Transform> local_rest_to_world;
    local_rest_to_world.reserve(n_joints());

    local_rest_to_world.push_back(local_to_parent[0]);
    for (int id_joint{1}; id_joint < n_joints(); ++id_joint)
    {
        local_rest_to_world.push_back(local_rest_to_world[m_A[id_joint]] * local_to_parent[id_joint]);
    }
    return local_rest_to_world;
}

const std::vector<Transform> Mano::compute_Rs(const VectorXd &theta) const
{
    using namespace utility;

    std::vector<Transform> local_rotation;
    local_rotation.reserve(n_joints());

    // global orientation
    local_rotation.push_back(Transform(RotMatXYZ(theta.segment<3>(0))));

    for (int id_finger{0}; id_finger < 5; ++id_finger)
    {
        Eigen::Map<VectorXd> theta_finger(const_cast<double *>(theta.data()) + 3 + 4 * id_finger, 4); // 4 dof for each finger (+3 is for global)
        local_rotation.push_back(Transform(RotMatYZ(theta_finger.segment<2>(0))));
        local_rotation.push_back(Transform(RotMatZ(theta_finger(2))));
        local_rotation.push_back(Transform(RotMatZ(theta_finger(3))));
    }
    return local_rotation;
}

const std::vector<Transform> Mano::compute_Gs(const std::vector<Transform> &local_to_parent, const std::vector<Transform> &local_rotation) const
{
    std::vector<Transform> local_to_world;
    local_to_world.reserve(n_joints());

    local_to_world.push_back(Transform(local_rotation[0]) * local_to_parent[0]);

    for (int id_joint{1}; id_joint < n_joints(); ++id_joint)
    {
        local_to_world.push_back(local_to_world[m_A[id_joint]] * local_to_parent[id_joint] * Transform(local_rotation[id_joint]));
    }
    return local_to_world;
}

const std::vector<Vector3d> Mano::lbs(const std::vector<Transform> &local_to_world, const std::vector<Transform> &local_rest_to_world, const MatrixX3dR &T_P) const
{
    // inverse transform
    std::vector<Transform> deform_wrt_rest(local_rest_to_world.size());
    for (int i{0}; i < n_joints(); ++i)
        deform_wrt_rest[i] = local_to_world[i] * local_rest_to_world[i].inverse();

    std::vector<Vector3d> deformed_verts(n_vertices(), Vector3d::Zero());
    for (int id_vertex{0}; id_vertex < n_vertices(); ++id_vertex)
    {
        for (int id_joint{0}; id_joint < n_joints(); ++id_joint)
        {
            deformed_verts[id_vertex] += m_W(id_vertex, id_joint) * (deform_wrt_rest[id_joint] * T_P.row(id_vertex).transpose());
        }
    }

    return deformed_verts;
}


std::vector<std::vector<Matrix3dR>> Mano::compute_dR_dts(const VectorXd &theta) const
{
    // init
    std::vector<std::vector<Matrix3dR>> dR_dts(theta.size());
    for (auto &dR_dt : dR_dts)
        dR_dt.resize(n_joints(), Matrix3dR::Zero());

    // wrt theta 0,1,2
    dR_dts[0][0] = utility::dRotMatXYZ_ax(theta.segment<3>(0));
    dR_dts[1][0] = utility::dRotMatXYZ_ay(theta.segment<3>(0));
    dR_dts[2][0] = utility::dRotMatXYZ_az(theta.segment<3>(0));

    // wrt rest thetas
    for (int id_finger{0}; id_finger < 5; ++id_finger)
    {
        const int id_theta_finger = 3 + 4 * id_finger;                                                // 3 for root, 4 dof for each finger
        const int id_joint_finger = 1 + 3 * id_finger;                                                // 1 for root, 3 joints per finger
        Eigen::Map<VectorXd> theta_finger(const_cast<double *>(theta.data()) + 3 + 4 * id_finger, 4); // 4 dof for each finger (+3 is for global)

        dR_dts[id_theta_finger][id_joint_finger] = utility::dRotMatYZ_ay(theta_finger.segment<2>(0));
        dR_dts[id_theta_finger + 1][id_joint_finger] = utility::dRotMatYZ_az(theta_finger.segment<2>(0));
        dR_dts[id_theta_finger + 2][id_joint_finger + 1] = utility::dRotMatZ(theta_finger(2));
        dR_dts[id_theta_finger + 3][id_joint_finger + 2] = utility::dRotMatZ(theta_finger(3));
    }

    return dR_dts;
}

std::vector<std::vector<Transform>> Mano::compute_dG_dts(const VectorXd &theta, const std::vector<std::vector<Matrix3dR>> &dR_dts, const std::vector<Transform> &Ts, const std::vector<Transform> &Rs, const std::vector<Transform> &Gs) const
{
    // init
    std::vector<std::vector<Transform>> dG_dts(theta.size());
    for (auto &dG_dt : dG_dts)
        dG_dt.resize(n_joints(), Transform(Matrix4d::Zero()));

    // wrt theta 0,1,2
    dG_dts[0][0] = Ts[0] * Transform(dR_dts[0][0]);
    dG_dts[1][0] = Ts[0] * Transform(dR_dts[1][0]);
    dG_dts[2][0] = Ts[0] * Transform(dR_dts[2][0]);
    for (int id_joint{1}; id_joint < n_joints(); ++id_joint)
    {
        dG_dts[0][id_joint] = dG_dts[0][m_A[id_joint]] * Ts[id_joint] * Rs[id_joint];
        dG_dts[1][id_joint] = dG_dts[1][m_A[id_joint]] * Ts[id_joint] * Rs[id_joint];
        dG_dts[2][id_joint] = dG_dts[2][m_A[id_joint]] * Ts[id_joint] * Rs[id_joint];
    }

    // wrt other thetas based on finger
    for (int id_finger{0}; id_finger < 5; ++id_finger)
    {
        const int id_theta_finger = 3 + 4 * id_finger;                                                // 3 for root, 4 dof for each finger
        const int id_joint_finger = 1 + 3 * id_finger;                                                // 1 for root, 3 joints per finger
        Eigen::Map<VectorXd> theta_finger(const_cast<double *>(theta.data()) + 3 + 4 * id_finger, 4); // 4 dof for each finger (+3 is for global)

        // wrt theta y at finger base
        dG_dts[id_theta_finger][id_joint_finger] = Gs[m_A[id_joint_finger]] * Ts[id_joint_finger] * Transform(dR_dts[id_theta_finger][id_joint_finger]);
        dG_dts[id_theta_finger][id_joint_finger + 1] = dG_dts[id_theta_finger][m_A[id_joint_finger + 1]] * Ts[id_joint_finger + 1] * Rs[id_joint_finger + 1];
        dG_dts[id_theta_finger][id_joint_finger + 2] = dG_dts[id_theta_finger][m_A[id_joint_finger + 2]] * Ts[id_joint_finger + 2] * Rs[id_joint_finger + 2];

        // wrt theta z at finger base
        dG_dts[id_theta_finger + 1][id_joint_finger] = Gs[m_A[id_joint_finger]] * Ts[id_joint_finger] * Transform(dR_dts[id_theta_finger + 1][id_joint_finger]);
        dG_dts[id_theta_finger + 1][id_joint_finger + 1] = dG_dts[id_theta_finger + 1][m_A[id_joint_finger + 1]] * Ts[id_joint_finger + 1] * Rs[id_joint_finger + 1];
        dG_dts[id_theta_finger + 1][id_joint_finger + 2] = dG_dts[id_theta_finger + 1][m_A[id_joint_finger + 2]] * Ts[id_joint_finger + 2] * Rs[id_joint_finger + 2];

        // wrt theta z at finger middle
        dG_dts[id_theta_finger + 2][id_joint_finger + 1] = Gs[m_A[id_joint_finger + 1]] * Ts[id_joint_finger + 1] * Transform(dR_dts[id_theta_finger + 2][id_joint_finger + 1]);
        dG_dts[id_theta_finger + 2][id_joint_finger + 2] = dG_dts[id_theta_finger + 2][m_A[id_joint_finger + 2]] * Ts[id_joint_finger + 2] * Rs[id_joint_finger + 2];

        // wrt theta z at finger distal
        dG_dts[id_theta_finger + 3][id_joint_finger + 2] = Gs[m_A[id_joint_finger + 2]] * Ts[id_joint_finger + 2] * Transform(dR_dts[id_theta_finger + 3][id_joint_finger + 2]);
    }

    return dG_dts;
}

std::vector<std::vector<Vector3d>> Mano::compute_dvp_dts(const VectorXd &theta, const std::vector<std::vector<Matrix3dR>> &dR_dts) const
{
    // init
    std::vector<std::vector<Vector3d>> dvp_dts(theta.size());
    for (auto &dvp_dt : dvp_dts)
        dvp_dt.resize(n_vertices(), Vector3d::Zero());

    // wrt each theta
    for (int id_theta{0}; id_theta < theta.size(); ++id_theta)
    {
        for (int id_vert{0}; id_vert < n_vertices(); ++id_vert)
        {
            for (int id_joint{1}; id_joint < n_joints(); ++id_joint)
            {
                for (int id_row{0}; id_row < 3; ++id_row)
                {
                    for (int id_col{0}; id_col < 3; ++id_col)
                    {
                        dvp_dts[id_theta][id_vert](0) += dR_dts[id_theta][id_joint](id_row, id_col) * m_P(3 * id_vert + 0, 9 * (id_joint-1) + id_row*3+id_col);
                        dvp_dts[id_theta][id_vert](1) += dR_dts[id_theta][id_joint](id_row, id_col) * m_P(3 * id_vert + 1, 9 * (id_joint-1) + id_row*3+id_col);
                        dvp_dts[id_theta][id_vert](2) += dR_dts[id_theta][id_joint](id_row, id_col) * m_P(3 * id_vert + 2, 9 * (id_joint-1) + id_row*3+id_col);
                    }
                }
            }
        }
    }

    return dvp_dts;
}




void Mano::read_bin(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    assert(file);

    read_trimesh(file);
    read_S(file);
    read_P(file);
    read_J(file);
    read_W(file);
    read_A(file);
    read_pose_pca_basis(file);
    read_pose_pca_mean(file);
}

void Mano::read_trimesh(std::ifstream &file)
{
    // vertices
    int n_verts{-1};
    int n_dim{-1};
    file.read(reinterpret_cast<char *>(&n_verts), sizeof n_verts);
    file.read(reinterpret_cast<char *>(&n_dim), sizeof n_dim);
    trimesh->vertices_.resize(n_verts);
    // file.read(reinterpret_cast<char *>(&mesh->vertices_[0](0)), sizeof(double) * n_verts * n_dim);
    file.read(reinterpret_cast<char *>(trimesh->vertices_.data()), sizeof(double) * n_verts * n_dim);

    // triangles
    int n_triangles{-1};
    int n_verts_in_triangle{-1}; // 3
    file.read(reinterpret_cast<char *>(&n_triangles), sizeof n_triangles);
    file.read(reinterpret_cast<char *>(&n_verts_in_triangle), sizeof n_verts_in_triangle);
    trimesh->triangles_.resize(n_triangles);
    file.read(reinterpret_cast<char *>(trimesh->triangles_.data()), sizeof(int) * n_triangles * n_verts_in_triangle);
}

void Mano::read_S(std::ifstream &file)
{
    int n_verts{-1};
    int dim{-1};
    int n_basis{-1};
    file.read(reinterpret_cast<char *>(&n_verts), sizeof n_verts);
    file.read(reinterpret_cast<char *>(&dim), sizeof dim);
    file.read(reinterpret_cast<char *>(&n_basis), sizeof n_basis);
    m_S.resize(n_verts * dim, n_basis);
    file.read(reinterpret_cast<char *>(m_S.data()), sizeof(double) * n_verts * dim * n_basis);
}

void Mano::read_P(std::ifstream &file)
{
    int n_verts{-1};
    int dim{-1};
    int n_basis{-1};
    file.read(reinterpret_cast<char *>(&n_verts), sizeof n_verts);
    file.read(reinterpret_cast<char *>(&dim), sizeof dim);
    file.read(reinterpret_cast<char *>(&n_basis), sizeof n_basis);
    m_P.resize(n_verts * dim, n_basis);
    file.read(reinterpret_cast<char *>(m_P.data()), sizeof(double) * n_verts * dim * n_basis);
}

void Mano::read_J(std::ifstream &file)
{
    int n_joints{-1};
    int n_verts{-1};
    file.read(reinterpret_cast<char *>(&n_joints), sizeof n_joints);
    file.read(reinterpret_cast<char *>(&n_verts), sizeof n_verts);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J_dense(n_joints, n_verts);
    file.read(reinterpret_cast<char *>(J_dense.data()), sizeof(double) * n_joints * n_verts);
    m_J = J_dense.sparseView();
}

void Mano::read_W(std::ifstream &file)
{
    int n_verts{-1};
    int n_joints{-1};
    file.read(reinterpret_cast<char *>(&n_verts), sizeof n_verts);
    file.read(reinterpret_cast<char *>(&n_joints), sizeof n_joints);
    m_W.resize(n_verts, n_joints);
    file.read(reinterpret_cast<char *>(m_W.data()), sizeof(double) * n_verts * n_joints);
}

void Mano::read_A(std::ifstream &file)
{
    int n_joints{-1};
    file.read(reinterpret_cast<char *>(&n_joints), sizeof n_joints);
    m_A.resize(n_joints);
    file.read(reinterpret_cast<char *>(m_A.data()), sizeof(int) * n_joints);
}

void Mano::read_pose_pca_basis(std::ifstream &file)
{
    int n_basis{-1};
    int dim{-1};
    file.read(reinterpret_cast<char *>(&n_basis), sizeof n_basis);
    file.read(reinterpret_cast<char *>(&dim), sizeof dim);
    pose_pca_basis.resize(n_basis, dim);
    file.read(reinterpret_cast<char *>(pose_pca_basis.data()), sizeof(double) * n_basis * dim);
}

void Mano::read_pose_pca_mean(std::ifstream &file)
{
    int n_basis{-1};
    file.read(reinterpret_cast<char *>(&n_basis), sizeof n_basis);
    pose_pca_mean.resize(n_basis);
    file.read(reinterpret_cast<char *>(pose_pca_mean.data()), sizeof(double) * n_basis);
}
