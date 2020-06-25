#include "fitting_functor.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/LevenbergMarquardt>

#include <vector>
#include <sstream>
#include <filesystem>
#include <iomanip>

using namespace Eigen;

int FittingFunctor::operator()(const InputType &x, ValueType &fvec)
{
    iteration += 1;

    // compute vertices and dVdp
    V = model->M(x.params);

    // visualize at each iteration
    V_lineset->points_ = V;
    vis->update(V_lineset);
    if(!vis->show())
    {
        std::cout << "Exiting...\n";
        std::exit(0);
    }
    
    if (!dir_log.empty())
    {
        std::stringstream ss;
        ss << dir_log.string() << "/" << std::setw(2) << std::setfill('0') << iteration << ".png";
        vis->save_image(ss.str());
    }

    // compute surface features with jacobian
    sf = model->S(x.sps, V);

    // data term
    for (int i = 0; i < X->size(); ++i)
    {
        fvec.segment(i * 3, 3) = sf.get_S()[i] - X->at(i);
    }

    return 0;
}


int FittingFunctor::df(const InputType &x, JacobianType &fjac)
{
    // derivative of model wrt parameters
    const std::vector<Eigen::VectorXd> dVdp = model->compute_dVdp(x.params);

    // derivative wrt vertices
    SparseMatrix<double> dSdV = sf.get_dSdV(model->n_verts());
    
    // init entries in jacobian to zero
    fjac.resize(
        3 * X->size(),        // n_residuals
        2 * X->size()         // n_correspondences (u+v)
            + x.params.size()  // model params
    );

    // fill non-zero entries
    for (int id_pt = 0; id_pt < X->size(); ++id_pt)
    {
        for (int id_coord{0}; id_coord < 3; ++id_coord) // x, y, z for each point
        {
            // derivatives wrt correspondences
            fjac.coeffRef(3*id_pt + id_coord, 2 * id_pt + 0) = sf.get_dSdu()[id_pt](id_coord); // dfi_j / du
            fjac.coeffRef(3*id_pt + id_coord, 2 * id_pt + 1) = sf.get_dSdv()[id_pt](id_coord); // dfi_j / dv

            // derivatives wrt parameters
            // int id_param = std::stoi(dir_log.filename());
            for (int id_param{0}; id_param < x.params.size(); ++id_param)
            {
                // dSdp = dSdV * dVdp
                for (int id_vert{0}; id_vert < model->n_verts(); ++id_vert)
                {
                    fjac.coeffRef(3*id_pt + id_coord, 2*X->size() + id_param) += dSdV.coeff(id_pt, id_vert) * dVdp[3*id_vert + id_coord](id_param);
                }
            }
        }
    }
    
    fjac.makeCompressed();
    
    return 0;
}


void FittingFunctor::increment_in_place(InputType *x, const StepType &p)
{
    // increment parameters
    x->params += p.segment(X->size()*2, x->params.size());

    // increment correspondences
    int loopers = 0;
    int total_hops = 0;
    for (int id_pt{0}; id_pt < X->size(); ++id_pt)
    {
        Vector2d dU = p.segment<2>(2 * id_pt);
        int n_hops = update_surface_point(model->face_adjs(), V, model->get_evaluator(), x->sps[id_pt], dU);
    }
}


double FittingFunctor::estimateNorm(const InputType &x, const StepType &diag)
{
    /* scale norm(InputType) by diag */

    // scale parameters
    Map<VectorXd> diag_params(const_cast<double *>(diag.data()) + x.sps.size() * 2, x.params.size());
    double total{0.0};
    // total += diag_params.segment(0, x.params.size()).cwiseProduct(x.params).squaredNorm();
    total += diag.segment(x.sps.size()*2, x.params.size()).cwiseProduct(x.params).squaredNorm();

    // scale correspondences
    for (int id_pt{0}; id_pt < x.sps.size(); ++id_pt)
    {
        const Vector2d &U = x.sps[id_pt].U;
        Vector2d di = diag.segment<2>(2 * id_pt); // correspoding diagonal elements
        total += U.cwiseProduct(di).squaredNorm();
    }

    return sqrt(total);
}



int FittingFunctor::update_surface_point(const std::vector<Eigen::Vector3i> &face_adjs, const std::vector<Eigen::Vector3d> &V, const SubdivEvaluator &evaluator, SurfacePoint &sp, const Vector2d &dU)
{
    constexpr int max_hops = 3;
    int face_old = sp.face;
    double u1_old = sp.U(0);
    double u2_old = sp.U(1);
    double du1 = dU(0);
    double du2 = dU(1);
    double u1_new = u1_old + du1;
    double u2_new = u2_old + du2;

    for (int count{0};; ++count)
    {
        bool crossing = (u1_new < 0.0) || (u2_new < 0.0) || ((u1_new + u2_new) > 1.0);

        if (!crossing)
        {
            sp.face = face_old;
            sp.U << u1_new, u2_new;
            return count;
        }

        // find the new face and coordinates of the crossing point within the old face and the new face
        int face_new;
        bool face_found = false;
        double u1_cross, u2_cross, aux;
        /* equation of line concept:
        * v_new = v_old + dv/du * (u_cross - u_old)
        * aux is used to compute offset in new face
        */
        if (u1_new < 0.0)
        {
            u1_cross = 0.0;
            u2_cross = u2_old + (du2 / du1) * (u1_cross - u1_old);
            aux = u2_cross;
            if ((u2_cross >= 0.0) && (u2_cross <= 1.0))
            {
                face_new = face_adjs[face_old](2);
                face_found = true;
            }
        }
        if ((u2_new < 0.0) && (!face_found))
        {
            u2_cross = 0;
            u1_cross = u1_old + (du1 / du2) * (u2_cross - u2_old);
            aux = u1_cross;
            if ((u1_cross >= 0.0) && (u1_cross <= 1.0))
            {
                face_new = face_adjs[face_old](0);
                face_found = true;
            }
        }
        if (((u1_new + u2_new) > 1.0) && (!face_found))
        {
            const double m = du2 / du1;
            u1_cross = (1 - u2_old + m * u1_old) / (m + 1);
            u2_cross = 1 - u1_cross;
            aux = u1_cross;
            if ((u1_cross >= 0.0) && (u1_cross <= 1.0))
            {
                face_new = face_adjs[face_old](1);
                face_found = true;
            }
        }
        // boundary face
        if (face_new == -1)
        {
            sp.face = face_old;
            sp.U << 0.3, 0.3;
            return -count;
        }
        assert(face_new != -1);
        assert(face_found);

        // find the coordinates of the crossing point as part of the new face
        // and update u_old (this will be new u in the next iter)
        int face_adj_of_old_wrt_new;
        for (int f{0}; f < 3; ++f)
        {
            if (face_adjs[face_new](f) == face_old)
            {
                face_adj_of_old_wrt_new = f;
                break;
            }
        }
        switch (face_adj_of_old_wrt_new)
        {
        case 0:
            u1_old = aux;
            u2_old = 0.0;
            break;
        case 1:
            u1_old = aux;
            u2_old = 1.0 - aux;
            break;
        case 2:
            u1_old = 0.0;
            u2_old = aux;
            break;
        }

        // evaluate subdiv surface at edge (wrt original face)
        std::vector<SurfacePoint> sps;
        sps.push_back({sp.face, {u1_cross, u2_cross}}); // crossing point wrt face
        sps.push_back({face_new, {u1_old, u2_old}}); // crossing point wrt face new
        SurfaceFeatures sf = evaluator.compute_surface_features(V, sps, false);
        Eigen::Matrix<double, 3, 2> J_Sa;
        J_Sa.col(0) = sf.get_dSdu()[0];
        J_Sa.col(1) = sf.get_dSdv()[0];

        Eigen::Matrix<double, 3, 2> J_Sb;
        J_Sb.col(0) = sf.get_dSdu()[1];
        J_Sb.col(1) = sf.get_dSdv()[1];

        // compute new u increments
        Vector2d du_remaining;
        du_remaining << u1_new - u1_cross, u2_new - u2_cross;
        Vector3d prod = J_Sa * du_remaining;
        Eigen::Matrix2d AtA = J_Sb.transpose() * J_Sb;
        Vector2d AtB = J_Sb.transpose() * prod;
        Vector2d u_incr = AtA.inverse() * AtB;
        du1 = u_incr[0];
        du2 = u_incr[1];

        if (count == max_hops)
        {
            double dmax = std::max(du1, du2);
            double scale = 0.5 / dmax;
            sp.face = face_old;
            sp.U << 0.3, 0.3;
            return -count;
        }

        // update for next iter
        u1_new = u1_old + du1;
        u2_new = u2_old + du2;
        face_old = face_new;
    }
}
