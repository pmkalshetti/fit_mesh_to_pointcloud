#include "evaluator.h"

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>

#include <random>

using namespace OpenSubdiv;
using namespace Eigen;


std::vector<SurfacePoint> SurfacePoint::generate(const int n_points, const int n_triangles)
{
    std::vector<SurfacePoint> sps(n_points);

    std::mt19937 random_generator(1);
    std::uniform_real_distribution dist_u(0.0, 1.0);
    std::uniform_real_distribution dist_v(0.0, 1.0);
    std::uniform_int_distribution dist_face(0, n_triangles-1);
    for (int i{0}; i < n_points; ++i)
    {
        // generate random u,v
        double u = dist_u(random_generator);
        double v = dist_v(random_generator);
        if ((u+v) > 1)
        {
            u = 1 - u;
            v = 1 - v;
        }

        sps[i].U << u, v;
        sps[i].face = dist_face(random_generator);
    }

    return sps;
}


SubdivEvaluator::SubdivEvaluator(const int n_vertices, const std::vector<Vector3i> &triangles, const bool has_boundary)
{
    // create refiner using descriptor
    Far::TopologyDescriptor desc;
    desc.numVertices = n_vertices;
    desc.numFaces = triangles.size();
    std::vector<int> num_verts_per_face(triangles.size(), 3);
    desc.numVertsPerFace = num_verts_per_face.data();
    desc.vertIndicesPerFace = &triangles[0](0);

    Sdc::SchemeType type = Sdc::SCHEME_LOOP;
    Sdc::Options options;
    if (has_boundary)
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
    else
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);
    using Refiner_t = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
    Far::TopologyRefiner *refiner_for_patch_table = Refiner_t::Create(desc, Refiner_t::Options(type, options));

    // refine topology (adaptive takes care of irregular vertices)
    const int max_isolation = 0; // do not change this!
    refiner_for_patch_table->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions(max_isolation));

    // generate PatchTable that will be used to evaluate surface limit
    Far::PatchTableFactory::Options patch_options;
    patch_options.endCapType = Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;
    patch_table = Far::PatchTableFactory::Create(*refiner_for_patch_table, patch_options);

    // compute total number of points needed to evaluate patch table
    // use local points around irregular vertices
    n_refiner_vertices = refiner_for_patch_table->GetNumVerticesTotal();
    n_local_points = patch_table->GetNumLocalPoints();

    // create a buffer to hold the position of the refined verts and local points
    // evaluation_verts_buffer.resize(n_refiner_vertices + n_local_points);

    // refiner for subdividing mesh
    refiner = Refiner_t::Create(desc, Refiner_t::Options(type, options));

    // uniformly refine topology up to maxlevel
    const int maxlevel{3};
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    delete refiner_for_patch_table;
}


SurfaceFeatures SubdivEvaluator::compute_surface_features(const std::vector<Vector3d> &V, const std::vector<SurfacePoint> &sps, const bool compute_dSdV) const
{
    // create a buffer to hold the position of the refined verts and local points
    std::vector<OSDVertex> refined_osdverts(n_refiner_vertices + n_local_points);

    // compute local points from coarse verts
    for (int i{0}; i < V.size(); ++i)
        refined_osdverts[i].point = V[i];
    int n_stencils = patch_table->GetLocalPointStencilTable()->GetNumStencils();
    patch_table->ComputeLocalPointValues(&refined_osdverts[0], &refined_osdverts[n_refiner_vertices]);

    // get all stencils from patch table
    // (necesary to obtain the weights for the gradients)
    const Far::StencilTable *stencil_table = patch_table->GetLocalPointStencilTable();
    std::vector<Far::Stencil> stencils(n_stencils);
    for (int i{0}; i < n_stencils; ++i)
    {
        stencils[i] = stencil_table->GetStencil(Far::Index(i));
    }

    // create PatchMap to locate patches in the table
    Far::PatchMap patch_map(*patch_table);

    // init surface features
    SurfaceFeatures sf(sps.size());
    
    // evaluate surface at uvs
    for (int i{0}; i < sps.size(); ++i)
    {
        // locate patch corresponding to uv
        int face = sps[i].face;
        double u = sps[i].U(0);
        double v = sps[i].U(1);
        const Far::PatchTable::PatchHandle *patch_handle = patch_map.FindPatch(face, u, v);

        // evaluate patch weights
        patch_table->EvaluateBasis(*patch_handle, u, v, sf.p_weights(), sf.du_weights(), sf.dv_weights());

        // identify control vertices corresponding to this patch
        Far::ConstIndexArray cvs = patch_table->GetPatchVertices(*patch_handle);

        // for each control vertex
        for (int cv1{0}; cv1 < cvs.size(); ++cv1)
        {
            // fill surface features by corresponding weighted combination of control vertices
            sf.update_S_dSdU(refined_osdverts[cvs[cv1]].point, i, cv1);
            
            // compute derivative wrt control vertices
            if (compute_dSdV)
            {
                std::vector<double> accumulated_weights(V.size(), 0.0);
                std::vector<bool> has_nonzero_weight(V.size(), false);
                for (int cv2{0}; cv2 < cvs.size(); ++cv2)
                {
                    if (cvs[cv2] < V.size())  // regular vertex
                    {
                        accumulated_weights[cvs[cv2]] += sf.p_weights()[cv2];
                        has_nonzero_weight[cv2] = true;
                    }
                    else  // local point
                    {
                        int idx_offset = cvs[cv2] - V.size();
                        // look at the stencil associated with this local point and distribute its weight over the control vertices
                        const Far::Index *stencil_idx = stencils[idx_offset].GetVertexIndices();
                        const float *stencil_weights = stencils[idx_offset].GetWeights();
                        for (int s{0}; s < stencils[idx_offset].GetSize(); ++s)
                        {
                            accumulated_weights[stencil_idx[s]] += sf.p_weights()[cv2] * stencil_weights[s];
                            has_nonzero_weight[stencil_idx[s]] = true;
                        }
                    }
                }

                // store the weights
                float scale = 1.f / cvs.size();
                for (int cv3{0}; cv3 < V.size(); ++cv3)
                {
                    if(has_nonzero_weight[cv3])
                    {
                        sf.update_dSdV(i, cv3, accumulated_weights[cv3]*scale);
                    }
                }
            }
        }
    }
    
    sf.compute_normal();

    return sf;
}

Mesh SubdivEvaluator::generate_refined_mesh(const std::vector<Eigen::Vector3d> &V, const int level) const
{
    // Reference: http://graphics.pixar.com/opensubdiv/docs/far_tutorial_1_1.html

    // allocate buffer for vertex primvar data.
    // length = sum of vertices at all levels
    std::vector<OSDVertex> vbuffer(refiner->GetNumVerticesTotal());
    OSDVertex *verts{&vbuffer[0]}; // this interface is used for filling up
    
    // copy coarse mesh positions
    for (int i{0}; i < V.size(); ++i)
    {
        verts[i].point = V[i];
    }
    // use primar refiner for filling other levels using interpolation
    Far::PrimvarRefiner primvar_refiner(*refiner);
    // Note: previous level verts are used for next level interpolation
    OSDVertex *src = verts;
    for (int l{1}; l <= level; ++l)
    {
        OSDVertex *dst = src + refiner->GetLevel(l - 1).GetNumVertices();
        primvar_refiner.Interpolate(l, src, dst);
        src = dst;
    }
    // src will point to vertices at `level` interpolation after end of above loop

    // extract refined vertices
    const Far::TopologyLevel &ref_level = refiner->GetLevel(level);
    int n_verts_out = ref_level.GetNumVertices();
    std::vector<Eigen::Vector3d> refined_vertices(n_verts_out);
    for (int v{0}; v < n_verts_out; ++v)
    {
        refined_vertices[v] = src[v].point;
    }

    // extract refined faces into out topo
    int n_faces_out = ref_level.GetNumFaces();
    std::vector<Eigen::Vector3i> refined_triangles(n_faces_out);
    for (int f{0}; f < n_faces_out; ++f)
    {
        Far::ConstIndexArray face_vert_ids = ref_level.GetFaceVertices(f);
        for (int v{0}; v < face_vert_ids.size(); ++v)
        {
            refined_triangles[f](v) = face_vert_ids[v];
        }
    }

    return Mesh(refined_vertices, refined_triangles);
}

