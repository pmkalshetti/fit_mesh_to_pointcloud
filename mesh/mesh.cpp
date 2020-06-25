#include "mesh.h"

using namespace Eigen;

void Mesh::update_adjacencies()
{
    face_adjs.resize(n_triangles(), Vector3i(-1, -1, -1));

    for (int f{0}; f < n_triangles(); ++f)
    {
        for (int k{0}; k < 3; ++k)
        {
            // find the kth edge
            int k_next = (k + 1) % 3;
            int edge[2] = {triangles()[f](k), triangles()[f](k_next)};

            // find face that shares its reverse
            int found = 0;
            int other = -1;
            for (int fa{0}; fa < n_triangles(); ++fa)
            {
                if (f == fa) continue;

                for (int l = 0; l < 3; ++l)
                {
                    int l_next = (l + 1) % 3;
                    if ((triangles()[fa](l) == edge[1]) && triangles()[fa](l_next) == edge[0])
                    {
                        other = fa;
                        found++;
                        break;
                    }
                }
                if (found) break;
            }
            // assert(found == 1);
            face_adjs[f](k) = other;
        }
    }
}