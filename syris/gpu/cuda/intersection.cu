// The function returns the index of the largest
// component of the direction vector.
__forceinline__ int findLargestComp(float dir[3]){
    int kz = 0;

    float max = fabsf(dir.x);

    for (int i = 1; i < 3; i++) {
        float f = fabsf(dir[i]);
        if (f > max) {
            max = f;
            kz = i;
        }
    }

    return kz;
}

// Both the ray and the triangle were transformed beforehand
// so that the ray is a unit vector along the z-axis (0,0,1),
// and the triangle is transformed with the same matrix
// (which is M in the paper). This function is called only
// when the ray is co-planar to the triangle (with the
// determinant being zero). The rotation by this function is
// to prepare for the ray-edge intersection calculations in 2D.
// The rotation is around the z-axis. For any point after
// the rotation, its new x* equals its original length
// with the correct sign, and the new y* = z. The current
// implementation avoids explicitly defining rotation angles
// and directions. The following ray-edge intersection will
// be in the x*-y* plane.
__forceinline__ float4 rotate2D(float *point) {
    float4 point_star;
    float r = sqrtf(point[0] * point[0] + point[1] * point[1]);
    if (point[0] != 0) {
        point_star.x = (point[0] > 0 ? 1 : -1) * r;
    }
    else {
        point_star.x = (point[1] > 0 ? 1 : -1) * r;
    }
    point_star.y = point[2];
    point_star.z = 0.f;
    return point_star;
}

// The function is for ray-edge intersection
// with the rotated ray along the z-axis and
// the transformed and rotated triangle edges
// The algorithm is described in
// https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
__forceinline__ bool rayEdgeIntersect(
    float4 const &edge_vertex_1,
    float4 const &edge_vertex_2,
    float &t)
{
  float x3 = edge_vertex_1.x;
  float y3 = edge_vertex_1.y;
  float x4 = edge_vertex_2.x;
  float y4 = edge_vertex_2.y;

  float y2 = fabsf(y3) > fabsf(y4) ? y3 : y4;

  float det = y2 * (x3 - x4);

  //  the ray is parallel to the edge if det == 0.0
  //  When the ray overlaps the edge (x3==x4==0.0), it also returns false,
  //  and the intersection will be captured by the other two edges.
  if (det == 0) {
    return false;
  }

  t = (x3 * y4 - x4 * y3) / det * y2;

  float u = x3 * y2 / det;

  float const epsilon = 0.00001f;
  return (u >= -epsilon && u <= 1 + epsilon);
}

// The algorithm is described in
// Watertight Ray/Triangle Intersection
// [1] Woop, S. et al. (2013),
// Journal of Computer Graphics Techniques Vol. 2(1)
// The major difference is that here we return the intersection points
// when the ray and the triangle is coplanar.
// In the paper, they just need the boolean return.
bool Ray::intersects(
    float4 const &V1, float4 const &V2, float4 const &V3,
    float &tmin, float &tmax) {
    // normalize the direction vector by its largest component.
    float dir[3] = {this->direction.x, this->direction.y, this->direction.z};
    auto kz = findLargestComp(dir);
    int kx = (kz + 1) % 3;
    int ky = (kz + 2) % 3;

    if (dir[kz] < 0) {
        // Swap kx and ky
        float tmp = dir[kx];
        dir[kx] = dir[ky];
        dir[ky] = tmp;
    }

    float S[3];

    S[2] = 1.0f / dir[kz];
    S[0] = dir[kx] * S[2];
    S[1] = dir[ky] * S[2];

    // calculate vertices relative to ray origin
    float oA[3];
    oA[0] = V1.x - this->tail.x;
    oA[1] = V1.y - this->tail.y;
    oA[2] = V1.z - this->tail.z;
    float oB[3];
    oB[0] = V2.x - this->tail.x;
    oB[1] = V2.y - this->tail.y;
    oB[2] = V2.z - this->tail.z;
    float oC[3];
    oC[0] = V3.x - this->tail.x;
    oC[1] = V3.y - this->tail.y;
    oC[2] = V3.z - this->tail.z;

    // oA, oB, oB need to be normalized, otherwise they
    // will scale with the problem size.
    float const mag_oA = sqrtf(oA[0] * oA[0] + oA[1] * oA[1] + oA[2] * oA[2]);
    float const mag_oB = sqrtf(oB[0] * oB[0] + oB[1] * oB[1] + oB[2] * oB[2]);
    float const mag_oC = sqrtf(oC[0] * oC[0] + oC[1] * oC[1] + oC[2] * oC[2]);

    float mag_bar = 3.0 / (mag_oA + mag_oB + mag_oC);

    float A[3];
    float B[3];
    float C[3];

    // perform shear and scale of vertices
    // normalized by mag_bar
    A[0] = (oA[kx] - s[0] * oA[kz]) * mag_bar;
    A[1] = (oA[ky] - s[1] * oA[kz]) * mag_bar;
    B[0] = (oB[kx] - s[0] * oB[kz]) * mag_bar;
    B[1] = (oB[ky] - s[1] * oB[kz]) * mag_bar;
    C[0] = (oC[kx] - s[0] * oC[kz]) * mag_bar;
    C[1] = (oC[ky] - s[1] * oC[kz]) * mag_bar;

    // calculate scaled barycentric coordinates
    float u = C[0] * B[1] - C[1] * B[0];
    float v = A[0] * C[1] - A[1] * C[0];
    float w = B[0] * A[1] - B[1] * A[0];

    // fallback to double precision
    if (u == 0 || v == 0 || w == 0) {
        u = (double)C[0] * B[1] - (double)C[1] * B[0];
        v = (double)A[0] * C[1] - (double)A[1] * C[0];
        w = (double)B[0] * A[1] - (double)B[1] * A[0];
    }

    tmin = INF;
    tmax = -INF;

    // 'Back-face culling' is not supported.
    // Back-facing culling is to check whether
    // a surface is 'visible' to a ray, which requires
    // consistent definition of the facing of triangles.
    // Once the facing of triangle is defined,
    // only one of the conditions is needed,
    // either (u < 0 || v < 0 || w < 0) or
    // (u > 0 || v > 0 || w > 0), for Back-facing culling.
    float const epsilon = 0.0000001f;
    if ((u < -epsilon || v < -epsilon || w < -epsilon) &&
        (u > epsilon || v > epsilon || w > epsilon)) {
        return false;
    }

    // calculate determinant
    float det = u + v + w;

    A[2] = S[2] * oA[kz];
    B[2] = S[2] * oB[kz];
    C[2] = S[2] * oC[kz];

    if (det < -epsilon || det > epsilon) {
        float t = (u * A[2] + v * B[2] + w * C[2]) / det;
        tmax = t;
        tmin = t;
        return true;
    }

    // The ray is co-planar to the triangle.
    // Check the intersection with each edge
    // the rotate2D function is to make sure the ray-edge
    // intersection check is at the plane where ray and edges
    // are at.
    float4 A_star = rotate2D(A);
    float4 B_star = rotate2D(B);
    float4 C_star = rotate2D(C);

    float t_ab = INF;
    bool ab_intersect = rayEdgeIntersect(A_star, B_star, t_ab);
    if (ab_intersect) {
        tmin = t_ab;
        tmax = t_ab;
    }
    float t_bc = INF;
    bool bc_intersect = rayEdgeIntersect(B_star, C_star, t_bc);
    if (bc_intersect) {
        tmin = fminf(tmin, t_bc);
        tmax = fmaxf(tmax, t_bc);
    }
    float t_ca = INF;
    bool ca_intersect = rayEdgeIntersect(C_star, A_star, t_ca);
    if (ca_intersect) {
        tmin = fminf(tmin, t_ca);
        tmax = fmaxf(tmax, t_ca);
    }

    if (ab_intersect || bc_intersect || ca_intersect) {
        // When (1) the origin of the ray is within the triangle
        // and (2) they ray is coplanar with the triangle, the
        // intersection length is zero.
        if (tmin * tmax <= 0) {
            tmin = 0;
            tmax = 0;
        }
        else {
            // need to separate tmin tmax >0 and <0 cases
            // e.g., tmin = -2 and tmax = -1, but
            // we want tmin = -1 and tmax = -2, when the
            // ray travels backward
            if (tmin < 0) {
                float tmp = tmin;
                tmin = tmax;
                tmax = tmp;
            }
        }
        return true;
    }

    return false;
} // namespace Experimental

__forceinline__ bool intersects(bool Ray::intersects(
    float4 const &V1, float4 const &V2, float4 const &V3, float &t))
{
  float tmin;
  float tmax;
  // intersects only if triangle is in front of the ray
  return intersection(ray, triangle, tmin, tmax) && (tmax >= 0.f);
}