#include "WatertightRay.cuh"
#include "Commons.cuh"

// One representable step in the selected precision, including signed zero.
__device__ FP_T RoundUp(FP_T a) {
#ifdef __FP_T_D__
    return nextafter(a, double(POS_INFINITY));
#else
    return nextafterf(a, POS_INFINITY);
#endif
}
__device__ FP_T RoundDown(FP_T a) {
#ifdef __FP_T_D__
    return nextafter(a, double(NEG_INFINITY));
#else
    return nextafterf(a, NEG_INFINITY);
#endif
}
__device__ FP_T reciprocalNear(FP_T a) {
    return copysign(RoundDown(RoundDown(fabs(a))), a);
}
__device__ FP_T reciprocalFar(FP_T a) {
    return copysign(RoundUp(RoundUp(fabs(a))), a);
}

__device__ FP_T subtractDown(FP_T a, FP_T b) {
#ifdef __FP_T_D__
    return __dsub_rd(a, b);
#else
    return __fsub_rd(a, b);
#endif
}
__device__ FP_T subtractUp(FP_T a, FP_T b) {
#ifdef __FP_T_D__
    return __dsub_ru(a, b);
#else
    return __fsub_ru(a, b);
#endif
}
__device__ FP_T multiplyDown(FP_T a, FP_T b) {
#ifdef __FP_T_D__
    return __dmul_rd(a, b);
#else
    return __fmul_rd(a, b);
#endif
}
__device__ FP_T multiplyUp(FP_T a, FP_T b) {
#ifdef __FP_T_D__
    return __dmul_ru(a, b);
#else
    return __fmul_ru(a, b);
#endif
}

// --- WatertightRay Constructor Implementation ---
__device__ WatertightRay::WatertightRay(FP_T4 origin, FP_T4 direction, const FP_T4& scene_min, const FP_T4& scene_max, const EpsilonParams& epsilons) : m_epsilons(epsilons)
{
    // Basic setup
    this->tail = origin;
    this->direction = direction;
    this->invDirection = MAKE_FP_T4(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z, 0.0f);
    this->scene_min = scene_min;
    this->scene_max = scene_max;
    
    // Create local arrays for indexed access
    const FP_T O[3] = { origin.x, origin.y, origin.z };
    const FP_T D[3] = { direction.x, direction.y, direction.z };
    const FP_T invD[3] = { invDirection.x, invDirection.y, invDirection.z };
    const FP_T bbMin[3] = { scene_min.x, scene_min.y, scene_min.z };
    const FP_T bbMax[3] = { scene_max.x, scene_max.y, scene_max.z };

    // Remap axes based on the ray's dominant direction (Swizzling)
    this->Kz = max_dim_index(abs(this->direction));
    this->Kx = this->Kz + 1;
    if (this->Kx == 3)
        this->Kx = 0;
    this->Ky = this->Kx + 1;
    if (this->Ky == 3)
        this->Ky = 0;

    if (D[this->Kz] < (0.0))
    {
        swap<int>(this->Kx, this->Ky);
    }

    this->Sz = (1.0) / D[this->Kz];
    this->Sx = D[this->Kx] / D[this->Kz];
    this->Sy = D[this->Ky] / D[this->Kz];

    // Pre-calculate sorted plane indices based on ray direction
    const int near_p[3] = {0, 1, 2}; // Indices for min planes
    const int far_p[3]  = {3, 4, 5}; // Indices for max planes
    this->near_kx_idx = near_p[this->Kx]; this->far_kx_idx = far_p[this->Kx];
    this->near_ky_idx = near_p[this->Ky]; this->far_ky_idx = far_p[this->Ky];
    this->near_kz_idx = near_p[this->Kz]; this->far_kz_idx = far_p[this->Kz];
    if (D[this->Kx] < 0.0f) swap(this->near_kx_idx, this->far_kx_idx);
    if (D[this->Ky] < 0.0f) swap(this->near_ky_idx, this->far_ky_idx);
    if (D[this->Kz] < 0.0f) swap(this->near_kz_idx, this->far_kz_idx);

    // Pre-calculate conservatively rounded reciprocal directions
    this->r_near_kx = reciprocalNear(invD[this->Kx]); this->r_far_kx = reciprocalFar(invD[this->Kx]);
    this->r_near_ky = reciprocalNear(invD[this->Ky]); this->r_far_ky = reciprocalFar(invD[this->Ky]);
    this->r_near_kz = reciprocalNear(invD[this->Kz]); this->r_far_kz = reciprocalFar(invD[this->Kz]);
    
    // Calculate error bounds and corrected origins
    FP_T ROBUST_EPSILON = fmax(m_epsilons.ray_box_epsilon,
        FP_T(2.5) * ::cuda::std::numeric_limits<FP_T>::epsilon());

    const FP_T L[3] = { RoundUp(fabs(O[0] - bbMin[0])), RoundUp(fabs(O[1] - bbMin[1])), RoundUp(fabs(O[2] - bbMin[2])) };
    const FP_T U[3] = { RoundUp(fabs(O[0] - bbMax[0])), RoundUp(fabs(O[1] - bbMax[1])), RoundUp(fabs(O[2] - bbMax[2])) };
    
    const FP_T max_dist[3] = {
        RoundUp(device_fmax<FP_T>(L[0], U[0])),
        RoundUp(device_fmax<FP_T>(L[1], U[1])),
        RoundUp(device_fmax<FP_T>(L[2], U[2]))
    };

    const FP_T Z_max = max_dist[this->Kz];

    const FP_T err_kx = RoundUp(ROBUST_EPSILON * RoundUp(max_dist[this->Kx] + Z_max));
    const FP_T err_ky = RoundUp(ROBUST_EPSILON * RoundUp(max_dist[this->Ky] + Z_max));

    // Directly calculate final member variables for the corrected origins
    this->o_near_kx = RoundUp(O[this->Kx] + err_kx);
    this->o_far_kx  = RoundDown(O[this->Kx] - err_kx);
    if (D[this->Kx] < 0.0f) swap(this->o_near_kx, this->o_far_kx);

    this->o_near_ky = RoundUp(O[this->Ky] + err_ky);
    this->o_far_ky  = RoundDown(O[this->Ky] - err_ky);
    if (D[this->Ky] < 0.0f) swap(this->o_near_ky, this->o_far_ky);
    
    // The dominant axis (Kz) needs no shear error correction for its origin
    this->o_near_kz = O[this->Kz];
    this->o_far_kz  = O[this->Kz];

    this->sign[0] = (this->invDirection.x < 0);
    this->sign[1] = (this->invDirection.y < 0);
    this->sign[2] = (this->invDirection.z < 0);
}


__device__ bool intersectSlab(FP_T near_plane, FP_T far_plane,
                              FP_T origin_near, FP_T origin_far,
                              FP_T reciprocal_near, FP_T reciprocal_far,
                              FP_T direction, FP_T &entry, FP_T &exit)
{
    // Avoid 0 * infinity on a box face. Both signs of zero are parallel.
    if (direction == 0) return near_plane <= origin_near && far_plane >= origin_far;
    FP_T near_delta = direction > 0 ? subtractDown(near_plane, origin_near)
                                   : subtractUp(near_plane, origin_near);
    FP_T far_delta = direction > 0 ? subtractUp(far_plane, origin_far)
                                  : subtractDown(far_plane, origin_far);
    entry = fmax(entry, multiplyDown(near_delta, reciprocal_near));
    exit = fmin(exit, multiplyUp(far_delta, reciprocal_far));
    return entry <= exit;
}

__device__ bool WatertightRay::intersects(const FP_T4& box_min, const FP_T4& box_max,
                                         FP_T tmin, FP_T tmax) const
{
    const FP_T b[6] = {box_min.x, box_min.y, box_min.z, box_max.x, box_max.y, box_max.z};
    const FP_T d[3] = {direction.x, direction.y, direction.z};
    return intersectSlab(b[near_kx_idx], b[far_kx_idx], o_near_kx, o_far_kx,
                         r_near_kx, r_far_kx, d[Kx], tmin, tmax)
        && intersectSlab(b[near_ky_idx], b[far_ky_idx], o_near_ky, o_far_ky,
                         r_near_ky, r_far_ky, d[Ky], tmin, tmax)
        && intersectSlab(b[near_kz_idx], b[far_kz_idx], o_near_kz, o_far_kz,
                         r_near_kz, r_far_kz, d[Kz], tmin, tmax);
}

__device__ FP_T WatertightRay::point_2_parametric (FP_T4 const &point) const {
    return dot(point - tail, direction) / dot(direction, direction);
}

__device__ bool WatertightRay::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T t_near = (0.0);
    FP_T t_far  = POS_INFINITY;
    return this->intersects(minBbox, maxBbox, t_near, t_far);
}
// Keep both products rounded separately. An asymmetric FMA can give the same
// nonzero sign to an edge and its reversal, opening a crack between triangles.
__device__ FP_T edge_difference(FP_T a, FP_T b, FP_T c, FP_T d)
{
#ifdef __FP_T_D__
    return __dsub_rn(__dmul_rn(a, b), __dmul_rn(c, d));
#else
    return __fsub_rn(__fmul_rn(a, b), __fmul_rn(c, d));
#endif
}

__device__ double edge_difference_double(double a, double b, double c, double d)
{
    return __dsub_rn(__dmul_rn(a, b), __dmul_rn(c, d));
}

// Shared edges must also report the same distance, independently of the third
// vertex and triangle winding. This lets the thickness pass identify duplicate
// boundary hits without merging nearby, physically distinct surfaces.
__device__ double boundary_distance(const FP_T *x, const FP_T *y, const FP_T *z,
                                    int a, int b, FP_T sz)
{
    if (x[a] > x[b] || (x[a] == x[b] && (y[a] > y[b] ||
        (y[a] == y[b] && z[a] > z[b])))) swap(a, b);
    double dx = double(x[b]) - double(x[a]);
    double dy = double(y[b]) - double(y[a]);
    double fraction = fabs(dx) >= fabs(dy) ? -double(x[a]) / dx : -double(y[a]) / dy;
    fraction = fmin(1.0, fmax(0.0, fraction));
    return fma(fraction, double(z[b]) - double(z[a]), double(z[a])) * double(sz);
}

__device__ bool WatertightRay::intersects(
    FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3,
    FP_T &t, unsigned col, unsigned row) const
{
    const FP_T4 relative[3] = {V1 - tail, V2 - tail, V3 - tail};
    FP_T x[3], y[3], z[3];
    for (int i = 0; i < 3; ++i) {
        const FP_T v[3] = {relative[i].x, relative[i].y, relative[i].z};
        x[i] = fma(-Sx, v[Kz], v[Kx]);
        y[i] = fma(-Sy, v[Kz], v[Ky]);
        z[i] = v[Kz];
    }

    double u = edge_difference(x[2], y[1], y[2], x[1]);
    double v = edge_difference(x[0], y[2], y[0], x[2]);
    double w = edge_difference(x[1], y[0], y[1], x[0]);
#ifndef __FP_T_D__
    // Woop, Benthin & Wald, JCGT 2(1), 2013, Appendix A. Reuse the
    // SAME sheared coordinates: a triangle-dependent transform would break
    // the common edge geometry. Products of two floats are exact in double.
    if (u == 0.0 || v == 0.0 || w == 0.0) {
        u = edge_difference_double(x[2], y[1], y[2], x[1]);
        v = edge_difference_double(x[0], y[2], y[0], x[2]);
        w = edge_difference_double(x[1], y[0], y[1], x[0]);
    }
#endif
    if ((u < 0.0 || v < 0.0 || w < 0.0) &&
        (u > 0.0 || v > 0.0 || w > 0.0)) return false;
    const double det = (u + v) + w;
    if (det == 0.0 || !isfinite(det)) return false;

    double distance;
    if (u == 0.0 && v == 0.0) distance = double(z[2]) * double(Sz);
    else if (u == 0.0 && w == 0.0) distance = double(z[1]) * double(Sz);
    else if (v == 0.0 && w == 0.0) distance = double(z[0]) * double(Sz);
    else if (u == 0.0) distance = boundary_distance(x, y, z, 1, 2, Sz);
    else if (v == 0.0) distance = boundary_distance(x, y, z, 0, 2, Sz);
    else if (w == 0.0) distance = boundary_distance(x, y, z, 0, 1, Sz);
    else distance = (fma(u, double(z[0]), fma(v, double(z[1]), w * double(z[2]))) / det) * double(Sz);

    t = FP_T(distance);
    return isfinite(t) && t > m_epsilons.tri_ray_tmin;
}
