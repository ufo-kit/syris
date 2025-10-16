#include "WatertightRay.cuh"
#include "Commons.cuh"

// Returns the next floating-point number towards +infinity.
__device__ float roundUp_bitwise(float val) {
    if (isinf(val) || isnan(val)) return val;
    int i = __float_as_int(val);
    
    // For positive numbers (and +0.0), incrementing the integer moves away from zero.
    // For negative numbers, the integer representation is inverted, so we decrement to move away from zero.
    if (val >= 0.0f) {
        i++;
    } else { // val < 0.0f
        i--;
    }
    return __int_as_float(i);
}

__device__ float roundDown_bitwise(float val) {
    if (isinf(val) || isnan(val)) return val;
    int i = __float_as_int(val);
    // For positive numbers, we decrement to move towards zero.
    // For negative numbers AND zero, we increment to move towards negative infinity.
    if (val > 0.0f) {
        i--;
    } else { // val <= 0.0f
        i++;
    }
    return __int_as_float(i);
}
__device__ FP_T RoundUp(FP_T a) { return a; }
// __device__ FP_T RoundUp(FP_T a) { return roundUp_bitwise(a); }
__device__ FP_T RoundDown(FP_T a) { return a; }
// __device__ FP_T RoundDown(FP_T a) { return roundDown_bitwise(a); }


// --- WatertightRay Constructor Implementation ---
__device__ WatertightRay::WatertightRay(FP_T4 origin, FP_T4 direction, const FP_T4& scene_min, const FP_T4& scene_max)
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

    // 1. Remap axes based on the ray's dominant direction (Swizzling)
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
    this->Sx = D[this->Kx] * this->Sz;
    this->Sy = D[this->Ky] * this->Sz;

    // 2. Pre-calculate sorted plane indices based on ray direction
    const int near_p[3] = {0, 1, 2}; // Indices for min planes
    const int far_p[3]  = {3, 4, 5}; // Indices for max planes
    this->near_kx_idx = near_p[this->Kx]; this->far_kx_idx = far_p[this->Kx];
    this->near_ky_idx = near_p[this->Ky]; this->far_ky_idx = far_p[this->Ky];
    this->near_kz_idx = near_p[this->Kz]; this->far_kz_idx = far_p[this->Kz];
    if (D[this->Kx] < 0.0f) swap(this->near_kx_idx, this->far_kx_idx);
    if (D[this->Ky] < 0.0f) swap(this->near_ky_idx, this->far_ky_idx);
    if (D[this->Kz] < 0.0f) swap(this->near_kz_idx, this->far_kz_idx);

    // 3. Pre-calculate conservatively rounded reciprocal directions
    this->r_near_kx = RoundDown(RoundDown(invD[this->Kx])); this->r_far_kx = RoundUp(RoundUp(invD[this->Kx]));
    this->r_near_ky = RoundDown(RoundDown(invD[this->Ky])); this->r_far_ky = RoundUp(RoundUp(invD[this->Ky]));
    this->r_near_kz = RoundDown(RoundDown(invD[this->Kz])); this->r_far_kz = RoundUp(RoundUp(invD[this->Kz]));
    
    // 4. Calculate error bounds and corrected origins
    // Use a practical epsilon that won't be lost to 32-bit float rounding.
    // FP_T ROBUST_EPSILON = 0;
    FP_T ROBUST_EPSILON = 5.0f * ldexpf(1.0f, -24);

    const FP_T L[3] = { fabsf(O[0] - bbMin[0]), fabsf(O[1] - bbMin[1]), fabsf(O[2] - bbMin[2]) };
    const FP_T U[3] = { fabsf(O[0] - bbMax[0]), fabsf(O[1] - bbMax[1]), fabsf(O[2] - bbMax[2]) };
    
    const FP_T max_dist[3] = {
        RoundUp(fmaxf(L[0], U[0])),
        RoundUp(fmaxf(L[1], U[1])),
        RoundUp(fmaxf(L[2], U[2]))
    };

    const FP_T Z_max = max_dist[this->Kz];

    const FP_T err_kx = ROBUST_EPSILON * RoundUp(max_dist[this->Kx] + Z_max);
    const FP_T err_ky = ROBUST_EPSILON * RoundUp(max_dist[this->Ky] + Z_max);

    // E_p is error bias for positive-going faces (min planes), E_n for negative-going (max planes)
    const FP_T E_p_kx = ROBUST_EPSILON * RoundUp(L[this->Kx] + Z_max);
    const FP_T E_n_kx = ROBUST_EPSILON * RoundUp(U[this->Kx] + Z_max);
    const FP_T E_p_ky = ROBUST_EPSILON * RoundUp(L[this->Ky] + Z_max);
    const FP_T E_n_ky = ROBUST_EPSILON * RoundUp(U[this->Ky] + Z_max);

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


__device__ bool WatertightRay::intersects(const FP_T4& box_min, const FP_T4& box_max, FP_T tmin, FP_T tmax) const
{
    // Unpack box into a flattened array for indexed access
    const FP_T B[6] = { box_min.x, box_min.y, box_min.z, box_max.x, box_max.y, box_max.z };

    // Use tmin and tmax as the running interval for the intersection
    FP_T t_entry = tmin;
    FP_T t_exit  = tmax;

    // --- Kx Slab ---
    float tNearX = (B[this->near_kx_idx] - this->o_near_kx) * this->r_near_kx;
    float tFarX  = (B[this->far_kx_idx]  - this->o_far_kx)  * this->r_far_kx;
    t_entry = fmaxf(t_entry, tNearX);
    t_exit  = fminf(t_exit,  tFarX);

    // --- Ky Slab ---
    float tNearY = (B[this->near_ky_idx] - this->o_near_ky) * this->r_near_ky;
    float tFarY  = (B[this->far_ky_idx]  - this->o_far_ky)  * this->r_far_ky;
    t_entry = fmaxf(t_entry, tNearY);
    t_exit  = fminf(t_exit,  tFarY);

    // --- Kz Slab ---
    float tNearZ = (B[this->near_kz_idx] - this->o_near_kz) * this->r_near_kz;
    float tFarZ  = (B[this->far_kz_idx]  - this->o_far_kz)  * this->r_far_kz;
    t_entry = fmaxf(t_entry, tNearZ);
    t_exit  = fminf(t_exit,  tFarZ);
    
    // Final check for a valid, overlapping interval
    return t_entry <= t_exit;
}

// __device__ bool WatertightRay::intersects(const FP_T4& box_min, const FP_T4& box_max, FP_T tmin, FP_T tmax) const
// {
//     FP_T4 bounds[2];
//     bounds[0] = box_min;
//     bounds[1] = box_max;

//     FP_T t_lower, t_upper;

//     // X Slab
//     t_lower = (bounds[this->sign[0]].x - this->tail.x) * this->invDirection.x;
//     t_upper = (bounds[1 - this->sign[0]].x - this->tail.x) * this->invDirection.x;
//     tmin = (fmax)(tmin, t_lower);
//     tmax = (fmin)(tmax, t_upper);

//     // Y Slab
//     t_lower = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     t_upper = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     tmin = (fmax)(tmin, t_lower);
//     tmax = (fmin)(tmax, t_upper);

//     // Z Slab
//     t_lower = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     t_upper = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     tmin = (fmax)(tmin, t_lower);
//     tmax = (fmin)(tmax, t_upper);
//     return tmax >= tmin;
// }

__device__ FP_T WatertightRay::point_2_parametric (FP_T4 const &point) const {
    return dot((point - this->tail), this->invDirection);
}

// __device__ bool WatertightRay::intersects(const FP_T4& box_min, const FP_T4& box_max, FP_T tmin, FP_T tmax) const
// {
//     FP_T4 bounds[2];
//     bounds[0] = box_min;
//     bounds[1] = box_max;

//     FP_T tmin_new = (bounds[this->sign[0]].x - this->tail.x) * this->invDirection.x;
//     FP_T tmax_new = (bounds[1-this->sign[0]].x - this->tail.x) * this->invDirection.x;
//     FP_T tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     FP_T tymax = (bounds[1-this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     if ( (tmin_new > tymax) || (tymin > tmax_new) )
//         return false;
//     if (tymin > tmin_new)
//         tmin_new = tymin;
//     if (tymax < tmax_new)
//         tmax_new = tymax;
//     FP_T tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     FP_T tzmax = (bounds[1-this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     if ( (tmin_new > tzmax) || (tzmin > tmax_new) )
//         return false;
//     if (tzmin > tmin_new)
//         tmin_new = tzmin;
//     if (tzmax < tmax_new)
//         tmax_new = tzmax;

//     return ( (tmin < tmax_new) && (tmax > tmin_new) );
// }

__device__ bool WatertightRay::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T t_near = (0.0);
    FP_T t_far  = POS_INFINITY;
    return this->intersects(minBbox, maxBbox, t_near, t_far);
}


__device__ bool WatertightRay::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t, unsigned col, unsigned row) const {
    constexpr FP_T epsilon = ::cuda::std::numeric_limits<FP_T>::epsilon();
    
    #ifdef DEBUG
    bool is_debug = (row == debug_row) && (col == debug_col);
    #endif

    // Calculate vertices relative to ray origin
    const FP_T4 A_t4 = V1 - this->tail;
    const FP_T4 B_t4 = V2 - this->tail;
    const FP_T4 C_t4 = V3 - this->tail;

    // This is the proper way to allow dynamic component selection via Kx, Ky, Kz.
    const FP_T A[3] = {A_t4.x, A_t4.y, A_t4.z};
    const FP_T B[3] = {B_t4.x, B_t4.y, B_t4.z};
    const FP_T C[3] = {C_t4.x, C_t4.y, C_t4.z};

    // Perform shear and scale of vertices using FMA for precision
    const FP_T Ax = (fma)(-Sx, A[Kz], A[Kx]);
    const FP_T Ay = (fma)(-Sy, A[Kz], A[Ky]);
    const FP_T Bx = (fma)(-Sx, B[Kz], B[Kx]);
    const FP_T By = (fma)(-Sy, B[Kz], B[Ky]);
    const FP_T Cx = (fma)(-Sx, C[Kz], C[Kx]);
    const FP_T Cy = (fma)(-Sy, C[Kz], C[Ky]);

    // Calculate scaled barycentric coordinates
    const FP_T U = (fma)(Cx, By, -Cy * Bx);
    const FP_T V = (fma)(Ax, Cy, -Ay * Cx);
    const FP_T W = (fma)(Bx, Ay, -By * Ax);

    const FP_T det = U + V + W;

    // More Robust Dynamic Epsilon Calculation
    constexpr FP_T gamma_factor = 8.0 * epsilon;
    const FP_T error_bound = gamma_factor * (fabs(U) + fabs(V) + fabs(W));

    // Double Precision Fallback for Degenerate Cases
    if ((fabs)(det) <= error_bound) {
        const double d_Sx = (double)Sx, d_Sy = (double)Sy, d_Sz = (double)Sz;
        const double d_Ax = fma(-d_Sx, (double)A[Kz], (double)A[Kx]);
        const double d_Ay = fma(-d_Sy, (double)A[Kz], (double)A[Ky]);
        const double d_Bx = fma(-d_Sx, (double)B[Kz], (double)B[Kx]);
        const double d_By = fma(-d_Sy, (double)B[Kz], (double)B[Ky]);
        const double d_Cx = fma(-d_Sx, (double)C[Kz], (double)C[Kx]);
        const double d_Cy = fma(-d_Sy, (double)C[Kz], (double)C[Ky]);

        const double d_U = fma(d_Cx, d_By, -d_Cy * d_Bx);
        const double d_V = fma(d_Ax, d_Cy, -d_Ay * d_Cx);
        const double d_W = fma(d_Bx, d_Ay, -d_By * d_Ax);
        const double d_det = d_U + d_V + d_W;

        constexpr FP_T d_epsilon = ::cuda::std::numeric_limits<double>::epsilon();
        constexpr double d_gamma_factor = 8.0 * d_epsilon;
        const double d_error_bound = d_gamma_factor * (fabs(d_U) + fabs(d_V) + fabs(d_W));
        if (fabs(d_det) <= d_error_bound) {
            #ifdef DEBUG
            if (is_debug){
                printf("Exit 0\n");
            }
            #endif
            return false;
        }

        bool signs_differ = (det > 0.0f)
                   ? (U < -d_error_bound || V < -d_error_bound || W < -d_error_bound)
                   : (U > d_error_bound || V > d_error_bound || W > d_error_bound);

        if (signs_differ) {
            #ifdef DEBUG
            if (is_debug){
                printf("Exit 1\n");
            }
            #endif
            return false;
        }

        const double d_Az = d_Sz * (double)A[Kz];
        const double d_Bz = d_Sz * (double)B[Kz];
        const double d_Cz = d_Sz * (double)C[Kz];
        const double t_numerator = fma(d_U, d_Az, fma(d_V, d_Bz, d_W * d_Cz));

        if (copysign(1.0, t_numerator) != copysign(1.0, d_det)){
            return false;
        }

        t = (FP_T)(t_numerator / d_det);
    } else {
        bool signs_differ = (det > 0.0f)
                   ? (U < -error_bound || V < -error_bound || W < -error_bound)
                   : (U > error_bound || V > error_bound || W > error_bound);

        if (signs_differ) {
            #ifdef DEBUG
            if (is_debug){
                printf("Exit 3\n"
                       "  tail: (%.8g, %.8g, %.8g)\n"
                       "  direction: (%.8g, %.8g, %.8g)\n"
                       "  V1: (%.8g, %.8g, %.8g)\n"
                       "  V2: (%.8g, %.8g, %.8g)\n"
                       "  V3: (%.8g, %.8g, %.8g)\n"
                       "  det: %.8g\n"
                       "  U: %.8g, V: %.8g, W: %.8g\n"
                       "  Sheared Vertices:\n"
                       "    Ax: %.8g, Ay: %.8g\n"
                       "    Bx: %.8g, By: %.8g\n"
                       "    Cx: %.8g, Cy: %.8g\n",
                       this->tail.x, this->tail.y, this->tail.z,
                       this->direction.x, this->direction.y, this->direction.z,
                       V1.x, V1.y, V1.z,
                       V2.x, V2.y, V2.z,
                       V3.x, V3.y, V3.z,
                       det, U, V, W,
                       Ax, Ay, Bx, By, Cx, Cy);
            }
            #endif
            return false;
        }


        const FP_T Az = Sz * A[Kz];
        const FP_T Bz = Sz * B[Kz];
        const FP_T Cz = Sz * C[Kz];
        const FP_T t_numerator = fma(U, Az, fma(V, Bz, W * Cz));

        if (copysign((1.0), t_numerator) != copysign((1.0), det)){
            #ifdef DEBUG
            printf("Exit 4\n");
            #endif
            return false;
        }

        t = t_numerator / det;
    }

    // Final check for valid intersection range
    constexpr FP_T T_MIN = epsilon;
    if (t > T_MIN) {
        return true;
    }
    return false;
}

// __device__ bool WatertightRay::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const
// {
//     // Set a default t_max to effectively infinity
//     FP_T t_max =  * (1.1);
    
//     if (this->intersects(V1, V2, V3, t, 0) && ())
//     return ;
// }
