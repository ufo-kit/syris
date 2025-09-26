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

    // BUG FIX IS HERE: The check for zero was incorrect.
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
    FP_T4 invDirection = make_float4(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z, 0.0f);
    
    // Create local arrays for indexed access
    const FP_T O[3] = { origin.x, origin.y, origin.z };
    const FP_T D[3] = { direction.x, direction.y, direction.z };
    const FP_T invD[3] = { invDirection.x, invDirection.y, invDirection.z };
    const FP_T bbMin[3] = { scene_min.x, scene_min.y, scene_min.z };
    const FP_T bbMax[3] = { scene_max.x, scene_max.y, scene_max.z };

    // 1. Remap axes based on the ray's dominant direction (Swizzling)
    this->Kz = maxDimIndex(abs4(this->direction));
    this->Kx = this->Kz + 1;
    if (this->Kx == 3)
        this->Kx = 0;
    this->Ky = this->Kx + 1;
    if (this->Ky == 3)
        this->Ky = 0;

    if (D[this->Kz] < FP_CONST(0.0))
    {
        swap<int>(this->Kx, this->Ky);
    }

    this->Sz = FP_CONST(1.0) / D[this->Kz];
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

__device__ bool WatertightRay::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T t_near = FP_CONST(0.0);
    FP_T t_far  = POS_INFINITY;
    return this->intersects(minBbox, maxBbox, t_near, t_far);
}


__device__ bool WatertightRay::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t, FP_T &tmax) const {
    // Calculate vertices relative to ray origin
    const FP_T4 A_t4 = V1 - this->tail;
    const FP_T4 B_t4 = V2 - this->tail;
    const FP_T4 C_t4 = V3 - this->tail;

    // --- Correctly unpack for indexed access ---
    // This is the proper way to allow dynamic component selection via Kx, Ky, Kz.
    const FP_T A[3] = {A_t4.x, A_t4.y, A_t4.z};
    const FP_T B[3] = {B_t4.x, B_t4.y, B_t4.z};
    const FP_T C[3] = {C_t4.x, C_t4.y, C_t4.z};

    // Perform shear and scale of vertices using FMA for precision
    const FP_T Ax = fma(-this->Sx, A[this->Kz], A[this->Kx]);
    const FP_T Ay = fma(-this->Sy, A[this->Kz], A[this->Ky]);
    const FP_T Bx = fma(-this->Sx, B[this->Kz], B[this->Kx]);
    const FP_T By = fma(-this->Sy, B[this->Kz], B[this->Ky]);
    const FP_T Cx = fma(-this->Sx, C[this->Kz], C[this->Kx]);
    const FP_T Cy = fma(-this->Sy, C[this->Kz], C[this->Ky]);

    // Calculate scaled barycentric coordinates
    const FP_T U = fma(Cx, By, -Cy * Bx);
    const FP_T V = fma(Ax, Cy, -Ay * Cx);
    const FP_T W = fma(Bx, Ay, -By * Ax);

    const FP_T det = U + V + W;

    // More Robust Dynamic Epsilon Calculation
    constexpr FP_T gamma_factor = (FP_T)24.0 * EPSILON;
    const FP_T error_bound = gamma_factor * (fabs(U) + fabs(V) + fabs(W));

    // Double Precision Fallback for Degenerate Cases
    if (fabs(det) < error_bound) {
        const double d_Sx = (double)this->Sx, d_Sy = (double)this->Sy;
        const double d_Ax = fma(-d_Sx, (double)A[this->Kz], (double)A[this->Kx]);
        const double d_Ay = fma(-d_Sy, (double)A[this->Kz], (double)A[this->Ky]);
        const double d_Bx = fma(-d_Sx, (double)B[this->Kz], (double)B[this->Kx]);
        const double d_By = fma(-d_Sy, (double)B[this->Kz], (double)B[this->Ky]);
        const double d_Cx = fma(-d_Sx, (double)C[this->Kz], (double)C[this->Kx]);
        const double d_Cy = fma(-d_Sy, (double)C[this->Kz], (double)C[this->Ky]);

        const double d_U = fma(d_Cx, d_By, -d_Cy * d_Bx);
        const double d_V = fma(d_Ax, d_Cy, -d_Ay * d_Cx);
        const double d_W = fma(d_Bx, d_Ay, -d_By * d_Ax);
        const double d_det = d_U + d_V + d_W;

        constexpr double d_gamma_factor = 24.0 * EPSILON;
        const double d_error_bound = d_gamma_factor * (fabs(d_U) + fabs(d_V) + fabs(d_W));
        if (fabs(d_det) < d_error_bound) return false;

        if (d_det > 0.0) {
            if (d_U < 0.0 || d_V < 0.0 || d_W < 0.0) return false;
        } else {
            if (d_U > 0.0 || d_V > 0.0 || d_W > 0.0) return false;
        }

        const double d_Az = (double)this->Sz * (double)A[this->Kz];
        const double d_Bz = (double)this->Sz * (double)B[this->Kz];
        const double d_Cz = (double)this->Sz * (double)C[this->Kz];
        const double t_numerator = fma(d_U, d_Az, fma(d_V, d_Bz, d_W * d_Cz));

        if (copysign(1.0, t_numerator) != copysign(1.0, d_det)) return false;

        t = (FP_T)(t_numerator / d_det);
    } else {
        // Single Precision Path (Common Case)
        if (det > 0.0) {
            if (U < -error_bound || V < -error_bound || W < -error_bound) return false;
        } else {
            if (U > error_bound || V > error_bound || W > error_bound) return false;
        }

        const FP_T Az = this->Sz * A[this->Kz];
        const FP_T Bz = this->Sz * B[this->Kz];
        const FP_T Cz = this->Sz * C[this->Kz];
        const FP_T t_numerator = fma(U, Az, fma(V, Bz, W * Cz));

        if (copysign(FP_CONST(1.0), t_numerator) != copysign(FP_CONST(1.0), det)) return false;

        t = t_numerator / det;
    }

    // Final check for valid intersection range
    constexpr FP_T T_MIN = EPSILON;
    if (t > T_MIN && t < tmax) {
        tmax = t;
        return true;
    }

    return false;
}


__device__ bool WatertightRay::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const
{
    // Set a default t_max to effectively infinity
    FP_T t_max = POS_INFINITY;
    
    // Call the main function to do the actual work
    return this->intersects(V1, V2, V3, t, t_max);
}
