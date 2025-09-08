#include "Ray.cuh"
#include "Commons.cuh"

__device__ Ray::Ray(const Ray &ray)
{
    this->tail = ray.tail;
    this->head = ray.head;
    this->direction = ray.direction;
    this->invDirection = ray.invDirection;
    this->sign[0] = ray.sign[0];
    this->sign[1] = ray.sign[1];
    this->sign[2] = ray.sign[2];
};

__device__ Ray::Ray(FP_T4 tail, FP_T4 direction)
{
    this->setTail(tail);
    this->setDirection(direction);
    FP_T dir_array[3] = {this->direction.x, this->direction.y, this->direction.z};

    this->Kz = maxDimIndex(abs4(this->direction));
    this->Kx = this->Kz + 1;
    if (this->Kx == 3)
        this->Kx = 0;
    this->Ky = this->Kx + 1;
    if (this->Ky == 3)
        this->Ky = 0;

    if (dir_array[this->Kz] < FP_CONST(0.0))
    {
        swap<int>(this->Kx, this->Ky);
    }

    this->Sz = FP_CONST(1.0) / dir_array[this->Kz];
    this->Sx = dir_array[this->Kx] * this->Sz;
    this->Sy = dir_array[this->Ky] * this->Sz;
};

__device__ void Ray::print() const
{
    printf("Ray: %f %f %f -> %f %f %f\n", this->tail.x, this->tail.y, this->tail.z, this->direction.x, this->direction.y, this->direction.z);
    printf("InvDirection: %f %f %f\n", this->invDirection.x, this->invDirection.y, this->invDirection.z);
    printf("Signs: %d, %d, %d\n", this->sign[0], this->sign[1], this->sign[2]);
}

__device__ FP_T Ray::invmagnitude(FP_T4 &v)
{
    constexpr FP_T ZERO_VECTOR_EPSILON_SQ = FP_CONST(1e-30);
    FP_T mag_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag_sq < ZERO_VECTOR_EPSILON_SQ) {
        return FP_CONST(0.0);
    }    
    return FP_CONST(1.0) / FP_MATH(sqrt)(mag_sq);
}


__device__ void Ray::updateDirection()
{
    this->direction.x = this->head.x - this->tail.x;
    this->direction.y = this->head.y - this->tail.y;
    this->direction.z = this->head.z - this->tail.z;
    FP_T invnorm = this->invmagnitude(this->direction);

    this->direction.x *= invnorm;
    this->direction.y *= invnorm;
    this->direction.z *= invnorm;
}

__device__ void Ray::updateInvDirection()
{
    // this->invDirection = make_fp_t4(__frcp_rn(this->direction.x), __frcp_rn(this->direction.y), __frcp_rn(this->direction.z), FP_CONST(0.0));
    this->invDirection = make_fp_t4(FP_CONST(1.0) / this->direction.x, FP_CONST(1.0) / this->direction.y, FP_CONST(1.0) / this->direction.z, FP_CONST(0.0));
}

__device__ void Ray::updateSign()
{
    this->sign[0] = (this->invDirection.x < 0);
    this->sign[1] = (this->invDirection.y < 0);
    this->sign[2] = (this->invDirection.z < 0);
}

__device__ FP_T4 Ray::computeParametric(FP_T t)
{
    FP_T4 P;
    P.x = this->tail.x + t * this->direction.x;
    P.y = this->tail.y + t * this->direction.y;
    P.z = this->tail.z + t * this->direction.z;
    P.w = FP_CONST(1.0);
    return P;
}

__device__ void Ray::setTail(FP_T4 &tail)
{
    this->tail = tail;
}
__device__ void Ray::setHead(FP_T4 &head)
{
    this->head = head;
}

__device__ void Ray::setDirection(FP_T4 &direction)
{
    FP_T invnorm = this->invmagnitude(direction);
    this->direction = direction * invnorm;
    this->updateInvDirection();
    this->updateSign();
}

__device__ void Ray::setInvDirection(FP_T4 &invDirection)
{
    this->invDirection = invDirection;
}

/*
    Slab method, see
    Marrs, Adam, Peter Shirley, and Ingo Wald, eds. Ray Tracing Gems II:
    Next Generation Real-Time Rendering with DXR, Vulkan, and OptiX.
    Berkeley, CA: Apress, 2021. https://doi.org/10.1007/978-1-4842-7185-8.
*/
// __device__ bool Ray::intersects (FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const {
//     // FP_T4 t_lower = pointwise_product( (minBbox - this->tail) , this->invDirection);
//     // FP_T4 t_upper = pointwise_product( (maxBbox - this->tail) , this->invDirection);

//     // FP_T4 tmins = min4(t_lower, t_upper);
//     // tmins.w = tmin;
//     // FP_T4 tmaxs = max4(t_lower, t_upper);
//     // tmaxs.w = tmax;

//     // FP_T tboxmin = max_component(tmins);
//     // FP_T tboxmax = min_component(tmaxs);

//     // if (tboxmin > tmin) {
//     //     tmin = tboxmin;
//     // }
//     // if (tboxmax < tmax) {
//     //     tmax = tboxmax;
//     // }

//     FP_T txmin, txmax, tymin, tymax, tzmin, tzmax;

//     // Calculate intersection parameters for each slab
//     txmin = (minBbox.x - this->tail.x) * this->invDirection.x;
//     txmax = (maxBbox.x - this->tail.x) * this->invDirection.x;
//     tymin = (minBbox.y - this->tail.y) * this->invDirection.y;
//     tymax = (maxBbox.y - this->tail.y) * this->invDirection.y;
//     tzmin = (minBbox.z - this->tail.z) * this->invDirection.z;
//     tzmax = (maxBbox.z - this->tail.z) * this->invDirection.z;

//     // Correctly order the intersection parameters
//     if (txmin > txmax) swap<FP_T>(txmin, txmax);
//     if (tymin > tymax) swap<FP_T>(tymin, tymax);
//     if (tzmin > tzmax) swap<FP_T>(tzmin, tzmax);

//     // Find the intersection of the slabs
//     tmin = max(max(txmin, tymin), tzmin);
//     tmax = min(min(txmax, tymax), tzmax);
    
//     return true;
// }

// __device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const
// {
//     // constexpr float EPSILON = 1e-9f;
//     constexpr FP_T EPSILON = cuda::std::numeric_limits<FP_T>::epsilon();
//     FP_T4 bounds[2];
//     bounds[0] = minBbox;
//     bounds[1] = maxBbox;
    
//     FP_T c = (bounds[this->sign[0]].x - this->tail.x);
//     FP_T d = (bounds[1 - this->sign[0]].x - this->tail.x);

//     tmin = (is_null<float>(c, EPSILON)) ? 0 : c * this->invDirection.x;
//     tmax = (is_null<float>(d, EPSILON)) ? 0 : d * this->invDirection.x;

//     FP_T tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     FP_T tymax = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;

//     if ((tmin > tymax) || (tymin > tmax))
//         return false;

//     if (tymin > tmin)
//         tmin = tymin;

//     if (tymax < tmax)
//         tmax = tymax;

//     FP_T tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     FP_T tzmax = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;

//     if ((tmin > tzmax) || (tzmin > tmax))
//         return false;

//     if (tzmin > tmin)
//         tmin = tzmin;

//     if (tzmax < tmax)
//         tmax = tzmax;

//     return true;
// }

__device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const
{
    FP_T4 bounds[2];
    bounds[0] = minBbox;
    bounds[1] = maxBbox;

    FP_T t_lower, t_upper;

    // X Slab
    t_lower = (bounds[this->sign[0]].x - this->tail.x) * this->invDirection.x;
    t_upper = (bounds[1 - this->sign[0]].x - this->tail.x) * this->invDirection.x;
    tmin = FP_MATH(fmax)(tmin, t_lower);
    tmax = FP_MATH(fmin)(tmax, t_upper);

    // Y Slab
    t_lower = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
    t_upper = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;
    tmin = FP_MATH(fmax)(tmin, t_lower);
    tmax = FP_MATH(fmin)(tmax, t_upper);

    // Z Slab
    t_lower = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
    t_upper = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;
    tmin = FP_MATH(fmax)(tmin, t_lower);
    tmax = FP_MATH(fmin)(tmax, t_upper);
    return tmax >= tmin;
}

__device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T t_near = FP_CONST(0.0);
    FP_T t_far  = cuda::std::numeric_limits<FP_T>::max();
    return this->intersects(minBbox, maxBbox, t_near, t_far);
}



// __device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &tmin, FP_T &tmax) const {
//     constexpr FP_T EPSILON = cuda::std::numeric_limits<FP_T>::epsilon();

//     // Calculate vertices relative to ray origin
//     FP_T A[3], B[3], C[3];
//     A[0] = V1.x - this->tail.x;
//     A[1] = V1.y - this->tail.y;
//     A[2] = V1.z - this->tail.z;
//     B[0] = V2.x - this->tail.x;
//     B[1] = V2.y - this->tail.y;
//     B[2] = V2.z - this->tail.z;
//     C[0] = V3.x - this->tail.x;
//     C[1] = V3.y - this->tail.y;
//     C[2] = V3.z - this->tail.z;

//     // Perform shear and scale of vertices
//     FP_T Ax = A[this->Kx] - this->Sx * A[this->Kz];
//     FP_T Ay = A[this->Ky] - this->Sy * A[this->Kz];
//     FP_T Bx = B[this->Kx] - this->Sx * B[this->Kz];
//     FP_T By = B[this->Ky] - this->Sy * B[this->Kz];
//     FP_T Cx = C[this->Kx] - this->Sx * C[this->Kz];
//     FP_T Cy = C[this->Ky] - this->Sy * C[this->Kz];

//     // Calculate scaled barycentric coordinates
//     FP_T U = Cx * By - Cy * Bx;
//     FP_T V = Ax * Cy - Ay * Cx;
//     FP_T W = Bx * Ay - By * Ax;

//     // Fall back to double precision if necessary
//     if (is_null<float>(U, EPSILON) || is_null<float>(V, EPSILON) || is_null<float>(W, EPSILON)) {
//         double CxBy = (double)Cx * (double)By;
//         double CyBx = (double)Cy * (double)Bx;
//         U = (FP_T)(CxBy - CyBx);
//         double AxCy = (double)Ax * (double)Cy;
//         double AyCx = (double)Ay * (double)Cx;
//         V = (FP_T)(AxCy - AyCx);
//         double BxAy = (double)Bx * (double)Ay;
//         double ByAx = (double)By * (double)Ax;
//         W = (FP_T)(BxAy - ByAx);
//     }

//     if ((U < FP_CONST(0.0) || V < FP_CONST(0.0) || W < FP_CONST(0.0)) &&
//         (U > FP_CONST(0.0) || V > FP_CONST(0.0) || W > FP_CONST(0.0)))
//         return false;

//     // Calculate determinant
//     FP_T det = U + V + W;

//     if (is_null<float>(det, EPSILON))
//         return false;

//     // Calculate scaled z-coordinates of vertices
//     FP_T Az = this->Sz * A[this->Kz];
//     FP_T Bz = this->Sz * B[this->Kz];
//     FP_T Cz = this->Sz * C[this->Kz];

//     // Calculate the hit distance
//     FP_T T = U * Az + V * Bz + W * Cz;

//     // Get Signed 0 of det
//     int det_sign = sign_mask(det);
//     if (xorf(T, det_sign) < FP_CONST(0.0))
//         return false;

//     // invmagnitude U, V, W, and T
//     // TODO : update this to use double accordingly
//     tmin = T / det;
//     return true;
// }

__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t, FP_T &tmax) const {
    // Calculate vertices relative to ray origin
    FP_T4 A_t4 = V1 - this->tail;
    FP_T4 B_t4 = V2 - this->tail;
    FP_T4 C_t4 = V3 - this->tail;

    FP_T A[3] = {A_t4.x, A_t4.y, A_t4.z};
    FP_T B[3] = {B_t4.x, B_t4.y, B_t4.z};
    FP_T C[3] = {C_t4.x, C_t4.y, C_t4.z};

    // Perform shear and scale of vertices using FP_MATH(FMA) for better precision
    FP_T Ax = FP_MATH(fma)(-this->Sx, A[this->Kz], A[this->Kx]);
    FP_T Ay = FP_MATH(fma)(-this->Sy, A[this->Kz], A[this->Ky]);
    FP_T Bx = FP_MATH(fma)(-this->Sx, B[this->Kz], B[this->Kx]);
    FP_T By = FP_MATH(fma)(-this->Sy, B[this->Kz], B[this->Ky]);
    FP_T Cx = FP_MATH(fma)(-this->Sx, C[this->Kz], C[this->Kx]);
    FP_T Cy = FP_MATH(fma)(-this->Sy, C[this->Kz], C[this->Ky]);

    // Calculate scaled barycentric coordinates using FP_MATH(FMA)
    FP_T U = FP_MATH(fma)(Cx, By, -Cy * Bx);
    FP_T V = FP_MATH(fma)(Ax, Cy, -Ay * Cx);
    FP_T W = FP_MATH(fma)(Bx, Ay, -By * Ax);

    // Calculate determinant
    FP_T det = U + V + W;

    // --- Dynamic Epsilon Calculation ---
    // Calculate a conservative error bound for the determinant calculation.
    // The number of floating point operations is ~7 to get to U,V,W.
    // A gamma factor accounts for error propagation. gamma(n) = (n*eps)/(1-n*eps).
    // For FP_T = float, a factor of 8*epsilon is a reasonable starting point.
    constexpr FP_T gamma = (FP_T)8 * cuda::std::numeric_limits<FP_T>::epsilon();
    FP_T error_bound = gamma * (fabs(U) + fabs(V) + fabs(W));

    // Fall back to double precision if the single-precision result is close to the error margin.
    if (fabs(det) < error_bound) {
        // Fallback to double precision for higher accuracy
        double d_Sx = (double)this->Sx, d_Sy = (double)this->Sy;
        double d_Ax = FP_MATH(fma)(-d_Sx, (double)A[this->Kz], (double)A[this->Kx]);
        double d_Ay = FP_MATH(fma)(-d_Sy, (double)A[this->Kz], (double)A[this->Ky]);
        double d_Bx = FP_MATH(fma)(-d_Sx, (double)B[this->Kz], (double)B[this->Kx]);
        double d_By = FP_MATH(fma)(-d_Sy, (double)B[this->Kz], (double)B[this->Ky]);
        double d_Cx = FP_MATH(fma)(-d_Sx, (double)C[this->Kz], (double)C[this->Kx]);
        double d_Cy = FP_MATH(fma)(-d_Sy, (double)C[this->Kz], (double)C[this->Ky]);
        double d_U = FP_MATH(fma)(d_Cx, d_By, -d_Cy * d_Bx);
        double d_V = FP_MATH(fma)(d_Ax, d_Cy, -d_Ay * d_Cx);
        double d_W = FP_MATH(fma)(d_Bx, d_Ay, -d_By * d_Ax);
        double d_det = d_U + d_V + d_W;

        if (fabs(d_det) < cuda::std::numeric_limits<double>::epsilon()) return false;

        // Check barycentric coordinates against a zero bound in double precision
        if ((d_U < 0.0 || d_V < 0.0 || d_W < 0.0) && (d_U > 0.0 || d_V > 0.0 || d_W > 0.0)) return false;

        double d_Az = (double)this->Sz * (double)A[this->Kz];
        double d_Bz = (double)this->Sz * (double)B[this->Kz];
        double d_Cz = (double)this->Sz * (double)C[this->Kz];
        double d_T = FP_MATH(fma)(d_U, d_Az, FP_MATH(fma)(d_V, d_Bz, d_W * d_Cz));

        if (FP_MATH(copysign)(1.0, d_T) != FP_MATH(copysign)(1.0, d_det)) return false;

        t = (FP_T)(d_T / d_det);
        return true;
    }


    // Check if barycentric coordinates have the same sign, accounting for error
    if ((U < -error_bound || V < -error_bound || W < -error_bound) &&
        (U > error_bound || V > error_bound || W > error_bound))
        return false;

    // Calculate scaled z-coordinates of vertices
    FP_T Az = this->Sz * A[this->Kz];
    FP_T Bz = this->Sz * B[this->Kz];
    FP_T Cz = this->Sz * C[this->Kz];

    // Calculate the hit distance using FP_MATH(FMA)
    FP_T T = FP_MATH(fma)(U, Az, FP_MATH(fma)(V, Bz, W * Cz));

    // Check if the intersection is in front of the ray, considering the sign of det
    // xorf is a bitwise trick; a standard comparison is often clearer and just as fast.
    if (FP_MATH(copysign)(FP_CONST(1.0), T) != FP_MATH(copysign)(FP_CONST(1.0), det))
        return false;

    // Calculate final intersection distance
    t = T / det;
    
    constexpr FP_T T_MIN = cuda::std::numeric_limits<FP_T>::epsilon();
    if (t > T_MIN && t < tmax) {
        tmax = t; // Update the max t-value for future tests
        return true;
    }

    return false;
}

__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const
{
    // Set a default t_max to effectively infinity
    FP_T t_max = cuda::std::numeric_limits<FP_T>::max();
    
    // Call the main function to do the actual work
    return this->intersects(V1, V2, V3, t, t_max);
}
