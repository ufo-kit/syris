#include "Ray.cuh"

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

    if (dir_array[this->Kz] < 0.0f)
    {
        swap<int>(this->Kx, this->Ky);
    }

    // this->Sz =  __frcp_rn(dir_array[this->Kz]);
    this->Sz = 1.0f / dir_array[this->Kz];
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
    return rnorm3df(v.x, v.y, v.z);
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
    // this->invDirection = make_fp_t4(__frcp_rn(this->direction.x), __frcp_rn(this->direction.y), __frcp_rn(this->direction.z), 0.0f);
    this->invDirection = make_fp_t4(1.0f / this->direction.x, 1.0f / this->direction.y, 1.0f / this->direction.z, 0.0f);
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
    P.w = 1.0f;
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
    float invnorm = rnorm3df(direction.x, direction.y, direction.z);
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

__device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const
{
    // constexpr float EPSILON = 1e-9f;
    constexpr FP_T EPSILON = cuda::std::numeric_limits<FP_T>::epsilon();
    FP_T4 bounds[2];
    bounds[0] = minBbox;
    bounds[1] = maxBbox;
    
    FP_T c = (bounds[this->sign[0]].x - this->tail.x);
    FP_T d = (bounds[1 - this->sign[0]].x - this->tail.x);

    tmin = (is_null<float>(c, EPSILON)) ? 0 : c * this->invDirection.x;
    tmax = (is_null<float>(d, EPSILON)) ? 0 : d * this->invDirection.x;

    FP_T tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
    FP_T tymax = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    FP_T tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
    FP_T tzmax = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

__device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T tmin = 0, tmax = 1e9;
    if (!this->intersects(minBbox, maxBbox, tmin, tmax)){
        return false;
    }

    return tmin >= 0;
}


__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &tmin, FP_T &tmax) const {
    constexpr FP_T EPSILON = cuda::std::numeric_limits<FP_T>::epsilon();

    // Calculate vertices relative to ray origin
    FP_T A[3], B[3], C[3];
    A[0] = V1.x - this->tail.x;
    A[1] = V1.y - this->tail.y;
    A[2] = V1.z - this->tail.z;
    B[0] = V2.x - this->tail.x;
    B[1] = V2.y - this->tail.y;
    B[2] = V2.z - this->tail.z;
    C[0] = V3.x - this->tail.x;
    C[1] = V3.y - this->tail.y;
    C[2] = V3.z - this->tail.z;

    // Perform shear and scale of vertices
    FP_T Ax = A[this->Kx] - this->Sx * A[this->Kz];
    FP_T Ay = A[this->Ky] - this->Sy * A[this->Kz];
    FP_T Bx = B[this->Kx] - this->Sx * B[this->Kz];
    FP_T By = B[this->Ky] - this->Sy * B[this->Kz];
    FP_T Cx = C[this->Kx] - this->Sx * C[this->Kz];
    FP_T Cy = C[this->Ky] - this->Sy * C[this->Kz];

    // Calculate scaled barycentric coordinates
    FP_T U = Cx * By - Cy * Bx;
    FP_T V = Ax * Cy - Ay * Cx;
    FP_T W = Bx * Ay - By * Ax;

    // Fall back to double precision if necessary
    if (is_null<float>(U, EPSILON) || is_null<float>(V, EPSILON) || is_null<float>(W, EPSILON)) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (FP_T)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (FP_T)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (FP_T)(BxAy - ByAx);
    }

    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
        (U > 0.0f || V > 0.0f || W > 0.0f))
        return false;

    // Calculate determinant
    FP_T det = U + V + W;

    if (is_null<float>(det, EPSILON))
        return false;

    // Calculate scaled z-coordinates of vertices
    FP_T Az = this->Sz * A[this->Kz];
    FP_T Bz = this->Sz * B[this->Kz];
    FP_T Cz = this->Sz * C[this->Kz];

    // Calculate the hit distance
    FP_T T = U * Az + V * Bz + W * Cz;

    // Get Signed 0 of det
    int det_sign = sign_mask(det);
    if (xorf(T, det_sign) < 0.0f)
        return false;

    // invmagnitude U, V, W, and T
    // TODO : update this to use double accordingly
    tmin = T / det;
    return true;
}

// __device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &tmin, FP_T &tmax) const
// {
//     constexpr FP_T EPSILON = 1e-7f;
//     FP_T4 const e_1 = V2 - V1;
//     FP_T4 const e_2 = V3 - V1;
//     FP_T4 const dir = this->direction;
//     FP_T4 const tail = this->tail;
//     FP_T4 const P = cross4(dir, e_2);

//     FP_T const det = e_1 * P;

//     if (is_null<float>(det, EPSILON))
//     {
//         return false;
//     }

//     FP_T const inv_det = 1.0f / det;
//     FP_T4 const T = tail - V1;
//     FP_T const u = (T * P) * inv_det;

//     if ((u < 0.0f) || (u > 1.0f))
//     {
//         return false;
//     }

//     FP_T4 const Q = cross4(T, e_1);
//     FP_T const v = (dir * Q) * inv_det;
//     if ((v < 0.0f) || (u + v > 1.0f))
//     {
//         return false;
//     }

//     tmin = (e_2 * Q) * inv_det;

//     return true;
// }

__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const
{
    FP_T tmax = 1e9;
    
    if (!this->intersects(V1, V2, V3, t, tmax)) {
        return false;
    }

    return t >= 0;
}