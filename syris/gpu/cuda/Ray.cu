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

    this->Kz = max_dim_index(abs(this->direction));
    this->Kx = this->Kz + 1;
    if (this->Kx == 3)
        this->Kx = 0;
    this->Ky = this->Kx + 1;
    if (this->Ky == 3)
        this->Ky = 0;

    if (dir_array[this->Kz] < (0.0))
    {
        swap<int>(this->Kx, this->Ky);
    }

    this->Sz = (1.0) / dir_array[this->Kz];
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
    constexpr FP_T ZERO_VECTOR_EPSILON_SQ = ::cuda::std::numeric_limits<FP_T>::epsilon();
    FP_T mag_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag_sq < ZERO_VECTOR_EPSILON_SQ) {
        return (0.0);
    }    
    return (1.0) / (sqrt)(mag_sq);
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
    // this->invDirection = MAKE_FP_T4(__frcp_rn(this->direction.x), __frcp_rn(this->direction.y), __frcp_rn(this->direction.z), (0.0));
    this->invDirection = MAKE_FP_T4((1.0) / this->direction.x, (1.0) / this->direction.y, (1.0) / this->direction.z, (0.0));
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
    P.w = (1.0);
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

// __device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const
// {
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
    tmin = (fmax)(tmin, t_lower);
    tmax = (fmin)(tmax, t_upper);

    // Y Slab
    t_lower = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
    t_upper = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;
    tmin = (fmax)(tmin, t_lower);
    tmax = (fmin)(tmax, t_upper);

    // Z Slab
    t_lower = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
    t_upper = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;
    tmin = (fmax)(tmin, t_lower);
    tmax = (fmin)(tmax, t_upper);
    return tmax >= tmin;
}

__device__ bool Ray::intersects(FP_T4 const &minBbox, FP_T4 const &maxBbox) const
{
    FP_T t_near = (0.0);
    FP_T t_far  = 1e30;
    return this->intersects(minBbox, maxBbox, t_near, t_far);
}


__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t, FP_T &tmax) const {
    constexpr float epsilon = ::cuda::std::numeric_limits<FP_T>::epsilon();

    FP_T4 E1 = V2 - V1;
    FP_T4 E2 = V3 - V1;
    FP_T4 P = cross4(this->direction, E2);
    FP_T det = dot(E1, P);

    // If determinant is near zero, ray lies in plane of triangle or is parallel
    if (det > -epsilon && det < epsilon) {
        return false;
    }

    FP_T inv_det = (1.0) / det;
    FP_T4 T = this->tail - V1;
    FP_T U = dot(T, P) * inv_det;

    // Check barycentric U coordinate
    if (U < 0.0f || U > 1.0f) {
        return false;
    }

    FP_T4 Q = cross4(T, E1);
    FP_T V = dot(this->direction, Q) * inv_det;

    // Check barycentric V coordinate and U+V sum
    if (V < 0.0f || U + V > 1.0f) {
        return false;
    }

    t = dot(E2, Q) * inv_det;

    // Check if the intersection is in front of the ray and closer than tmax
    if (t > epsilon) {
        tmax = t;
        return true;
    }

    return false;
}

__device__ bool Ray::intersects(FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const
{
    // Set a default t_max to effectively infinity
    FP_T t_max = 1e30;
    
    // Call the main function to do the actual work
    return this->intersects(V1, V2, V3, t, t_max);
}
