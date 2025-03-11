#include "Ray.cuh"

__device__ Ray::Ray(const Ray &ray) {
    this->tail = ray.tail;
    this->head = ray.head;
    this->direction = ray.direction;
    this->invDirection = ray.invDirection;
    this->sign[0] = ray.sign[0];
    this->sign[1] = ray.sign[1];
    this->sign[2] = ray.sign[2];
};
__device__ Ray::Ray(float4 tail, float4 direction){
    this->tail = tail;
    this->direction = direction;
    this->updateInvDirection();
    this->updateSign();
    float dir_array[3] = {this->direction.x, this->direction.y, this->direction.z};

    this->Kz = maxDimIndex(abs4(this->direction));
    this->Kx = this->Kz + 1; if (this->Kx == 3) this->Kx = 0;
    this->Ky = this->Kx + 1; if (this->Ky == 3) this->Ky = 0;

    if (dir_array[this->Kz] < 0.0f) {
        swap(this->Kx, this->Ky);
    }

    this->Sz = 1.0f / dir_array[this->Kz];
    this->Sx = dir_array[this->Kx] * this->Sz;
    this->Sy = dir_array[this->Ky] * this->Sz;
};

__device__ void Ray::print() {
    printf ("Ray: %f %f %f -> %f %f %f\n", this->tail.x, this->tail.y, this->tail.z, this->direction.x, this->direction.y, this->direction.z);
    printf ("InvDirection: %f %f %f\n", this->invDirection.x, this->invDirection.y, this->invDirection.z);
    printf ("Signs: %d, %d, %d\n", this->sign[0], this->sign[1], this->sign[2]);
}

__device__ float Ray::normalize(float4 &v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ void Ray::updateDirection() {
    float norm = this->normalize(this->direction);
    this->direction.x = this->head.x - this->tail.x;
    this->direction.y = this->head.y - this->tail.y;
    this->direction.z = this->head.z - this->tail.z;
    
    this->direction.x /= norm;
    this->direction.y /= norm;
    this->direction.z /= norm;
}

__device__ void Ray::updateInvDirection() {
    this->invDirection = make_float4(1.0f / this->direction.x, 1.0f / this->direction.y, 1.0f / this->direction.z, 0.0f);
}

__device__ void Ray::updateSign() {
    this->sign[0] = (this->invDirection.x < 0);
    this->sign[1] = (this->invDirection.y < 0);
    this->sign[2] = (this->invDirection.z < 0);
}

__device__ float4 Ray::computeParametric(float t) {
    float4 P;
    P.x = this->tail.x + t * this->direction.x;
    P.y = this->tail.y + t * this->direction.y;
    P.z = this->tail.z + t * this->direction.z;
    P.w = 1.0f;
    return P;
}

__device__ void Ray::setTail(float4 &tail) {
    this->tail = tail;
}
__device__ void Ray::setHead(float4 &head) {
    this->head = head;
}

__device__ void Ray::setDirection(float4 &direction) {
    this->direction = direction;
    this->updateInvDirection();
    this->updateSign();
}

__device__ void Ray::setInvDirection(float4 &invDirection) {
    this->invDirection = invDirection;
}

/**
 * Ray-box intersection
 */
// __device__ bool Ray::intersects (float4 const &minBbox, float4 const &maxBbox, float &tmin, float &tmax) {
//     float4 bounds[2];
//     bounds[0] = minBbox;
//     bounds[1] = maxBbox;

//     float tymin, tymax, tzmin, tzmax;

//     // printf ("sign: %d %d %d\n", this->sign[0], this->sign[1], this->sign[2]);
//     float c = (bounds[this->sign[0]].x - this->tail.x);
//     float d = (bounds[1 - this->sign[0]].x - this->tail.x);
    
//     tmin = (c == 0) ? 0 : c * this->invDirection.x;
//     tmax = (d == 0) ? 0 : d * this->invDirection.x;
//     // printf ("%f %f \n", t.x, tmax);

//     tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
//     tymax = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;

//     if ((tmin > tymax) || (tymin > tmax))
//         return false;

//     if (tymin > tmin)
//         tmin = tymin;

//     if (tymax < tmax)
//         tmax = tymax;

//     tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
//     tzmax = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;

//     if ((tmin > tzmax) || (tzmin > tmax))
//         return false;

//     if (tzmin > tmin)
//         tmin = tzmin;

//     if (tzmax < tmax)
//         tmax = tzmax;

//     return true;
// }


/*
    Slab method, see
    Marrs, Adam, Peter Shirley, and Ingo Wald, eds. Ray Tracing Gems II: 
    Next Generation Real-Time Rendering with DXR, Vulkan, and OptiX. 
    Berkeley, CA: Apress, 2021. https://doi.org/10.1007/978-1-4842-7185-8.
*/
__device__ bool Ray::intersects (float4 const &minBbox, float4 const &maxBbox, float &tmin, float &tmax) {
    float4 t_lower = (minBbox - this->tail) * this->invDirection;
    float4 t_upper = (maxBbox - this->tail) * this->invDirection;

    float4 tmins = min4(t_lower, t_upper);
    tmins.w = tmin;
    float4 tmaxs = max4(t_lower, t_upper);
    tmaxs.w = tmax;

    float tboxmin = max_component(tmins);
    float tboxmax = min_component(tmaxs);

    return tboxmin <= tboxmax;
}

__device__ bool Ray::intersects (float4 const &minBbox, float4 const &maxBbox) {
    float tmin = 0, tmax = INFINITY;
    // intersects only if box is in front of the ray
    return this->intersects(minBbox, maxBbox, tmin, tmax);
}

// // Both the ray and the triangle were transformed beforehand
// // so that the ray is a unit vector along the z-axis (0,0,1),
// // and the triangle is transformed with the same matrix
// // (which is M in the paper). This function is called only
// // when the ray is co-planar to the triangle (with the
// // determinant being zero). The rotation by this function is
// // to prepare for the ray-edge intersection calculations in 2D.
// // The rotation is around the z-axis. For any point after
// // the rotation, its new x* equals its original length
// // with the correct sign, and the new y* = z. The current
// // implementation avoids explicitly defining rotation angles
// // and directions. The following ray-edge intersection will
// // be in the x*-y* plane.
// __forceinline__ float4 rotate2D(float *point) {
//     float4 point_star;
//     float r = sqrtf(point[0] * point[0] + point[1] * point[1]);
//     if (point[0] != 0) {
//         point_star.x = (point[0] > 0 ? 1 : -1) * r;
//     }
//     else {
//         point_star.x = (point[1] > 0 ? 1 : -1) * r;
//     }
//     point_star.y = point[2];
//     point_star.z = 0.f;
//     return point_star;
// }

// // The function is for ray-edge intersection
// // with the rotated ray along the z-axis and
// // the transformed and rotated triangle edges
// // The algorithm is described in
// // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
// __forceinline__ bool rayEdgeIntersect(
//     float4 const &edge_vertex_1,
//     float4 const &edge_vertex_2,
//     float &t)
// {
//   float x3 = edge_vertex_1.x;
//   float y3 = edge_vertex_1.y;
//   float x4 = edge_vertex_2.x;
//   float y4 = edge_vertex_2.y;

//   float y2 = fabsf(y3) > fabsf(y4) ? y3 : y4;

//   float det = y2 * (x3 - x4);

//   //  the ray is parallel to the edge if det == 0.0
//   //  When the ray overlaps the edge (x3==x4==0.0), it also returns false,
//   //  and the intersection will be captured by the other two edges.
//   if (det == 0) {
//     return false;
//   }

//   t = (x3 * y4 - x4 * y3) / det * y2;

//   float u = x3 * y2 / det;

//   float const epsilon = 0.00001f;
//   return (u >= -epsilon && u <= 1 + epsilon);
// }

__device__ bool Ray::intersects(float4 const &V1, float4 const &V2, float4 const &V3, float &tmin, float &tmax) {
    // Calculate vertices relative to ray origin
    float A[3], B[3], C[3];
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
    float Ax = A[this->Kx] - this->Sx * A[this->Kz];
    float Ay = A[this->Ky] - this->Sy * A[this->Kz];
    float Bx = B[this->Kx] - this->Sx * B[this->Kz];
    float By = B[this->Ky] - this->Sy * B[this->Kz];
    float Cx = C[this->Kx] - this->Sx * C[this->Kz];
    float Cy = C[this->Ky] - this->Sy * C[this->Kz];

    // Calculate scaled barycentric coordinates
    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    // Fall back to double precision if necessary
    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
        (U > 0.0f || V > 0.0f || W > 0.0f))
        return false;

    // Calculate determinant
    float det = U + V + W;

    if (det == 0.0f)
        return false;

    // Calculate scaled z-coordinates of vertices
    float Az = this->Sz * A[this->Kz];
    float Bz = this->Sz * B[this->Kz];
    float Cz = this->Sz * C[this->Kz];

    // Calculate the hit distance
    float T = U * Az + V * Bz + W * Cz;

    // Get Signed 0 of det
    int det_sign = sign_mask(det);
    if (xorf(T, det_sign) < 0.0f)
        return false;

    // Normalize U, V, W, and T
    float rcpDet = 1.0f / det;
    tmin = T * rcpDet;
    return true;
}

// __device__ bool Ray::intersects(
//     float4 const &V1, float4 const &V2, float4 const &V3, float &t)
// {
//     float4 e_1, e_2, P, Q, T;
//     float det, inv_det, u, v;

//     e_1 = sub4 (V2, V1);
//     e_2 = sub4 (V3, V1);
//     P = cross4 (this->direction, e_2);

//     det = dot4 (e_1, P);
    
//     if (det == 0) {
//         return false;
//     }

//     inv_det = 1 / det;
//     T = sub4(this->tail, V1);
//     u = dot4(T, P) * inv_det;
//     if (u < 0 || u > 1) {
//         return false;
//     }

//     Q = cross4 (T, e_1);
//     v = dot4 (this->direction, Q) * inv_det;
//     if (v < 0 || u + v > 1) {
//         return false;
//     }

//     t = dot4 (e_2, Q) * inv_det;

//     return t > 0;
// }

__device__ bool Ray::intersects(float4 const &V1, float4 const &V2, float4 const &V3, float &t) {
    float tmin;
    float tmax;
    return this->intersects(V1, V2, V3, t, tmax) && (t >= 0.f);
}

// __device__ bool Ray::intersects(float4 const &V1, float4 const &V2, float4 const &V3, float &t) {
//     float tmin = -1, tmax = -1;
//     bool intersects = this->intersects(V1, V2, V3, tmin, tmax) && (tmin >= 0.f);
//     t = intersects ? tmin : -1;
//     return intersects;
// }

// // computes the difference of products a*b - c*d using 
// // FMA instructions for improved numerical precision
// __device__ inline float diff_product(float a, float b, float c, float d) 
// {
//     float cd = c * d;
//     float diff = fmaf(a, b, -cd);
//     float error = fmaf(-c, d, cd);

//     return diff + error;
// }

// // // http://jcgt.org/published/0002/01/05/
// __device__ bool Ray::intersects(
//     float4 const &V1_e, float4 const &V2_e, float4 const &V3_e, float &t) {
// 	// todo: precompute for ray
//     float V1[3] = {V1_e.x, V1_e.y, V1_e.z};
//     float V2[3] = {V2_e.x, V2_e.y, V2_e.z};
//     float V3[3] = {V3_e.x, V3_e.y, V3_e.z};

// 	int kz = maxDimIndex(abs4(this->direction));
// 	int kx = kz+1; if (kx == 3) kx = 0;
// 	int ky = kx+1; if (ky == 3) ky = 0;

//     // TODO . make this in ray
//     float dir[3] = {this->direction.x, this->direction.y, this->direction.z};

// 	if (dir[kz] < 0.0f)
// 	{
// 		float tmp = kx;
// 		kx = ky;
// 		ky = tmp;
// 	}

// 	float Sx = dir[kx]/dir[kz];
// 	float Sy = dir[ky]/dir[kz];
// 	float Sz = 1.0f/dir[kz];

// 	// todo: end precompute
//     const float O[3] = {this->tail.x, this->tail.y, this->tail.z};
// 	float A[3], B[3], C[3];
//     for (int i = 0; i < 3; i++) {
//         A[i] = V1[i] - O[i];
//         B[i] = V2[i] - O[i];
//         C[i] = V3[i] - O[i];
//     }
	
// 	const float Ax = A[kx] - Sx*A[kz];
// 	const float Ay = A[ky] - Sy*A[kz];
// 	const float Bx = B[kx] - Sx*B[kz];
// 	const float By = B[ky] - Sy*B[kz];
// 	const float Cx = C[kx] - Sx*C[kz];
// 	const float Cy = C[ky] - Sy*C[kz];
		
//     float U = diff_product(Cx, By, Cy, Bx);
//     float V = diff_product(Ax, Cy, Ay, Cx);
//     float W = diff_product(Bx, Ay, By, Ax);

// 	if (U == 0.0f || V == 0.0f || W == 0.0f) 
// 	{
// 		double CxBy = (double)Cx*(double)By;
// 		double CyBx = (double)Cy*(double)Bx;
// 		U = (float)(CxBy - CyBx);
// 		double AxCy = (double)Ax*(double)Cy;
// 		double AyCx = (double)Ay*(double)Cx;
// 		V = (float)(AxCy - AyCx);
// 		double BxAy = (double)Bx*(double)Ay;
// 		double ByAx = (double)By*(double)Ax;
// 		W = (float)(BxAy - ByAx);
// 	}

// 	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
//     {
//         return false;
//     }

// 	float det = U+V+W;

// 	if (det == 0.0f) 
//     {
// 		return false;
//     }

// 	const float Az = Sz*A[kz];
// 	const float Bz = Sz*B[kz];
// 	const float Cz = Sz*C[kz];
// 	const float T = U*Az + V*Bz + W*Cz;

// 	int det_sign = sign_mask(det);
// 	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
//     {
// 		return false;
//     }

// 	const float rcpDet = 1.0f/det;
// 	// u = U*rcpDet;
// 	// v = V*rcpDet;
// 	t = T*rcpDet;
// 	// sign = det;
	
// 	// // optionally write out normal (todo: this branch is a performance concern, should probably remove)
// 	// if (normal)
// 	// {
// 	// 	const vec3 ab = b-a;
// 	// 	const vec3 ac = c-a;

// 	// 	// calculate normal
// 	// 	*normal = cross(ab, ac); 
// 	// }

// 	return true;
// }