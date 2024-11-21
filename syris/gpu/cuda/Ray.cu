#include "Ray.cuh"

__device__ Ray::Ray(const Ray &ray) {
    this->tail = ray.tail;
    this->head = ray.head;
    this->direction = ray.direction;
    this->invDirection = ray.invDirection;
    this->sign[0] = ray.sign[0];
    this->sign[1] = ray.sign[1];
    this->sign[2] = ray.sign[2];
    // printf ("Signs: %d, %d, %d\n", this->sign[0], this->sign[1], this->sign[2]);
};
__device__ Ray::Ray(float4 tail, float4 direction){
    this->tail = tail;
    this->direction = direction;
    this->updateInvDirection();
    this->updateSign();
    // this->print();
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
__device__ bool Ray::intersects (float4 const &minBbox, float4 const &maxBbox, float &tmin, float &tmax) {
    float4 bounds[2];
    bounds[0] = minBbox;
    bounds[1] = maxBbox;

    float tymin, tymax, tzmin, tzmax;

    // printf ("sign: %d %d %d\n", this->sign[0], this->sign[1], this->sign[2]);
    float c = (bounds[this->sign[0]].x - this->tail.x);
    float d = (bounds[1 - this->sign[0]].x - this->tail.x);
    
    tmin = (c == 0) ? 0 : c * this->invDirection.x;
    tmax = (d == 0) ? 0 : d * this->invDirection.x;
    // printf ("%f %f \n", t.x, tmax);

    tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
    tymax = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
    tzmax = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

__device__ bool Ray::intersects (float4 const &minBbox, float4 const &maxBbox) {
    float tmin = -1, tmax = -1;
    // intersects only if box is in front of the ray
    return this->intersects(minBbox, maxBbox, tmin, tmax) && (tmin >= 0.f);
}


__device__ int findLargestComp(float const (&dir)[3]) {
    int kz = 0;

    float max = fabsf(dir[0]);

    for (int i = 1; i < 3; i++) {
        float f = fabsf(dir[i]);

        if (f > max) {
            max = f;
            kz = i;
        }
    }

    return kz;
}

__device__ __forceinline__ float4 rotate2D (const float4 &point) {
    float4 point_star;
    float r = sqrtf(point.x * point.x + point.y * point.y);
    if (point.x != 0) {
        point_star.x = (point.x > 0 ? 1 : -1) * r;
    }
    else {
        point_star.x = (point.y > 0 ? 1 : -1) * r;
    }
    point_star.y = point.z;
    point_star.z = 0.0f;
    point_star.w = 0.0f;
    return point_star;
}

__device__ __forceinline__ bool rayEdgeIntersect (
    const float4 &edge_vertex_1, const float4 &edge_vertex_2, float &t) {
    float x3 = edge_vertex_1.x;
    float y3 = edge_vertex_1.y;
    float x4 = edge_vertex_2.x;
    float y4 = edge_vertex_2.y;

    float y2 = fabsf(y3) > fabsf(y4) ? y3 : y4;

    float det = y2 * (x3 - x4);

    if (det == 0.0f) {
        return false;
    }
    t = (x3 * y4 - x4 * y3) / det * y2;

    float u = x3 * y2 / det;

    float const epsilon = 0.00001f;
    return (u >= 0.0f - epsilon && u <= 1.0f + epsilon);
} 

/**
 * S. Woop et al. 2013, "Watertight Ray/Triangle Intersection"
 */
__device__ bool Ray::intersects(
    float4 const &V1, float4 const &V2, float4 const &V3, float &tmin, float &tmax)
{
    // calculate dimension where the ray direction is maximal
    float Dir[3] = {this->direction.x, this->direction.y, this->direction.z};

    int const kz = findLargestComp (Dir);
    int kx = kz + 1; if (kx == 3) kx = 0;
    int ky = kx + 1; if (ky == 3) ky = 0;

    // swap kx and ky dimensions to preserve winding direction of triangles
    if (Dir[kz] < 0.0f) {
        int temp = kx;
        kx = ky;
        ky = temp;
    }

    /* calculate shear constants */
    float3 S;
    S.z = 1.0f / Dir[kz];
    S.x = Dir[kx] * S.z;
    S.y = Dir[ky] * S.z;

    /* calculate vertices relative to ray origin */
    float OA[3] = {V1.x - tail.x, V1.y - tail.y, V1.z - tail.z};
    float OB[3] = {V2.x - tail.x, V2.y - tail.y, V2.z - tail.z};
    float OC[3] = {V3.x - tail.x, V3.y - tail.y, V3.z - tail.z};

    float mag_OA = sqrtf(OA[0] * OA[0] + OA[1] * OA[1] + OA[2] * OA[2]);
    float mag_OB = sqrtf(OB[0] * OB[0] + OB[1] * OB[1] + OB[2] * OB[2]);
    float mag_OC = sqrtf(OC[0] * OC[0] + OC[1] * OC[1] + OC[2] * OC[2]);
    float mag = 3.0 / (mag_OA + mag_OB + mag_OC);

    /* perform shear and scale of vertices */
    float4 A, B, C;
    A.x = (OA[kx] - S.x * OA[kz]) * mag;
    A.y = (OA[ky] - S.y * OA[kz]) * mag;
    A.z = 0.0f;
    B.x = (OB[kx] - S.x * OB[kz]) * mag;
    B.y = (OB[ky] - S.y * OB[kz]) * mag;
    B.z = 0.0f;
    C.x = (OC[kx] - S.x * OC[kz]) * mag;
    C.y = (OC[ky] - S.y * OC[kz]) * mag;
    C.z = 0.0f;
    // float Ax = (OA[kx] - Sx * OA[kz]) * mag;
    // float Ay = (OA[ky] - Sy * OA[kz]) * mag;
    // float Bx = (OB[kx] - Sx * OB[kz]) * mag;
    // float By = (OB[ky] - Sy * OB[kz]) * mag;
    // float Cx = (OC[kx] - Sx * OC[kz]) * mag;
    // float Cy = (OC[ky] - Sy * OC[kz]) * mag;

    // calculate scaled barycentric coordinates
    float U = C.x * B.y - C.y * B.x;
    float V = A.x * C.y - A.y * C.x;
    float W = B.x * A.y - B.y * A.x;
    // float U = Cx * By - Cy * Bx;
    // float V = Ax * Cy - Ay * Cx;
    // float W = Bx * Ay - By * Ax;
    
    /* fallback to test against edges using double precision  (if float is indeed float) */
    if (U == (float)0.0f || V == (float)0.0f || W == (float)0.0f) {
        U = (double)(C.x * B.y) - (double)(C.y * B.x);
        V = (double)(A.x * C.y) - (double)(A.y * C.x);
        W = (double)(B.x * A.y) - (double)(B.y * A.x);
        // U = (double)Cx * By - (double)Cy * Bx;
        // V = (double)Ax * Cy - (double)Ay * Cx;
        // W = (double)Bx * Ay - (double)By * Ax;
    }

    tmin = INFINITY;
    tmax = -INFINITY;

    float const epsilon = 0.0000001f;
    if ((U < -epsilon || V < -epsilon || W < -epsilon) &&
        (U > epsilon || V > epsilon || W > epsilon)){
        // printf ("U: %f, V: %f, W: %f\n", U, V, W);
        return false;
    }

    /* calculate determinant */
    float det = U + V + W;

    /* Calculates scaled z-coordinate of vertices and uses them to calculate the hit distance. */
    A.z = S.z * OA[kz];
    B.z = S.z * OB[kz];
    C.z = S.z * OC[kz];

    if (det < -epsilon || det > epsilon) {
        float t = U * A.z + V * B.z + W * C.z;
        t /= det;
        tmin = t;
        tmax = t;

        // printf ("classic tmin: %f, tmax: %f\n", tmin, tmax);
        return true;
    }

    float4 A_star = rotate2D(A);
    float4 B_star = rotate2D(B);
    float4 C_star = rotate2D(C);

    float t_ab = INFINITY, t_bc = INFINITY, t_ca = INFINITY;
    bool ab_intersect = rayEdgeIntersect(A_star, B_star, t_ab);
    if (ab_intersect) {
        tmin = t_ab;
        tmax = t_ab;
    }
    bool bc_intersect = rayEdgeIntersect(B_star, C_star, t_bc);
    if (bc_intersect) {
        tmin = fminf(tmin, t_bc);
        tmax = fmaxf(tmax, t_bc);
    }
    bool ca_intersect = rayEdgeIntersect(C_star, A_star, t_ca);
    if (ca_intersect) {
        tmin = fminf(tmin, t_ca);
        tmax = fmaxf(tmax, t_ca);
    }

    if (ab_intersect || bc_intersect || ca_intersect) {
        if (tmin * tmax <= 0) {
            tmin = 0;
            tmax = 0;
        }
        else {
            if (tmin < 0) {
                // swap tmin and tmax
                float temp = tmin;
                tmin = tmax;
                tmax = temp;
            }
        }
        // printf ("co planar tmin: %f, tmax: %f\n", tmin, tmax);
        return true;
    }

    // printf ("end :tmin: %f, tmax: %f\n", tmin, tmax);
    return false;
    // const int det_sign = signbit(det);
    // const float xort_t = xor_signmask(T, det_sign);
    // if (xort_t < 0.0f)
    //     printf ("T: %f\n", T);
    //     return false;

    // // normalize U, V, W, and T
    // const float rcpDet = 1.0f / det;
    // // *u = U*rcpDet;
    // // *v = V*rcpDet;
    // // *w = W*rcpDet;

    // t = T * rcpDet;
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

//     return t >= 0;
// }

// __device__ bool Ray::intersects(float4 const &V1, float4 const &V2, float4 const &V3, float &t) {
//     float tmin = -1, tmax = -1;
//     bool intersects = this->intersects(V1, V2, V3, tmin, tmax) && (tmin >= 0.f);
//     t = intersects ? tmin : -1;
//     return intersects;
// }

// computes the difference of products a*b - c*d using 
// FMA instructions for improved numerical precision
__device__ inline float diff_product(float a, float b, float c, float d) 
{
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

// // http://jcgt.org/published/0002/01/05/
__device__ bool Ray::intersects(
    float4 const &V1_e, float4 const &V2_e, float4 const &V3_e, float &t) {
	// todo: precompute for ray
    float V1[3] = {V1_e.x, V1_e.y, V1_e.z};
    float V2[3] = {V2_e.x, V2_e.y, V2_e.z};
    float V3[3] = {V3_e.x, V3_e.y, V3_e.z};

	int kz = maxDimIndex(abs4(this->direction));
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

    // TODO . make this in ray
    float dir[3] = {this->direction.x, this->direction.y, this->direction.z};

	if (dir[kz] < 0.0f)
	{
		float tmp = kx;
		kx = ky;
		ky = tmp;
	}

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	// todo: end precompute
    const float O[3] = {this->tail.x, this->tail.y, this->tail.z};
	float A[3], B[3], C[3];
    for (int i = 0; i < 3; i++) {
        A[i] = V1[i] - O[i];
        B[i] = V2[i] - O[i];
        C[i] = V3[i] - O[i];
    }
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
    {
        return false;
    }

	float det = U+V+W;

	if (det == 0.0f) 
    {
		return false;
    }

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
		return false;
    }

	const float rcpDet = 1.0f/det;
	// u = U*rcpDet;
	// v = V*rcpDet;
	t = T*rcpDet;
	// sign = det;
	
	// // optionally write out normal (todo: this branch is a performance concern, should probably remove)
	// if (normal)
	// {
	// 	const vec3 ab = b-a;
	// 	const vec3 ac = c-a;

	// 	// calculate normal
	// 	*normal = cross(ab, ac); 
	// }

	return true;
}