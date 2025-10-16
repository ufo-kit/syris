#include "Commons.cuh"

#define MAX_INTERSECTIONS 200

/*
 * Moeller-Trumbore intersection algorithm
 */
__device__ FP_T compute_intersection_point (
    const FP_T3& V_1,
    const FP_T3& V_2,
    const FP_T3& V_3,
    const FP_T3& O,  
    const FP_T3& D
)
{
    constexpr FP_T epsilon = ::cuda::std::numeric_limits<FP_T>::epsilon();

    FP_T3 e_1, e_2, P, Q, T;
    FP_T det, inv_det, u, v, t;

    e_1 = V_2 - V_1;
    e_2 = V_3 - V_1;
    P = cross3 (D, e_2);
    det = dot (e_1, P);

    if (det > -epsilon && det < epsilon) {
        return -1.0;
    }

    inv_det = 1.0f / det;
    T = O - V_1;
    u = dot (T, P) * inv_det;

    if (u < 0.0f || u > 1.0f) {
        return -1.0;
    }

    Q = cross3 (T, e_1);
    v = dot (D, Q) * inv_det;

    if (v < 0 || u + v > 1) {
        return -1.0;
    }

    t = dot (e_2, Q) * inv_det;

    if (t > epsilon) {
        return t;
    }

    return -1.0;
}

// __device__ FP_T compute_intersection_point (
//     const FP_T3& V_1,
//     const FP_T3& V_2,
//     const FP_T3& V_3,
//     const FP_T3& O,
//     const FP_T3& D,
//     const int idx, const int idy
// ) {
//     constexpr FP_T epsilon = ::cuda::std::numeric_limits<FP_T>::epsilon();

//     // --- Create a single boolean to check if this is the target thread ---
//     const bool is_debug_thread = (idx == DEBUG_X_ID && idy == DEBUG_Y_ID);

//     int Kz = 2;
//     int Kx = 0;
//     int Ky = 1;

//     const FP_T D_arr[3] = { D.x, D.y, D.z };

//     FP_T Sz = (1.0) / D_arr[Kz];
//     FP_T Sx = D_arr[Kx] * Sz;
//     FP_T Sy = D_arr[Ky] * Sz;

//     // Calculate vertices relative to ray origin
//     const FP_T3 A_t4 = V_1 - O;
//     const FP_T3 B_t4 = V_2 - O;
//     const FP_T3 C_t4 = V_3 - O;

//     // This is the proper way to allow dynamic component selection via Kx, Ky, Kz.
//     const FP_T A[3] = {A_t4.x, A_t4.y, A_t4.z};
//     const FP_T B[3] = {B_t4.x, B_t4.y, B_t4.z};
//     const FP_T C[3] = {C_t4.x, C_t4.y, C_t4.z};

//     // Perform shear and scale of vertices using FMA for precision
//     const FP_T Ax = (fma)(-Sx, A[Kz], A[Kx]);
//     const FP_T Ay = (fma)(-Sy, A[Kz], A[Ky]);
//     const FP_T Bx = (fma)(-Sx, B[Kz], B[Kx]);
//     const FP_T By = (fma)(-Sy, B[Kz], B[Ky]);
//     const FP_T Cx = (fma)(-Sx, C[Kz], C[Kx]);
//     const FP_T Cy = (fma)(-Sy, C[Kz], C[Ky]);

//     // Calculate scaled barycentric coordinates
//     const FP_T U = (fma)(Cx, By, -Cy * Bx);
//     const FP_T V = (fma)(Ax, Cy, -Ay * Cx);
//     const FP_T W = (fma)(Bx, Ay, -By * Ax);

//     const FP_T det = U + V + W;

//     // More Robust Dynamic Epsilon Calculation
//     constexpr FP_T gamma_factor = (FP_T)24.0 * epsilon;
//     const FP_T error_bound = gamma_factor * (fabs(U) + fabs(V) + fabs(W));
//     float t = -1;

//     // Double Precision Fallback for Degenerate Cases
//     if ((fabs)(det) <= error_bound) {
//         const double d_Sx = (double)Sx, d_Sy = (double)Sy;
//         const double d_Ax = fma(-d_Sx, (double)A[Kz], (double)A[Kx]);
//         const double d_Ay = fma(-d_Sy, (double)A[Kz], (double)A[Ky]);
//         const double d_Bx = fma(-d_Sx, (double)B[Kz], (double)B[Kx]);
//         const double d_By = fma(-d_Sy, (double)B[Kz], (double)B[Ky]);
//         const double d_Cx = fma(-d_Sx, (double)C[Kz], (double)C[Kx]);
//         const double d_Cy = fma(-d_Sy, (double)C[Kz], (double)C[Ky]);

//         const double d_U = fma(d_Cx, d_By, -d_Cy * d_Bx);
//         const double d_V = fma(d_Ax, d_Cy, -d_Ay * d_Cx);
//         const double d_W = fma(d_Bx, d_Ay, -d_By * d_Ax);
//         const double d_det = d_U + d_V + d_W;

//         constexpr double d_gamma_factor = 1e-15;
//         const double d_error_bound = d_gamma_factor * (fabs(d_U) + fabs(d_V) + fabs(d_W));
//         if (fabs(d_det) < d_error_bound) {
//             return -1;
//         }

//         bool signs_differ = (d_det > 0.0f) ? (d_U < 0.0f || d_V < 0.0f || d_W < 0.0f)
//                                   : (d_U > 0.0f || d_V > 0.0f || d_W > 0.0f);

//         if (signs_differ) {
//             return -1;
//         }

//         const double d_Az = (double)Sz * (double)A[Kz];
//         const double d_Bz = (double)Sz * (double)B[Kz];
//         const double d_Cz = (double)Sz * (double)C[Kz];
//         const double t_numerator = fma(d_U, d_Az, fma(d_V, d_Bz, d_W * d_Cz));

//         if (copysign(1.0, t_numerator) != copysign(1.0, d_det)){
//             return -1;
//         }

//         t = (FP_T)(t_numerator / d_det);

//     } else {
//         // Single Precision Path (Common Case)
//         if ((fabs)(det) < epsilon) {
//             return -1;
//         }

//         bool signs_differ = (det > 0.0f) ? (U < 0.0f || V < 0.0f || W < 0.0f)
//                                   : (U > 0.0f || V > 0.0f || W > 0.0f);

//         if (signs_differ) {
//             return -1;
//         }


//         const FP_T Az = Sz * A[Kz];
//         const FP_T Bz = Sz * B[Kz];
//         const FP_T Cz = Sz * C[Kz];
//         const FP_T t_numerator = fma(U, Az, fma(V, Bz, W * Cz));

//         if (copysign((1.0), t_numerator) != copysign((1.0), det)){
//             return -1;
//         }

//         t = t_numerator / det;
//     }

//     // Final check for valid intersection range
//     constexpr FP_T T_MIN = epsilon;
//     if (t > T_MIN) {
//         return t;
//     }

//     return -1;
// }

__device__ int find_leftmost (FP_T3 *v_1,
                   FP_T3 *v_2,
                   FP_T3 *v_3,
                   int x_0,
                   int x_1,
                   FP_T value)
{
    int i;

    while (x_0 <= x_1) {
        i = (x_0 + x_1) / 2;
        if (v_3[i].x == value) {
            while (v_3[i].x == value) {
                i--;
            }
            return i;
        } else if (v_3[i].x < value) {
            x_0 = i + 1;
        } else {
            x_1 = i - 1;
        }
    }

    return i;
}

__device__ int compute_intersections (
    FP_T3 *v_1,
    FP_T3 *v_2,
    FP_T3 *v_3,
    int num_triangles,
    FP_T3 O,
    FP_T3 D,
    FP_T max_dx,
    FP_T *intersections,
    const int idx, const int idy // ADDED FOR DEBUGGING
)
{
    int num_intersections = 0;
    FP_T current;
    FP_T xp = O.x + 1.0f;
    FP_T xm = O.x - 1.0f;
    FP_T yp = O.y + 1.0f;
    FP_T ym = O.y - 1.0f;
    FP_T stop = xp + max_dx;
    
    #ifdef DEBUG
    const bool is_debug_thread = (idx == DEBUG_X_ID && idy == DEBUG_Y_ID);
    if (is_debug_thread) {
        // Use a unique identifier like [C_I] for "compute_intersections"
        printf("[C_I (%d, %d)] --- Entering compute_intersections ---\n", idx, idy);
    }
    #endif
    
    int i = find_leftmost (v_1, v_2, v_3, 0, num_triangles, xm);

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("[C_I (%d, %d)] Starting triangle search at index i = %d. Loop stop condition: v.x <= %f\n", idx, idy, i, stop);
    }
    #endif

    while (i < num_triangles && v_1[i].x <= stop && v_2[i].x <= stop && v_3[i].x <= stop) {
        if (!((v_1[i].x < xm && v_2[i].x < xm && v_3[i].x < xm) ||
              (v_1[i].x > xp && v_2[i].x > xp && v_3[i].x > xp) ||
              (v_1[i].y < ym && v_2[i].y < ym && v_3[i].y < ym) ||
              (v_1[i].y > yp && v_2[i].y > yp && v_3[i].y > yp))) {
            /* There is a ray-bounding box intersection */
            current = compute_intersection_point (v_1[i], v_2[i], v_3[i], O, D); 
            if (current > -1) {
                // Before writing, check we haven't exceeded the max
                if (num_intersections < MAX_INTERSECTIONS) {
                    #ifdef DEBUG
                    if (is_debug_thread) {
                        printf("[C_I (%d, %d)] ✅ Intersection FOUND with tri %d. t = %f. Total hits: %d\n", i, current, num_intersections + 1);
                    }
                    #endif
                    intersections[num_intersections] = current;
                    num_intersections++;
                }
            }
        }
        i++;
    }

    // Sorting is not part of the original function, but good practice
    sort (intersections, num_intersections);
    // thrust::sort(thrust::seq, intersections, intersections + num_intersections);

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("[C_I (%d, %d)] Loop finished. Total intersections found: %d\n", idx, idy, num_intersections);
        
        // Print all the found and sorted intersection values
        if (num_intersections > 0) {
            printf("[C_I (%d, %d)] Sorted intersection t-values:\n", idx, idy);
            for (int k = 0; k < num_intersections; ++k) {
                printf("[C_I (%d, %d)]   t[%d] = %f\n", idx, idy, k, intersections[k]);
            }
        }
        
        printf("[C_I (%d, %d)] --- Exiting compute_intersections ---\n", idx, idy);
    }
    #endif


    return num_intersections;
}

__device__ FP_T project_thickness_legacy (FP_T *intersections, int num_intersections)
{
    int i, j;
    FP_T result = 0.0;
    constexpr FP_T epsilon = ::cuda::std::numeric_limits<FP_T>::epsilon();

    i = 0;
    while (i < num_intersections) {
        j = i + 1;
        while (j < num_intersections && fabsf (intersections[j] - intersections[i]) < epsilon) {
            j++;
        }
        if (i < num_intersections && j < num_intersections) {
            result += fabsf (intersections[j] - intersections[i]);
        }
        i = j + 1;
    }

    return result;
}



extern "C" __global__ void compute_thickness_kernel(
    FP_T3 *v_1, FP_T3 *v_2, FP_T3 *v_3, int num_triangles,
    FP_T *output, int full_image_width,
    int2 roi_offset,
    FP_T2 mesh_offset, 
    FP_T scale,
    FP_T max_dx, 
    FP_T min_z, 
    int supersampling 
) {
    // --- 1. Map Thread to Pixel ---
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int global_idx = idx + roi_offset.x;
    const int global_idy = idy + roi_offset.y;

    // --- 2. Generate Ray ---
    FP_T x_0 = idx + roi_offset.x + (0.5) + mesh_offset.x;
    FP_T y_0 = idy + roi_offset.y + (0.5) + mesh_offset.y;
    FP_T3 D = MAKE_FP_T3(0.0f, 0.0f, 1.0f);
    FP_T3 O;
    O.z = min_z - (1.0);
    
    FP_T results[16];
    for (int i = 0; i < supersampling; i++) {
        for (int j = 0; j < supersampling; j++) {
            O.x = ((2.0) * i - supersampling + (1.0)) / ((2.0) * supersampling) + x_0;
            O.y = ((2.0) * j - supersampling + (1.0)) / ((2.0) * supersampling) + y_0;

            FP_T private_intersections[MAX_INTERSECTIONS];
            int num_intersections = 0;
            
            // Pass the thread IDs to the intersection function
            num_intersections = compute_intersections(v_1, v_2, v_3, num_triangles, O, D, max_dx, private_intersections, global_idx, global_idy);
            
            if (num_intersections == MAX_INTERSECTIONS) {
                output[(idy + roi_offset.y) * full_image_width + idx + roi_offset.x] = NAN;
                return;
            }
            results[i * supersampling + j] = project_thickness_legacy (private_intersections, num_intersections);
        }
    }
    
    int num_samples = supersampling * supersampling;
    sort (results, supersampling * supersampling);
    
    FP_T final_thickness = scale * results[num_samples / 2];
    output[(idy + roi_offset.y) * full_image_width + (idx + roi_offset.x)] = final_thickness;
}