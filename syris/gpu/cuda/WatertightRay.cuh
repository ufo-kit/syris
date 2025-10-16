#pragma once
#include "Commons.cuh"

class WatertightRay {
public:
    // --- Pre-calculated Ray Data ---
    FP_T4 tail;
    FP_T4 direction;
    FP_T4 invDirection;

    // Remapped axis indices (which original axis corresponds to the new X, Y, Z)
    int Kx, Ky, Kz;
    FP_T Sx, Sy, Sz;

    // Corrected origins for conservative testing of remapped axes
    FP_T o_near_kx, o_near_ky, o_near_kz;
    FP_T o_far_kx, o_far_ky, o_far_kz;

    // Conservatively rounded reciprocal directions for remapped axes
    FP_T r_near_kx, r_near_ky, r_near_kz;
    FP_T r_far_kx, r_far_ky, r_far_kz;

    // Indices (0-5) into a flattened AABB array [min.x, min.y, min.z, max.x, max.y, max.z]
    // These are sorted based on ray direction for the remapped axes.
    int near_kx_idx, far_kx_idx;
    int near_ky_idx, far_ky_idx;
    int near_kz_idx, far_kz_idx;

    int sign[3];

    FP_T4 scene_min, scene_max;

    /**
     * @brief Constructs a WatertightRay and performs all pre-calculations.
     * @param origin The starting point of the ray.
     * @param direction The direction of the ray (should be normalized).
     * @param scene_min The minimum corner of the entire scene's bounding box.
     * @param scene_max The maximum corner of the entire scene's bounding box.
     */
    __device__ WatertightRay(FP_T4 origin, FP_T4 direction, const FP_T4& scene_min, const FP_T4& scene_max);

    /**
     * @brief Performs a conservative ray-AABB intersection test.
     * @param box_min The minimum corner of the AABB to test against.
     * @param box_max The maximum corner of the AABB to test against.
     * @param tmin The minimum valid intersection distance for the ray.
     * @param tmax The maximum valid intersection distance for the ray.
     * @return True if the ray intersects the box within the [tmin, tmax] interval, false otherwise.
     */
    __device__ bool intersects(const FP_T4& minBbox, const FP_T4& maxBbox, FP_T tmin, FP_T tmax) const;
    __device__ bool intersects (FP_T4 const &minBbox, FP_T4 const &maxBbox) const; // AABB
    __device__ bool intersects (FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &tmin, FP_T &tmax) const; // Triangle
    __device__ bool intersects (FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t, unsigned col, unsigned row) const; // Triangle

    __device__ FP_T point_2_parametric (FP_T4 const &point) const;
};
