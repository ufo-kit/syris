#pragma once

#include "Commons.cuh"

// One formula for single samples, adaptive samples and the staged sampler.
// The host passes the detector centre in mixed/Float64 mode, and its corner
// in Float32 mode. All positions already share the mesh's normalized frame.
struct DetectorPixel {
    FP_T4 origin, u, v;
#if defined(SYRIS_MIXED_PRECISION) || defined(__FP_T_D__)
    FP_T col_offset, row_offset;
#endif

    __device__ DetectorPixel(FP_T4 detector_origin, FP_T4 scaled_u, FP_T4 scaled_v,
                             uint2 shape, unsigned col, unsigned row)
        : u(scaled_u), v(scaled_v)
    {
#if defined(SYRIS_MIXED_PRECISION) || defined(__FP_T_D__)
        origin = detector_origin;
        col_offset = FP_T(col) - FP_T(shape.x) * FP_T(0.5);
        row_offset = FP_T(row) - FP_T(shape.y) * FP_T(0.5);
#else
        origin = detector_origin + u * col + v * row;
#endif
    }

    __device__ FP_T4 point(FP_T x, FP_T y) const
    {
#if defined(SYRIS_MIXED_PRECISION) || defined(__FP_T_D__)
        return origin + u * (col_offset + x) + v * (row_offset + y);
#else
        return origin + u * x + v * y;
#endif
    }
};
