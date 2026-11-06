#pragma once

// Experimental adaptive scheduling. Trace a detector-wide corner grid and
// pixel centers, accept coarse pixels, then refine only a compacted queue.
// Unmatched corners use the original ray calculation in a repair pass. The
// original local sampler supplies all deeper levels.

__device__ FP_T4 stagedPoint(FP_T4 origin, FP_T4 u, FP_T4 v,
                             unsigned col, unsigned row, FP_T x, FP_T y)
{
    FP_T4 base = origin + u * col + v * row;
    return base + u * x + v * y;
}

__device__ bool sameCoordinate(FP_T a, FP_T b)
{
#ifdef __FP_T_D__
    return __double_as_longlong(a) == __double_as_longlong(b);
#else
    return __float_as_uint(a) == __float_as_uint(b);
#endif
}

__device__ bool samePoint(FP_T4 a, FP_T4 b)
{
    return sameCoordinate(a.x, b.x) && sameCoordinate(a.y, b.y)
        && sameCoordinate(a.z, b.z) && sameCoordinate(a.w, b.w);
}

__device__ unsigned stagedRoots(unsigned col, unsigned row, uint2 n,
                                const FP_T *samples, FP_T *values)
{
    unsigned corners = (n.x + 1) * (n.y + 1), pixels = n.x * n.y;
    unsigned pixel = row * n.x + col;
    values[4] = samples[corners + pixel];
    for (unsigned k = 0; k < 4; ++k)
        values[k] = samples[corners + pixels * (k + 1) + pixel];
    return 31;
}

// A common ABI lets the Python launcher reuse arguments for all four stages.
// No host readback of the refinement count is needed between launches.
#define STAGED_ARGUMENTS \
    unsigned *counter, unsigned nb_keys, FP_T *image, \
    FP_T4 *vertices, FP_T4 *normals, uint2 n, \
    FP_T4 u, FP_T4 v, FP_T4 w, FP_T4 origin, FP_T2 ps, FP_T4 source, \
    int parallel, int *rope, int *left, unsigned *permutation, \
    FP_T4 *lo, FP_T4 *hi, FP_T4 scene_min, FP_T4 scene_max, \
    FP_T abs_tol, FP_T rel_tol, int depth, const FP_T *eps, \
    FP_T *samples, unsigned *queue_count, unsigned *queue

__device__ Tree stagedTree(unsigned n, int *rope, int *left,
                           unsigned *indices, FP_T4 *lo, FP_T4 *hi)
{
    Tree tree;
    tree.nb_keys = n; tree.rope = rope; tree.left = left;
    tree.indices = indices; tree.bboxMin = lo; tree.bboxMax = hi;
    return tree;
}

__device__ EpsilonParams stagedEps(const FP_T *e)
{
    return EpsilonParams(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
}

extern "C" __global__ void project_shared_samples_kernel(STAGED_ARGUMENTS)
{
    Tree tree = stagedTree(nb_keys, rope, left, permutation, lo, hi);
    EpsilonParams epsilons = stagedEps(eps);
    FP_T4 su = u * ps.x, sv = v * ps.y;
    unsigned corners = (n.x + 1) * (n.y + 1);
    unsigned count = corners + n.x * n.y;
    while (true) {
        unsigned index = atomicAdd(counter, 1);
        if (index >= count) break;
        unsigned col, row;
        FP_T x, y;
        if (index < corners) {
            unsigned gx = index % (n.x + 1), gy = index / (n.x + 1);
            col = min(gx, n.x - 1); row = min(gy, n.y - 1);
            x = FP_T(gx - col); y = FP_T(gy - row);
        } else {
            col = (index - corners) % n.x; row = (index - corners) / n.x;
            x = y = FP_T(0.5);
        }
        FP_T4 point = stagedPoint(origin, su, sv, col, row, x, y);
        FP_T4 direction = w, tail = point;
        if (!parallel) {
            direction = point - source;
            direction = direction * inverse_length(direction.x, direction.y, direction.z);
            tail = source;
        }
        samples[index] = normals
            ? traceSubPixel(tail, direction, scene_min, scene_max, tree, vertices, normals, epsilons, row, col)
            : traceSubPixel(tail, direction, scene_min, scene_max, tree, vertices, epsilons, row, col);
    }
}

// Fill each pixel's corner stencil. Exact matches reuse the global grid;
// other corners trace their original ray in this small sampling kernel, not
// in the large adaptive kernel. This keeps the refinement queue selective.
extern "C" __global__ void project_missing_samples_kernel(STAGED_ARGUMENTS)
{
    Tree tree = stagedTree(nb_keys, rope, left, permutation, lo, hi);
    EpsilonParams epsilons = stagedEps(eps);
    FP_T4 su = u * ps.x, sv = v * ps.y;
    unsigned corners = (n.x + 1) * (n.y + 1), pixels = n.x * n.y;
    while (true) {
        unsigned index = atomicAdd(counter, 1);
        if (index >= 4 * pixels) break;
        unsigned k = index / pixels, pixel = index % pixels;
        unsigned col = pixel % n.x, row = pixel / n.x;
        unsigned x = k & 1, y = k >> 1, gx = col + x, gy = row + y;
        unsigned owner_col = min(gx, n.x - 1), owner_row = min(gy, n.y - 1);
        FP_T4 point = stagedPoint(origin, su, sv, col, row, FP_T(x), FP_T(y));
        FP_T4 shared = stagedPoint(origin, su, sv, owner_col, owner_row,
                                   FP_T(gx - owner_col), FP_T(gy - owner_row));
        FP_T value;
        if (samePoint(point, shared)) {
            value = samples[gy * (n.x + 1) + gx];
        } else {
            FP_T4 direction = w, tail = point;
            if (!parallel) {
                direction = point - source;
                direction = direction * inverse_length(direction.x, direction.y, direction.z);
                tail = source;
            }
            value = normals
                ? traceSubPixel(tail, direction, scene_min, scene_max, tree, vertices, normals, epsilons, row, col)
                : traceSubPixel(tail, direction, scene_min, scene_max, tree, vertices, epsilons, row, col);
        }
        samples[corners + pixels + index] = value;
    }
}

extern "C" __global__ void classify_shared_samples_kernel(STAGED_ARGUMENTS)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n.x * n.y) return;
    FP_T values[5];
    unsigned mask = stagedRoots(index % n.x, index / n.x, n, samples, values);
    if (mask == 31) {
        FP_T midpoint = values[4];
        for (int k = 0; k < 5; ++k) {
            if (!isfinite(values[k])) { image[index] = FP_T(NAN); return; }
        }
        sort(values, 5);
        if (values[4] - values[0] < fmax(abs_tol, rel_tol * values[4])) {
            image[index] = midpoint;
            return;
        }
    }
    queue[atomicAdd(queue_count, 1)] = index;
}

extern "C" __global__ void refine_shared_samples_kernel(STAGED_ARGUMENTS)
{
    Tree tree = stagedTree(nb_keys, rope, left, permutation, lo, hi);
    EpsilonParams epsilons = stagedEps(eps);
    AdaptiveSamplingParams sampling(abs_tol, rel_tol, depth);
    FP_T4 su = u * ps.x, sv = v * ps.y;
    while (true) {
        unsigned work = atomicAdd(counter, 1);
        if (work >= *queue_count) break;
        unsigned index = queue[work], col = index % n.x, row = index / n.x;
        FP_T roots[5];
        unsigned mask = stagedRoots(col, row, n, samples, roots);
        FP_T4 base = origin + su * col + sv * row;
        if (parallel) {
            image[index] = normals
                ? tracePixelAdaptive(base, su, sv, w, scene_min, scene_max, tree, vertices, normals, epsilons, sampling, row, col, roots, mask)
                : tracePixelAdaptive(base, su, sv, w, scene_min, scene_max, tree, vertices, epsilons, sampling, row, col, roots, mask);
        } else {
            image[index] = normals
                ? tracePixelAdaptive_conebeam(base, su, sv, source, scene_min, scene_max, tree, vertices, normals, epsilons, sampling, row, col, roots, mask)
                : tracePixelAdaptive_conebeam(base, su, sv, source, scene_min, scene_max, tree, vertices, epsilons, sampling, row, col, roots, mask);
        }
    }
}

#undef STAGED_ARGUMENTS
