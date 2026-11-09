#include "Commons.cuh"
#include "Ray.cuh"
#include "WatertightRay.cuh"
#include "PixelScheduler.cuh"
#include "Detector.cuh"
#include <cuda/atomic>


__device__ bool isLeaf(const Tree &tree, const unsigned int index)
{
    return index < tree.nb_keys;
}

__device__ unsigned int toInternalRepresentation(const Tree &tree, const unsigned int index)
{
    return index + tree.nb_keys;
}

__device__ void growBox(const FP_T4 &bbMinInput, const FP_T4 &bbMaxInput, FP_T4 *bbMinOutput, FP_T4 *bbMaxOutput)
{
    bbMinOutput->x = (fmin)(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = (fmin)(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = (fmin)(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = (fmax)(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = (fmax)(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = (fmax)(bbMaxInput.z, bbMaxOutput->z);
}

__device__ delta_t delta(const Tree &tree, const int index)
{
    constexpr delta_t MAX = ::cuda::std::numeric_limits<delta_t>::max();
    constexpr delta_t MIN = ::cuda::std::numeric_limits<delta_t>::min();

    if (index < 0 || index >= tree.nb_keys - 1)
    {
        return MAX;
    }

    morton_t a = tree.keys[index];
    morton_t b = tree.keys[index + 1];
    morton_t x = a ^ b;
    return x + (!x) * (MIN + (index ^ (index + 1))) - 1; //
}

__device__ void setRope(Tree &tree, unsigned int skip_index, int range_right, delta_t delta_right)
{
    int rope;

    if (range_right != tree.nb_keys - 1)
    {
        int r = range_right + 1;
        rope = delta_right < delta(tree, r) ? r : toInternalRepresentation(tree, r);
    }
    else
    {
        rope = SENTINEL;
    }
    tree.rope[skip_index] = rope;
}

__device__ void setLeftChild(Tree &tree, unsigned int parent, unsigned int left_child)
{
    tree.left[parent] = left_child;
}

__device__ void setBBMin(Tree &tree, unsigned int parent, FP_T4 bbMin)
{
    tree.bboxMin[parent] = bbMin;
}

__device__ void setBBMax(Tree &tree, unsigned int parent, FP_T4 bbMax)
{
    tree.bboxMax[parent] = bbMax;
}

__device__ FP_T4 getBBMin(const Tree &tree, const unsigned int index)
{
    return tree.bboxMin[index];
}

__device__ FP_T4 getBBMax(const Tree &tree, const unsigned int index)
{
    return tree.bboxMax[index];
}

__device__ int getRope(const Tree &tree, const unsigned int index)
{
    return tree.rope[index];
}

__device__ int getLeftChild(const Tree &tree, const unsigned int index)
{
    return tree.left[index];
}

__device__ void updateParents(Tree &tree, int i)
{
    if (tree.nb_keys == 1) {
        tree.rope[0] = SENTINEL;
        return;
    }
    int range_left = i;
    int range_right = i;
    delta_t delta_left = delta(tree, i - 1);
    delta_t delta_right = delta(tree, i);

    FP_T4 bbMinCurrent = getBBMin(tree, i);
    FP_T4 bbMaxCurrent = getBBMax(tree, i);

    setRope(tree, i, range_right, delta_right);

    unsigned const root = toInternalRepresentation(tree, 0);

    do
    {
        int left_child;
        if (delta_right < delta_left)
        {
            const int apetrei_parent = range_right;

            cuda::atomic_ref<int, cuda::thread_scope_device> slot(
                tree.entered[toInternalRepresentation(tree, apetrei_parent)]);
            int previous = INVALID;
            slot.compare_exchange_strong(previous, range_left, cuda::memory_order_acq_rel);
            range_right = previous;

            if (range_right == INVALID)
            {
                return;
            }
            delta_right = delta(tree, range_right);

            left_child = i;

            int right_child = apetrei_parent + 1;

            
            if (right_child != range_right)
            {
                right_child = toInternalRepresentation(tree, right_child);
            }

            // The successful publisher releases its completed child bounds;
            // this failed CAS acquires them before the sibling is read.

            FP_T4 bbMinRight = getBBMin(tree, right_child);
            FP_T4 bbMaxRight = getBBMax(tree, right_child);
            growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
        }
        else
        {
            int const apetrei_parent = range_left - 1;
            cuda::atomic_ref<int, cuda::thread_scope_device> slot(
                tree.entered[toInternalRepresentation(tree, apetrei_parent)]);
            int previous = INVALID;
            slot.compare_exchange_strong(previous, range_right, cuda::memory_order_acq_rel);
            range_left = previous;

            if (range_left == INVALID)
            {
                return;
            }

            delta_left = delta(tree, range_left - 1);

            left_child = apetrei_parent;
            bool const left_is_leaf = (left_child == range_left);

            
            if (!left_is_leaf)
            {
                left_child = toInternalRepresentation(tree, left_child);
            }
            

            FP_T4 bbMinLeft = getBBMin(tree, left_child);
            FP_T4 bbMaxLeft = getBBMax(tree, left_child);
            growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
        }

        int karras_parent = delta_right < delta_left ? range_right : range_left;
        karras_parent = toInternalRepresentation(tree, karras_parent);

        setLeftChild(tree, karras_parent, left_child);
        setBBMin(tree, karras_parent, bbMinCurrent);
        setBBMax(tree, karras_parent, bbMaxCurrent);
        setRope(tree, karras_parent, range_right, delta_right);

        i = karras_parent;
    } while (i != root);

    return;
}

__device__ int treeRoot(const Tree &tree)
{
    return tree.nb_keys == 1 ? 0 : int(tree.nb_keys);
}

// Advance a stackless traversal by one intersected leaf. Candidate triangles
// are tested immediately, so there is no bounded broad-phase candidate list.
template<typename RayType>
__device__ int nextLeaf(const Tree &tree, const RayType &ray, int &node)
{
    while (node != SENTINEL) {
        int current = node;
        if (!ray.intersects(getBBMin(tree, current), getBBMax(tree, current))) {
            node = getRope(tree, current);
        } else if (isLeaf(tree, current)) {
            node = getRope(tree, current);
            return current;
        } else {
            node = getLeftChild(tree, current);
        }
    }
    return SENTINEL;
}

// Work in distances close to the mesh. Subtracting two large camera distances
// can erase a thin layer even though its vertices are accurately represented.
// Evaluate the shift in double, then use one common rebased ray for all faces.
template<typename RayType>
__device__ RayType rebaseRay(const RayType &ray, const Tree &tree)
{
    FP_T4 lo = getBBMin(tree, treeRoot(tree));
    FP_T4 hi = getBBMax(tree, treeRoot(tree));
    double o[3] = {ray.tail.x, ray.tail.y, ray.tail.z};
    double d[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    double lower[3] = {lo.x, lo.y, lo.z}, upper[3] = {hi.x, hi.y, hi.z};
    double entry = 0.0, extent = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        extent = fmax(extent, upper[axis] - lower[axis]);
        if (d[axis] != 0.0) {
            double near_plane = d[axis] > 0.0 ? lower[axis] : upper[axis];
            entry = fmax(entry, (near_plane - o[axis]) / d[axis]);
        }
    }
    // The shift itself is computed in double. For a distant Float64 camera,
    // its rounding can exceed a guard based only on the small mesh extent,
    // placing the rebased origin just INSIDE the first surface.
    const double shift_error = 8.0 * ::cuda::std::numeric_limits<double>::epsilon() * entry;
    const double guard = fmax(shift_error,
        fmax(32.0 * ::cuda::std::numeric_limits<FP_T>::epsilon() * extent,
             2.0 * double(ray.m_epsilons.tri_ray_tmin)));
    double shift = fmax(0.0, entry - guard);
    FP_T4 origin = MAKE_FP_T4(FP_T(fma(shift, d[0], o[0])),
                              FP_T(fma(shift, d[1], o[1])),
                              FP_T(fma(shift, d[2], o[2])), 0);
    return RayType(origin, ray.direction, lo, hi, ray.m_epsilons);
}

template<bool WithNormals>
struct ThicknessSum {
    double total = 0.0;
    HIT_T entry = 0, last = 0;
    int inside = 0;
    bool first = true, valid = true;

    __device__ void event(HIT_T t, int orientation) {
        if constexpr (WithNormals) {
            if (orientation == 0) return; // A tangential contact, or touching faces.
            if (first && orientation < 0) inside = 1; // Ray starts in material.
            first = false;
            int previous = inside;
            if (orientation > 0) {
                if (inside == 0) entry = t;
                ++inside;
            } else {
                --inside;
                if (inside < 0) valid = false;
            }
            if (previous > 0 && inside == 0) total += double(t) - double(entry);
        } else {
            if (inside) total += double(t) - double(entry);
            else entry = t;
            inside ^= 1;
        }
        last = t;
    }

    __device__ FP_T result() const {
        if constexpr (WithNormals) {
            // Open or inconsistently oriented surfaces must not look like a
            // valid truncated projection.
            return valid && inside == 0 ? FP_T(total) : FP_T(NAN);
        } else {
            // A closed mesh has an odd number of forward crossings precisely
            // when the ray origin is inside material.
            return FP_T(inside ? double(last) - total : total);
        }
    }
};

template<bool WithNormals, typename RayType>
__device__ int hitOrientation(const RayType &ray, const FP_T4 *normals, unsigned triangle)
{
    if constexpr (WithNormals) {
        FP_T product = dot(ray.direction, normals[triangle]);
        return (product < 0) - (product > 0);
    }
    return 0;
}

// Bounded nearest-event batches for rays exceeding the local hit capacity.
// Each pass visits the complete BVH and retains the next BatchSize DISTINCT
// distances, summing ALL orientations at retained distances. The cutoff only
// decreases, so an evicted event cannot re-enter this batch. This guarantees
// that duplicate groups are never split, even if a group exceeds the buffer.
// Storage is bounded; the number of batches and geometric crossings is not.
template<bool WithNormals, typename RayType>
__device__ __noinline__ FP_T traceEvents(const RayType &ray, const Tree &tree,
                                        FP_T4 *vertices, FP_T4 *normals,
                                        unsigned row, unsigned col)
{
    constexpr unsigned BatchSize = 32;
    struct Event { HIT_T t; int orientation; };
    Event events[BatchSize];
    ThicknessSum<WithNormals> sum;
    HIT_T previous = NEG_INFINITY;
    while (true) {
        unsigned count = 0, furthest = 0;
        int node = treeRoot(tree), leaf;
        while ((leaf = nextLeaf(tree, ray, node)) != SENTINEL) {
            unsigned triangle = tree.indices[leaf], base = 3 * triangle;
            HIT_T t;
            if (!ray.intersects(vertices[base], vertices[base+1], vertices[base+2], t, col, row)
                || t <= previous) continue;
            if (count == BatchSize && t > events[furthest].t) continue;
            int orientation = hitOrientation<WithNormals>(ray, normals, triangle);
            unsigned i = 0;
            while (i < count && events[i].t != t) ++i;
            if (i < count) {
                events[i].orientation += orientation;
                continue;
            }
            if (count < BatchSize) {
                events[count++] = {t, orientation};
                if (t > events[furthest].t) furthest = count - 1;
            } else {
                events[furthest] = {t, orientation};
                for (unsigned j = 0; j < count; ++j)
                    if (events[j].t > events[furthest].t) furthest = j;
            }
        }
        if (count == 0) break;
        // Small, fixed-size batches: insertion sort avoids a second hit list.
        for (unsigned i = 1; i < count; ++i) {
            Event value = events[i];
            unsigned j = i;
            while (j > 0 && value.t < events[j-1].t) {
                events[j] = events[j-1];
                --j;
            }
            events[j] = value;
        }
        for (unsigned i = 0; i < count; ++i)
            sum.event(events[i].t, events[i].orientation);
        previous = events[count - 1].t;
        if (count < BatchSize) break;
    }
    return sum.result();
}

template<bool WithNormals, unsigned Capacity, typename RayType>
__device__ FP_T traceCollected(const RayType &ray, const Tree &tree,
                              FP_T4 *vertices, FP_T4 *normals,
                              unsigned row, unsigned col, bool &overflow)
{
    List<HitRecord, Capacity> hits;
    int node = treeRoot(tree), leaf;
    while ((leaf = nextLeaf(tree, ray, node)) != SENTINEL) {
        unsigned triangle = tree.indices[leaf];
        unsigned base = 3 * triangle;
        HIT_T t;
        if (ray.intersects(vertices[base], vertices[base+1], vertices[base+2], t, col, row)) {
            if (!hits.push_back(HitRecord(t, triangle))) {
                overflow = true;
                return 0;
            }
        }
    }
    sort_hit(hits.values, hits.size());
    ThicknessSum<WithNormals> sum;
    unsigned i = 0;
    while (i < hits.size()) {
        HIT_T t = hits.values[i].t;
        int orientation = 0;
        do {
            orientation += hitOrientation<WithNormals>(ray, normals, hits.values[i].triangle_idx);
            ++i;
        } while (i < hits.size() && hits.values[i].t == t);
        // Boundary distances are canonicalized by the triangle predicate.
        // No distance-relative epsilon may merge a real entry and exit.
        sum.event(t, orientation);
    }
    return sum.result();
}

// Keep the large list out of the common path's stack frame. The retry uses the
// SAME rebased ray and event rules, including duplicate boundaries and NaNs.
template<bool WithNormals, typename RayType>
__device__ __noinline__ FP_T traceLargeHits(const RayType &ray, const Tree &tree,
                                          FP_T4 *vertices, FP_T4 *normals,
                                          unsigned row, unsigned col)
{
    bool overflow = false;
    FP_T value = traceCollected<WithNormals, MAX_COLLISIONS>(
        ray, tree, vertices, normals, row, col, overflow);
    return overflow ? traceEvents<WithNormals>(ray, tree, vertices, normals, row, col) : value;
}

#ifdef SYRIS_SIGNED_THICKNESS
// Experimental contract: normals describe a closed, consistently oriented
// boundary of material (winding number 0 or 1), with no duplicate/overlapping
// shells. A balanced signed sum does NOT validate that contract. Keep this
// disabled for general meshes, whose sorted crossing checks remain necessary.
// On ambiguous rays, use the unchanged ordered-event semantics instead.
template<typename RayType>
__device__ __noinline__ FP_T traceSignedFallback(const RayType &ray, const Tree &tree,
                                                FP_T4 *vertices, FP_T4 *normals,
                                                unsigned row, unsigned col)
{
#ifdef SYRIS_SIGNED_DEFER_FALLBACK
    // Thickness is nonnegative. This internal marker is consumed by the
    // deferred pipeline, which retries the entire pixel with the sorted code.
    return FP_T(NEG_INFINITY);
#else
    bool overflow = false;
    FP_T value = traceCollected<true, MAX_COLLISIONS>(
        ray, tree, vertices, normals, row, col, overflow);
    return overflow ? traceEvents<true>(ray, tree, vertices, normals, row, col) : value;
#endif
}

struct SignedLength {
    double sum = 0.0, correction = 0.0;

    __device__ void add(double value) {
        // Neumaier compensation, with explicit rounding to preserve the error
        // terms. Distances have already been rebased close to the scene.
        double next = __dadd_rn(sum, value);
        double error = fabs(sum) >= fabs(value)
            ? __dadd_rn(__dsub_rn(sum, next), value)
            : __dadd_rn(__dsub_rn(value, next), sum);
        correction = __dadd_rn(correction, error);
        sum = next;
    }

    __device__ double result() const { return __dadd_rn(sum, correction); }
};

template<typename RayType>
__device__ FP_T traceSignedThickness(const RayType &ray, const Tree &tree,
                                    FP_T4 *vertices, FP_T4 *normals,
                                    unsigned row, unsigned col)
{
    SignedLength sum;
    int balance = 0, first_orientation = 0;
    HIT_T first = POS_INFINITY;
    int node = treeRoot(tree), leaf;
    while ((leaf = nextLeaf(tree, ray, node)) != SENTINEL) {
        unsigned triangle = tree.indices[leaf], base = 3 * triangle;
        HIT_T t;
        bool boundary = false;
        if (!ray.intersects(vertices[base], vertices[base+1], vertices[base+2],
                            t, col, row, &boundary)) continue;
        FP_T4 n = normals[triangle];
        FP_T product = dot(ray.direction, n);
        FP_T magnitude = fabs(ray.direction.x * n.x) + fabs(ray.direction.y * n.y)
                       + fabs(ray.direction.z * n.z);
        if (boundary || !isfinite(product) ||
            fabs(product) <= FP_T(8) * ::cuda::std::numeric_limits<FP_T>::epsilon() * magnitude)
            return traceSignedFallback(ray, tree, vertices, normals, row, col);
        int orientation = (product < 0) - (product > 0);
        if (t < first) { first = t; first_orientation = orientation; }
        balance += orientation;
        sum.add(-double(orientation) * double(t));
    }
    double value = sum.result();
    // Catch inconsistent endpoint counts, leading exits on an outside ray,
    // and nonphysical totals. These checks are necessary, NOT sufficient to
    // infer validity of a general surface or replace mesh validation.
    bool endpoints = (balance == 0 && first_orientation >= 0)
                  || (balance == -1 && first_orientation < 0);
    if (!endpoints || !isfinite(value) || value < 0)
        return traceSignedFallback(ray, tree, vertices, normals, row, col);
    return FP_T(value);
}
#endif

template<bool WithNormals, typename RayType>
__device__ FP_T traceThickness(const RayType &input_ray, const Tree &tree,
                              FP_T4 *vertices, FP_T4 *normals,
                              unsigned row, unsigned col)
{
    if (tree.nb_keys == 0) return 0;
    if (!input_ray.intersects(getBBMin(tree, treeRoot(tree)), getBBMax(tree, treeRoot(tree)))) return 0;
    RayType ray = rebaseRay(input_ray, tree);
#ifdef SYRIS_SIGNED_THICKNESS
    if constexpr (WithNormals)
        return traceSignedThickness(ray, tree, vertices, normals, row, col);
#endif
    bool overflow = false;
    FP_T value = traceCollected<WithNormals, SYRIS_HIT_CAPACITY>(
        ray, tree, vertices, normals, row, col, overflow);
    if (!overflow) return value;
    if constexpr (SYRIS_HIT_CAPACITY < MAX_COLLISIONS)
        return traceLargeHits<WithNormals>(ray, tree, vertices, normals, row, col);
    return traceEvents<WithNormals>(ray, tree, vertices, normals, row, col);
}

template<typename RayType>
__device__ FP_T traceRay(const RayType &ray, const Tree &tree,
                        FP_T4 *vertices, unsigned row, unsigned col)
{
    return traceThickness<false>(ray, tree, vertices, nullptr, row, col);
}

template<typename RayType>
__device__ FP_T traceRay(const RayType &ray, const Tree &tree,
                        FP_T4 *vertices, FP_T4 *normals, unsigned row, unsigned col)
{
    return traceThickness<true>(ray, tree, vertices, normals, row, col);
}

extern "C" __global__ void calculateBbBoxKernel(FP_T4 *vertices, FP_T4 *bbMin, FP_T4 *bbMax, unsigned int nb_keys)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nb_keys)
    {
        FP_T4 V1 = vertices[tid * 3];
        FP_T4 V2 = vertices[tid * 3 + 1];
        FP_T4 V3 = vertices[tid * 3 + 2];
        calculateTriangleBoundingBox(V1, V2, V3, bbMin[tid], bbMax[tid]);

        tid += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void projectTriangleCentroid(
    unsigned int const nb_keys, FP_T4 const *vertices, morton_t *keys,
    FP_T4 *bbMin, FP_T4 *bbMax, FP_T4 const scene_bbMin, FP_T4 const scene_bbMax)
{

    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys)
    {
        // Get the triangle vertices
        FP_T4 V1 = vertices[index * 3];
        FP_T4 V2 = vertices[index * 3 + 1];
        FP_T4 V3 = vertices[index * 3 + 2];

        // Calculate the bounding box of the triangle
        calculateTriangleBoundingBox(V1, V2, V3, bbMin[index], bbMax[index]);

        // Calculate the centroid of the AABB
        FP_T4 centroid = getBoundingBoxCentroid(bbMin[index], bbMax[index]);

        FP_T4 normalizedCentroid = normalize(centroid, scene_bbMin, scene_bbMax);

        // Calculate the morton code of the triangle
        morton_t mortonCode = calculateMortonCode64(normalizedCentroid);

        // Store the morton code
        keys[index] = mortonCode;

        index += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void growTreeKernel(
    unsigned int nb_keys, morton_t *keys, unsigned int *permutation,
    int *rope, int *left, int *entered,
    FP_T4 *bboxMin, FP_T4 *bboxMax)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys)
    {
        Tree tree;
        tree.nb_keys = nb_keys;
        tree.keys = keys;
        tree.indices = permutation;
        tree.entered = entered;
        tree.rope = rope;
        tree.left = left;
        tree.bboxMin = bboxMin;
        tree.bboxMax = bboxMax;

        updateParents(tree, index);
        index += blockDim.x * gridDim.x;
    }
}

__device__ FP_T traceSubPixel(
    const FP_T4& sample_origin, const FP_T4& direction,
    const FP_T4& scene_bbMin, const FP_T4& scene_bbMax,
    Tree& tree, FP_T4 *__restrict__ vertices,
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    WatertightRay ray = WatertightRay(sample_origin, direction, scene_bbMin, scene_bbMax, epsilons);
    return traceRay<WatertightRay>(ray, tree, vertices, row, col);
}

__device__ FP_T traceSubPixel(
    const FP_T4& sample_origin, const FP_T4& direction,
    const FP_T4& scene_bbMin, const FP_T4& scene_bbMax,
    Tree& tree, FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    WatertightRay ray = WatertightRay(sample_origin, direction, scene_bbMin, scene_bbMax, epsilons);
    return traceRay<WatertightRay>(ray, tree, vertices, normals, row, col);
}

// =========================================================================
//  LEVEL 2: CACHING FUNCTIONS (One per projection type)
// =========================================================================

__device__ FP_T traceAndCache(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    const DetectorPixel &pixel, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices, 
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    int ix = round(u * (CACHE_DIM - 1));
    int iy = round(v * (CACHE_DIM - 1));
    int index = iy * CACHE_DIM + ix;

    if (is_cached[index]) {
        return value_cache[index];
    }

    FP_T4 sample_origin = pixel.point(u, v);

    FP_T value = traceSubPixel(
        sample_origin, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T traceAndCache_conebeam(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    const DetectorPixel &pixel, FP_T4 source, // Note: 'source'
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices, 
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    int ix = round(u * (CACHE_DIM - 1));
    int iy = round(v * (CACHE_DIM - 1));
    int index = iy * CACHE_DIM + ix;

    if (is_cached[index]) {
        return value_cache[index];
    }

    FP_T4 sample_origin = pixel.point(u, v);
    
    // (Source -> Pixel)
    FP_T4 direction = sample_origin - source;
    FP_T norm = inverse_length(direction.x, direction.y, direction.z);
    direction = direction * norm; // Normalize

    FP_T value = traceSubPixel(
        source, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T traceAndCache(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    const DetectorPixel &pixel, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    int ix = round(u * (CACHE_DIM - 1));
    int iy = round(v * (CACHE_DIM - 1));
    int index = iy * CACHE_DIM + ix;

    if (is_cached[index]) {
        return value_cache[index];
    }

    FP_T4 sample_origin = pixel.point(u, v);

    FP_T value = traceSubPixel(
        sample_origin, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T traceAndCache_conebeam(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    const DetectorPixel &pixel, FP_T4 source, // Note: 'source'
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    unsigned int row, unsigned int col)
{
    int ix = round(u * (CACHE_DIM - 1));
    int iy = round(v * (CACHE_DIM - 1));
    int index = iy * CACHE_DIM + ix;

    if (is_cached[index]) {
        return value_cache[index];
    }

    FP_T4 sample_origin = pixel.point(u, v);

    // (Source -> Pixel)
    FP_T4 direction = sample_origin - source;
    FP_T norm = inverse_length(direction.x, direction.y, direction.z);
    direction = direction * norm;
    FP_T value = traceSubPixel(
        source, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}


// =========================================================================
//  LEVEL 3: ADAPTIVE SAMPLING (One per projection type)
// =========================================================================

// Seed only samples whose ray coordinates match the local pixel calculation.
// Unmatched corners are evaluated normally; adjacent pixel formulas can round
// differently even when they describe the same mathematical grid point.
__device__ void seedAdaptiveCache(FP_T *values, bool *cached,
                                  const FP_T *roots, unsigned mask)
{
    const int indices[5] = {0, CACHE_DIM - 1, CACHE_DIM * (CACHE_DIM - 1),
                            CACHE_SIZE - 1, CACHE_SIZE / 2};
    for (int i = 0; i < 5; ++i) {
        if (mask & (1u << i)) {
            values[indices[i]] = roots[i];
            cached[indices[i]] = true;
        }
    }
}

__device__ FP_T tracePixelAdaptive(
    const DetectorPixel &pixel, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree, FP_T4 *__restrict__ vertices,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col,
    const FP_T *root_values = nullptr, unsigned root_mask = 0)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    seedAdaptiveCache(value_cache, is_cached, root_values, root_mask);
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache(u,      v,      value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[1] = traceAndCache(u + s,  v,      value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[2] = traceAndCache(u,      v + s,  value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[3] = traceAndCache(u + s,  v + s,  value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[4] = traceAndCache(u + hs, v + hs, value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        
        // Decision
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) {
#ifdef SYRIS_SIGNED_DEFER_FALLBACK
            if (values[j] == FP_T(NEG_INFINITY)) return FP_T(NEG_INFINITY);
#endif
            if (!isfinite(values[j])) return FP_T(NAN);
        }
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            // Midpoint quadrature over every accepted region. Accumulate
            // immediately so the number of leaves is not a storage limit.
            total_value += double(midpoint_value) * double(s * s);
            total_area += double(s * s);
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    return (total_area > 0.0) ? (total_value / total_area) : 0.0;
}

__device__ FP_T tracePixelAdaptive(
    const DetectorPixel &pixel, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col,
    const FP_T *root_values = nullptr, unsigned root_mask = 0)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    seedAdaptiveCache(value_cache, is_cached, root_values, root_mask);
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache(u,      v,      value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[1] = traceAndCache(u + s,  v,      value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[2] = traceAndCache(u,      v + s,  value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[3] = traceAndCache(u + s,  v + s,  value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[4] = traceAndCache(u + hs, v + hs, value_cache, is_cached, pixel, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        
        // Decision
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) {
#ifdef SYRIS_SIGNED_DEFER_FALLBACK
            if (values[j] == FP_T(NEG_INFINITY)) return FP_T(NEG_INFINITY);
#endif
            if (!isfinite(values[j])) return FP_T(NAN);
        }
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            // Midpoint quadrature over every accepted region. Accumulate
            // immediately so the number of leaves is not a storage limit.
            total_value += double(midpoint_value) * double(s * s);
            total_area += double(s * s);
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    return (total_area > 0.0) ? (total_value / total_area) : 0.0;
}

__device__ FP_T tracePixelAdaptive_conebeam(
    const DetectorPixel &pixel, FP_T4 source, // Note: 'source'
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree, FP_T4 *__restrict__ vertices,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col,
    const FP_T *root_values = nullptr, unsigned root_mask = 0)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    seedAdaptiveCache(value_cache, is_cached, root_values, root_mask);
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache_conebeam(u,      v,      value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[1] = traceAndCache_conebeam(u + s,  v,      value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[2] = traceAndCache_conebeam(u,      v + s,  value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[3] = traceAndCache_conebeam(u + s,  v + s,  value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[4] = traceAndCache_conebeam(u + hs, v + hs, value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) {
#ifdef SYRIS_SIGNED_DEFER_FALLBACK
            if (values[j] == FP_T(NEG_INFINITY)) return FP_T(NEG_INFINITY);
#endif
            if (!isfinite(values[j])) return FP_T(NAN);
        }
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            // Midpoint quadrature over every accepted region. Accumulate
            // immediately so the number of leaves is not a storage limit.
            total_value += double(midpoint_value) * double(s * s);
            total_area += double(s * s);
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    return (total_area > 0.0) ? (total_value / total_area) : 0.0;
}

__device__ FP_T tracePixelAdaptive_conebeam(
    const DetectorPixel &pixel, FP_T4 source,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col,
    const FP_T *root_values = nullptr, unsigned root_mask = 0)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    seedAdaptiveCache(value_cache, is_cached, root_values, root_mask);
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache_conebeam(u,      v,      value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[1] = traceAndCache_conebeam(u + s,  v,      value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[2] = traceAndCache_conebeam(u,      v + s,  value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[3] = traceAndCache_conebeam(u + s,  v + s,  value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[4] = traceAndCache_conebeam(u + hs, v + hs, value_cache, is_cached, pixel, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) {
#ifdef SYRIS_SIGNED_DEFER_FALLBACK
            if (values[j] == FP_T(NEG_INFINITY)) return FP_T(NEG_INFINITY);
#endif
            if (!isfinite(values[j])) return FP_T(NAN);
        }
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            // Midpoint quadrature over every accepted region. Accumulate
            // immediately so the number of leaves is not a storage limit.
            total_value += double(midpoint_value) * double(s * s);
            total_area += double(s * s);
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    return (total_area > 0.0) ? (total_value / total_area) : 0.0;
}

// =========================================================================
// KERNELS
// =========================================================================

extern "C" __global__ void project_parallel_kernel(
    // base_args (4 args)
    unsigned *__restrict__ globalCounter,
    unsigned nb_keys,
    FP_T *__restrict__ image,
    FP_T4 *__restrict__ vertices,
    
    // camera_args (6 args)
    uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W,
    FP_T4 detector_origin,
    FP_T2 ps,

    // tree_args (7 args)
    int *__restrict__ rope, int *__restrict__ left,
    unsigned *__restrict__ permutation,
    FP_T4 *__restrict__ bboxMin, FP_T4 *__restrict__ bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    
    // sampling_args (3 args)
    FP_T p_abs_tolerance,
    FP_T p_rel_tolerance,
    int p_max_depth,
    
    // epsilon_args (9 args)
    FP_T p_ray_box_epsilon,
    FP_T p_tri_ray_tmin,
    FP_T p_tri_gamma_multiplier,
    FP_T p_tri_abs_min_error,
    FP_T p_tri_d_gamma_multiplier,
    FP_T p_tri_d_abs_min_error,
    FP_T p_group_abs_epsilon,
    FP_T p_group_rel_epsilon,
    FP_T p_unique_abs_epsilon
#ifdef SYRIS_PROJECTION_RETRY
    , const unsigned *retry_indices, const unsigned *retry_count
#endif
)
{
    Tree tree; /* ... tree setup ... */
    tree.nb_keys = nb_keys; tree.rope = rope; tree.left = left; tree.indices = permutation; tree.bboxMin = bboxMin; tree.bboxMax = bboxMax;

    EpsilonParams epsilons(
        p_ray_box_epsilon, p_tri_ray_tmin, p_tri_gamma_multiplier, p_tri_abs_min_error,
        p_tri_d_gamma_multiplier, p_tri_d_abs_min_error,
        p_group_abs_epsilon, p_group_rel_epsilon, p_unique_abs_epsilon
    );

    AdaptiveSamplingParams sampling_params(p_abs_tolerance, p_rel_tolerance, p_max_depth);

    FP_T4 scaled_U = U * ps.x; FP_T4 scaled_V = V * ps.y;
    unsigned int totalPixels = N.x * N.y;

    while (true) {
#ifdef SYRIS_PROJECTION_RETRY
        unsigned int work = atomicAdd(globalCounter, 1);
        if (work >= *retry_count) break;
        unsigned int index = retry_indices[work];
#else
        unsigned int index;
        if (!nextProjectionPixel(globalCounter, N, index)) break;
        if (index >= totalPixels) continue;
#endif

        unsigned int row = index / N.x; unsigned int col = index % N.x;
        DetectorPixel pixel(detector_origin, scaled_U, scaled_V, N, col, row);

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive(
                pixel, W,
                scene_bbMin, scene_bbMax, tree, vertices,
                epsilons, sampling_params,
                row, col
            );
        } else {
            FP_T4 sample_origin = pixel.point(FP_T(0.5), FP_T(0.5));
            image[index] = traceSubPixel(
                sample_origin, W, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col
            );
        }
    }
}

extern "C" __global__ void project_parallel_normals_kernel(
    // base_args (5 args)
    unsigned *globalCounter,
    unsigned nb_keys,
    FP_T *image,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    
    // camera_args (6 args)
    uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W,
    FP_T4 detector_origin,
    FP_T2 ps,

    // tree_args (7 args)
    int *rope, int *left, unsigned *permutation,
    FP_T4 *bboxMin, FP_T4 *bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    
    // sampling_args (3 args)
    FP_T p_abs_tolerance,
    FP_T p_rel_tolerance,
    int p_max_depth,
    
    // epsilon_args (9 args)
    FP_T p_ray_box_epsilon,
    FP_T p_tri_ray_tmin,
    FP_T p_tri_gamma_multiplier,
    FP_T p_tri_abs_min_error,
    FP_T p_tri_d_gamma_multiplier,
    FP_T p_tri_d_abs_min_error,
    FP_T p_group_abs_epsilon,
    FP_T p_group_rel_epsilon,
    FP_T p_unique_abs_epsilon
#ifdef SYRIS_PROJECTION_RETRY
    , const unsigned *retry_indices, const unsigned *retry_count
#endif
)
{
    Tree tree; /* ... tree setup ... */
    tree.nb_keys = nb_keys; tree.rope = rope; tree.left = left; tree.indices = permutation; tree.bboxMin = bboxMin; tree.bboxMax = bboxMax;

    EpsilonParams epsilons(
        p_ray_box_epsilon, p_tri_ray_tmin, p_tri_gamma_multiplier, p_tri_abs_min_error,
        p_tri_d_gamma_multiplier, p_tri_d_abs_min_error,
        p_group_abs_epsilon, p_group_rel_epsilon, p_unique_abs_epsilon
    );

    AdaptiveSamplingParams sampling_params(p_abs_tolerance, p_rel_tolerance, p_max_depth);
    
    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;
    unsigned totalPixels = N.x * N.y;

    while (true) {
#ifdef SYRIS_PROJECTION_RETRY
        unsigned int work = atomicAdd(globalCounter, 1);
        if (work >= *retry_count) break;
        unsigned int index = retry_indices[work];
#else
        unsigned int index;
        if (!nextProjectionPixel(globalCounter, N, index)) break;
        if (index >= totalPixels) continue;
#endif

        unsigned int row = index / N.x;
        unsigned int col = index % N.x;

        #ifdef DEBUG
        const bool is_debug = (row == debug_row) && (col == debug_col);
        if (is_debug) {
            printf("Launching paralel on index %d\n", index);
        }
        #endif

        DetectorPixel pixel(detector_origin, scaled_U, scaled_V, N, col, row);

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive(
                pixel, W,
                scene_bbMin, scene_bbMax, tree, vertices, normals,
                epsilons, sampling_params, row, col);
        } else {
            FP_T4 sample_origin = pixel.point(FP_T(0.5), FP_T(0.5));
            FP_T4 direction = W;

            image[index] = traceSubPixel(
                sample_origin, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col
            );

            #ifdef DEBUG
            if (is_debug) {
                printf("Tracing sub pixel. Supersampling: no. Normals: yes. Conebeam: no. Final ret: %f \n", image[index]);
            }
            #endif
        }
    }
}


extern "C" __global__ void project_conebeam_kernel(
    // base_args (4 args)
    unsigned *__restrict__ globalCounter,
    unsigned nb_keys,
    FP_T *__restrict__ image,
    FP_T4 *__restrict__ vertices,
    
    // camera_args (7 args)
    uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W,
    FP_T4 detector_origin,
    FP_T2 ps,
    FP_T4 source,

    // tree_args (7 args)
    int *__restrict__ rope, int *__restrict__ left,
    unsigned *__restrict__ permutation,
    FP_T4 *__restrict__ bboxMin, FP_T4 *__restrict__ bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    
    // sampling_args (3 args)
    FP_T p_abs_tolerance,
    FP_T p_rel_tolerance,
    int p_max_depth,
    
    // epsilon_args (9 args)
    FP_T p_ray_box_epsilon,
    FP_T p_tri_ray_tmin,
    FP_T p_tri_gamma_multiplier,
    FP_T p_tri_abs_min_error,
    FP_T p_tri_d_gamma_multiplier,
    FP_T p_tri_d_abs_min_error,
    FP_T p_group_abs_epsilon,
    FP_T p_group_rel_epsilon,
    FP_T p_unique_abs_epsilon
#ifdef SYRIS_PROJECTION_RETRY
    , const unsigned *retry_indices, const unsigned *retry_count
#endif
)
{
    Tree tree; /* ... tree setup ... */
    tree.nb_keys = nb_keys; tree.rope = rope; tree.left = left; tree.indices = permutation; tree.bboxMin = bboxMin; tree.bboxMax = bboxMax;
    
    EpsilonParams epsilons(
        p_ray_box_epsilon, p_tri_ray_tmin, p_tri_gamma_multiplier, p_tri_abs_min_error,
        p_tri_d_gamma_multiplier, p_tri_d_abs_min_error,
        p_group_abs_epsilon, p_group_rel_epsilon, p_unique_abs_epsilon
    );

    AdaptiveSamplingParams sampling_params(p_abs_tolerance, p_rel_tolerance, p_max_depth);

    FP_T4 scaled_U = U * ps.x; FP_T4 scaled_V = V * ps.y;
    unsigned totalPixels = N.x * N.y;

    while (true) {
#ifdef SYRIS_PROJECTION_RETRY
        unsigned int work = atomicAdd(globalCounter, 1);
        if (work >= *retry_count) break;
        unsigned int index = retry_indices[work];
#else
        unsigned int index;
        if (!nextProjectionPixel(globalCounter, N, index)) break;
        if (index >= totalPixels) continue;
#endif

        unsigned int row = index / N.x; unsigned int col = index % N.x;
        DetectorPixel pixel(detector_origin, scaled_U, scaled_V, N, col, row);

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive_conebeam(
                pixel, source,
                scene_bbMin, scene_bbMax, tree, vertices,
                epsilons, sampling_params, row, col
            );
        } else {
            FP_T4 sample_origin = pixel.point(FP_T(0.5), FP_T(0.5));
            FP_T4 direction = sample_origin - source;
            FP_T norm = inverse_length(direction.x, direction.y, direction.z);
            direction = direction * norm;
            
            image[index] = traceSubPixel(
                source, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col
            );
        }
    }
}

extern "C" __global__ void project_conebeam_normals_kernel(
    // base_args (5 args)
    unsigned *globalCounter,
    unsigned nb_keys,
    FP_T *__restrict__ image,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    
    // camera_args (7 args)
    uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W,
    FP_T4 detector_origin,
    FP_T2 ps,
    FP_T4 source,

    // tree_args (7 args)
    int *__restrict__ rope, int *__restrict__ left,
    unsigned *__restrict__ permutation,
    FP_T4 *__restrict__ bboxMin, FP_T4 *__restrict__ bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    
    // sampling_args (3 args)
    FP_T p_abs_tolerance,
    FP_T p_rel_tolerance,
    int p_max_depth,
    
    // epsilon_args (9 args)
    FP_T p_ray_box_epsilon,
    FP_T p_tri_ray_tmin,
    FP_T p_tri_gamma_multiplier,
    FP_T p_tri_abs_min_error,
    FP_T p_tri_d_gamma_multiplier,
    FP_T p_tri_d_abs_min_error,
    FP_T p_group_abs_epsilon,
    FP_T p_group_rel_epsilon,
    FP_T p_unique_abs_epsilon
#ifdef SYRIS_PROJECTION_RETRY
    , const unsigned *retry_indices, const unsigned *retry_count
#endif
)
{
    Tree tree; /* ... tree setup ... */
    tree.nb_keys = nb_keys; tree.rope = rope; tree.left = left; tree.indices = permutation; tree.bboxMin = bboxMin; tree.bboxMax = bboxMax;

    AdaptiveSamplingParams sampling_params(p_abs_tolerance, p_rel_tolerance, p_max_depth);

    EpsilonParams epsilons(
        p_ray_box_epsilon, p_tri_ray_tmin, p_tri_gamma_multiplier, p_tri_abs_min_error,
        p_tri_d_gamma_multiplier, p_tri_d_abs_min_error,
        p_group_abs_epsilon, p_group_rel_epsilon, p_unique_abs_epsilon
    );
    
    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;
    unsigned totalPixels = N.x * N.y;
    
    while (true) {
#ifdef SYRIS_PROJECTION_RETRY
        unsigned int work = atomicAdd(globalCounter, 1);
        if (work >= *retry_count) break;
        unsigned int index = retry_indices[work];
#else
        unsigned int index;
        if (!nextProjectionPixel(globalCounter, N, index)) break;
        if (index >= totalPixels) continue;
#endif

        unsigned int row = index / N.x;
        unsigned int col = index % N.x;

        #ifdef DEBUG
        const bool is_debug = (row == debug_row) && (col == debug_col);
        if (is_debug) {
            printf("Launching pconebeam_normals_kernel on index %d\n", index);
        }
        #endif

        DetectorPixel pixel(detector_origin, scaled_U, scaled_V, N, col, row);

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive_conebeam(
                pixel, source,
                scene_bbMin, scene_bbMax, tree, vertices, normals,
                epsilons, sampling_params, row, col
            );
        } else {
            FP_T4 sample_origin = pixel.point(FP_T(0.5), FP_T(0.5));
            FP_T4 direction = sample_origin - source;
            FP_T norm = inverse_length(direction.x, direction.y, direction.z);
            direction = direction * norm;

            image[index] = traceSubPixel(
                source, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col
            );

            #ifdef DEBUG
            if (is_debug) {
                printf("Tracing sub pixel. Supersampling: no. Normals: yes. Conebeam: yes. Final ret: %f \n", image[index]);
            }
            #endif
        }
    }
}

#include "staged.cuh"

// Marks only the internal deferred-sum sentinel. Genuine NaNs stay NaNs.
extern "C" __global__ void queue_signed_fallbacks(const FP_T *image, unsigned n,
                                                 unsigned *indices, unsigned *count)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && image[i] == FP_T(NEG_INFINITY)) indices[atomicAdd(count, 1)] = i;
}
