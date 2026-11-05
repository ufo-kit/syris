#include "Commons.cuh"
#include "Ray.cuh"
#include "WatertightRay.cuh"
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
    const double guard = fmax(32.0 * ::cuda::std::numeric_limits<FP_T>::epsilon() * extent,
                             2.0 * double(ray.m_epsilons.tri_ray_tmin));
    double shift = fmax(0.0, entry - guard);
    FP_T4 origin = MAKE_FP_T4(FP_T(fma(shift, d[0], o[0])),
                              FP_T(fma(shift, d[1], o[1])),
                              FP_T(fma(shift, d[2], o[2])), 0);
    return RayType(origin, ray.direction, lo, hi, ray.m_epsilons);
}

template<bool WithNormals>
struct ThicknessSum {
    double total = 0.0;
    FP_T entry = 0, last = 0;
    int inside = 0;
    bool first = true, valid = true;

    __device__ void event(FP_T t, int orientation) {
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

// Rare fallback for rays with more actual hits than fit in local memory.
// Repeated BVH searches enumerate complete, ordered events using constant
// storage. Runtime may increase, but the projection is never truncated.
template<bool WithNormals, typename RayType>
__device__ FP_T traceEvents(const RayType &ray, const Tree &tree,
                           FP_T4 *vertices, FP_T4 *normals,
                           unsigned row, unsigned col)
{
    ThicknessSum<WithNormals> sum;
    FP_T previous = NEG_INFINITY;
    while (true) {
        FP_T next = POS_INFINITY;
        int orientation = 0, node = treeRoot(tree), leaf;
        while ((leaf = nextLeaf(tree, ray, node)) != SENTINEL) {
            unsigned triangle = tree.indices[leaf];
            unsigned base = 3 * triangle;
            FP_T t;
            if (ray.intersects(vertices[base], vertices[base+1], vertices[base+2], t, col, row)
                && t > previous) {
                if (t < next) {
                    next = t;
                    orientation = hitOrientation<WithNormals>(ray, normals, triangle);
                } else if (t == next) {
                    orientation += hitOrientation<WithNormals>(ray, normals, triangle);
                }
            }
        }
        if (!isfinite(next)) break;
        sum.event(next, orientation);
        previous = next;
    }
    return sum.result();
}

template<bool WithNormals, typename RayType>
__device__ FP_T traceThickness(const RayType &input_ray, const Tree &tree,
                              FP_T4 *vertices, FP_T4 *normals,
                              unsigned row, unsigned col)
{
    if (tree.nb_keys == 0) return 0;
    if (!input_ray.intersects(getBBMin(tree, treeRoot(tree)), getBBMax(tree, treeRoot(tree)))) return 0;
    RayType ray = rebaseRay(input_ray, tree);
    List<HitRecord> hits;
    int node = treeRoot(tree), leaf;
    while ((leaf = nextLeaf(tree, ray, node)) != SENTINEL) {
        unsigned triangle = tree.indices[leaf];
        unsigned base = 3 * triangle;
        FP_T t;
        if (ray.intersects(vertices[base], vertices[base+1], vertices[base+2], t, col, row)) {
            if (!hits.push_back(HitRecord(t, triangle)))
                return traceEvents<WithNormals>(ray, tree, vertices, normals, row, col);
        }
    }
    sort_hit(hits.values, hits.size());
    ThicknessSum<WithNormals> sum;
    unsigned i = 0;
    while (i < hits.size()) {
        FP_T t = hits.values[i].t;
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
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 direction,
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

    FP_T4 sample_origin = base_pixel_origin + scaled_U * u + scaled_V * v;

    FP_T value = traceSubPixel(
        sample_origin, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T traceAndCache_conebeam(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 source, // Note: 'source'
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

    FP_T4 sample_origin = base_pixel_origin + scaled_U * u + scaled_V * v;
    
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
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 direction,
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

    FP_T4 sample_origin = base_pixel_origin + scaled_U * u + scaled_V * v;

    FP_T value = traceSubPixel(
        sample_origin, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col
    );
    
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T traceAndCache_conebeam(
    FP_T u, FP_T v, FP_T* value_cache, bool* is_cached,
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 source, // Note: 'source'
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

    FP_T4 sample_origin = base_pixel_origin + scaled_U * u + scaled_V * v;

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

__device__ FP_T tracePixelAdaptive(
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree, FP_T4 *__restrict__ vertices,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache(u,      v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[1] = traceAndCache(u + s,  v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[2] = traceAndCache(u,      v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[3] = traceAndCache(u + s,  v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[4] = traceAndCache(u + hs, v + hs, value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        
        // Decision
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) if (!isfinite(values[j])) return FP_T(NAN);
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
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 direction,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache(u,      v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[1] = traceAndCache(u + s,  v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[2] = traceAndCache(u,      v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[3] = traceAndCache(u + s,  v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[4] = traceAndCache(u + hs, v + hs, value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, direction, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        
        // Decision
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) if (!isfinite(values[j])) return FP_T(NAN);
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
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 source, // Note: 'source'
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree, FP_T4 *__restrict__ vertices,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache_conebeam(u,      v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[1] = traceAndCache_conebeam(u + s,  v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[2] = traceAndCache_conebeam(u,      v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[3] = traceAndCache_conebeam(u + s,  v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        values[4] = traceAndCache_conebeam(u + hs, v + hs, value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, epsilons, row, col);
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) if (!isfinite(values[j])) return FP_T(NAN);
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
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 source,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax, Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T4 *__restrict__ normals,
    const EpsilonParams& epsilons,
    const AdaptiveSamplingParams& sampling_params,
    unsigned int row, unsigned int col)
{
    Quad quad_stack[MAX_QUADS_PER_PIXEL]; int stack_ptr = 0;
    double total_value = 0.0, total_area = 0.0;
    FP_T value_cache[CACHE_SIZE]; bool is_cached[CACHE_SIZE];
    for(int i = 0; i < CACHE_SIZE; ++i) { is_cached[i] = false; }
    quad_stack[stack_ptr++] = {0.0, 0.0, 1.0, 0};

    while (stack_ptr > 0) {
        Quad current_quad = quad_stack[--stack_ptr];
        FP_T u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0;
        
        FP_T values[5];
        values[0] = traceAndCache_conebeam(u,      v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[1] = traceAndCache_conebeam(u + s,  v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[2] = traceAndCache_conebeam(u,      v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[3] = traceAndCache_conebeam(u + s,  v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        values[4] = traceAndCache_conebeam(u + hs, v + hs, value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, source, scene_bbMin, scene_bbMax, tree, vertices, normals, epsilons, row, col);
        FP_T midpoint_value = values[4];
        for (int j = 0; j < 5; ++j) if (!isfinite(values[j])) return FP_T(NAN);
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
    FP_T4 upperleft_origin,
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
        unsigned int index = atomicAdd(globalCounter, 1);
        if (index >= totalPixels) break; 

        unsigned int row = index / N.x; unsigned int col = index % N.x;
        FP_T4 base_pixel_origin = upperleft_origin + scaled_U * col + scaled_V * row;

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive(
                base_pixel_origin, scaled_U, scaled_V, W,
                scene_bbMin, scene_bbMax, tree, vertices,
                epsilons, sampling_params,
                row, col
            );
        } else {
            FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;
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
    FP_T4 upperleft_origin,
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
        unsigned int index = atomicAdd(globalCounter, 1);
        if (index >= totalPixels) break; 

        unsigned int row = index / N.x;
        unsigned int col = index % N.x;

        #ifdef DEBUG
        const bool is_debug = (row == debug_row) && (col == debug_col);
        if (is_debug) {
            printf("Launching paralel on index %d\n", index);
        }
        #endif

        FP_T4 base_pixel_origin = upperleft_origin + scaled_U * col + scaled_V * row;

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive(
                base_pixel_origin, scaled_U, scaled_V, W,
                scene_bbMin, scene_bbMax, tree, vertices, normals,
                epsilons, sampling_params, row, col);
        } else {
            FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;
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
    FP_T4 upperleft_origin,
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
        unsigned int index = atomicAdd(globalCounter, 1);
        if (index >= totalPixels) break; 

        unsigned int row = index / N.x; unsigned int col = index % N.x;
        FP_T4 base_pixel_origin = upperleft_origin + scaled_U * col + scaled_V * row;

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive_conebeam(
                base_pixel_origin, scaled_U, scaled_V, source,
                scene_bbMin, scene_bbMax, tree, vertices,
                epsilons, sampling_params, row, col
            );
        } else {
            FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;
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
    FP_T4 upperleft_origin,
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
        unsigned int index = atomicAdd(globalCounter, 1);
        if (index >= totalPixels) break; 

        unsigned int row = index / N.x;
        unsigned int col = index % N.x;

        #ifdef DEBUG
        const bool is_debug = (row == debug_row) && (col == debug_col);
        if (is_debug) {
            printf("Launching pconebeam_normals_kernel on index %d\n", index);
        }
        #endif

        FP_T4 base_pixel_origin = upperleft_origin + scaled_U * col + scaled_V * row;

        if ((sampling_params.max_depth > 0) && (sampling_params.max_depth <= MAX_DEPTH)) {
            image[index] = tracePixelAdaptive_conebeam(
                base_pixel_origin, scaled_U, scaled_V, source,
                scene_bbMin, scene_bbMax, tree, vertices, normals,
                epsilons, sampling_params, row, col
            );
        } else {
            FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;
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
