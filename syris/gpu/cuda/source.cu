#include "Commons.cuh"
#include "Ray.cuh"
#include "WatertightRay.cuh"


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

            range_right = atomicCAS(&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_left);

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

            // Memory sync
            __threadfence();

            FP_T4 bbMinRight = getBBMin(tree, right_child);
            FP_T4 bbMaxRight = getBBMax(tree, right_child);
            growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
        }
        else
        {
            int const apetrei_parent = range_left - 1;
            range_left = atomicCAS(&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_right);

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
            
            // Memory sync
            __threadfence();

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

template <typename RayType>
__device__ void query(const Tree &tree, const RayType &ray, List<int> &candidates, unsigned col, unsigned row)
{
    const bool is_debug_thread = (row == debug_row && col == debug_col);
    int current_node = toInternalRepresentation(tree, 0);

    do
    {
        const FP_T4 bbMax = getBBMax(tree, current_node);
        const FP_T4 bbMin = getBBMin(tree, current_node);
        if (ray.intersects(bbMin, bbMax))
        {
            if (isLeaf(tree, current_node))
            {
                if (!candidates.push_back(current_node))
                {
                    return;
                }
                current_node = getRope(tree, current_node);
            }
            else
            {
                current_node = getLeftChild(tree, current_node);
            }
        }
        else
        {
            current_node = getRope(tree, current_node);
        }
    } while (current_node != SENTINEL);
}



__device__ FP_T project_thickness(List<FP_T> &tvalues)
{
    // This function assumes tvalues have already been sorted and made unique.
    FP_T total_thickness = 0.0f;

    // Iterate over the list in pairs of (entry, exit).
    // The loop condition `i + 1 < tvalues.size()` automatically and correctly
    // handles an odd number of intersections by ignoring the last element.
    for (int i = 0; i + 1 < tvalues.size(); i += 2)
    {
        FP_T entry_point = tvalues.values[i];
        FP_T exit_point = tvalues.values[i + 1];

        // Ensure we don't add negative thickness if something is wrong.
        if (exit_point > entry_point) {
            total_thickness += (exit_point - entry_point);
        }
    }

    return total_thickness;
}

__device__ FP_T match_pairs(
    const List<int>& candidates, const List<FP_T>& tvalues, const Ray& ray,
    FP_T4* __restrict__ vertices,
    FP_T4* __restrict__ normals,
    const Tree& tree)
{
    // Return early if there's nothing to pair up.
    if (tvalues.size() < 2) {
        return (0.0);
    }

    FP_T total_thickness = (0.0);
    bool is_inside = false;
    FP_T entry_t = (0.0);

    for (int i = 0; i < tvalues.size(); i++) {
        // This simple toggle is more robust than counting normals.
        // Every other intersection point flips the state.
        
        if (!is_inside) {
            // We are currently outside, so this hit is an ENTRY.
            entry_t = tvalues.values[i];
            is_inside = true;
        } else {
            // We were inside, so this hit is an EXIT.
            FP_T exit_t = tvalues.values[i];
            total_thickness += (exit_t - entry_t);
            is_inside = false;
        }
    }

    // It's possible to end in an "inside" state if there's an odd number
    // of intersections (e.g., ray starts inside a non-closed mesh).
    // In most cases, we can ignore this, as the paired segments are what matter.

    return total_thickness;
}

__device__ void unique_from_sorted_with_epsilon(List<float>& list, const FP_T scale) {
    if (list.count <= 1) {
        return;
    }

    AreFPValuesClose are_close(scale);

    int unique_idx = 1; // Index for the next unique element
    for (int i = 1; i < list.count; i++) {
        if (!are_close(list.values[i], list.values[i - 1])) {
            // If it's a new unique element, move it to the next unique slot.
            if (i != unique_idx) { // Avoid self-assignment
               list.values[unique_idx] = list.values[i];
            }
            unique_idx++;
        }
    }

    list.count = unique_idx;
}

template<typename RayType>
__device__ FP_T traceRay(
    const RayType &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T &scale, unsigned row, unsigned col)
{
    List<int> candidates, intersected;
    List<FP_T> tvalues;

    
    #ifdef DEBUG
    bool is_debug_thread = (row == debug_row) && (col == debug_col);
    if (is_debug_thread) {
        printf("\n--- DEBUG TRACE FOR PIXEL (%d, %d) ---\n", row, col);
        printf("\tRay Origin:    (%f, %f, %f)\n", ray.tail.x, ray.tail.y, ray.tail.z);
        printf("\tRay Direction: (%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
        printf("\tBVH nb_keys:   %u\n", tree.nb_keys);
    }
    #endif

    // This is where the acceleration structure (BVH) is actually useful
    query<RayType>(tree, ray, candidates, row, col);

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 1] BVH Query: Found %u candidate triangles.\n", candidates.size());
    }
    #endif

    if (candidates.size() == 0)
    {
        return 0.0;
    }
    
    // Test the candidates for actual intersections
    for (unsigned i = 0; i < candidates.size(); i++)
    {
        unsigned int original_triangle_idx = candidates.get(i);
        int primIndex = original_triangle_idx * 3;
        
        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0;
        bool hit = ray.intersects(V1, V2, V3, t, col, row);

        #ifdef DEBUG
        if (is_debug_thread) {
            printf("\t\t-> Testing candidate triangle %u... Result: %s, t-value: %f\n",
                original_triangle_idx,
                hit ? "HIT" : "MISS",
                hit ? t : 0.0f);
        }
        #endif

        if (hit) {
            tvalues.push_back(t);
        }
    }

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2] Intersection Test: Found %u actual intersections.\n", tvalues.size());
        if (tvalues.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 0; i < tvalues.size(); ++i) {
                printf("%f ", tvalues.get(i));
            }
            printf("\n");
        }
    }
    #endif

    if (tvalues.size() == 0 || tvalues.size() == 1)
    {
        return 0.0;
    }

    sort(tvalues.values, tvalues.size());

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2.1]sorted %d\n", tvalues.size());
        if (tvalues.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 0; i < tvalues.size(); ++i) {
                printf("%f ", tvalues.get(i));
            }
            printf("\n");
        }
    }
    #endif

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2.15]is equal %d\n", tvalues.size());
        if (tvalues.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 1; i < tvalues.size(); ++i) {
                printf("%d ", tvalues.get(i) == tvalues.get(i-1));
            }
            printf("\n");
        }
    }
    #endif

    unique_from_sorted_with_epsilon(tvalues, scale);
    
    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2.2] unique %d\n", tvalues.size());
        if (tvalues.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 0; i < tvalues.size(); ++i) {
                printf("%f ", tvalues.get(i));
            }
            printf("\n");
        }
    }
    #endif
    
    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2.3] eveness %d\n", tvalues.size());
        if (tvalues.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 0; i < tvalues.size(); ++i) {
                printf("%f ", tvalues.get(i));
            }
            printf("\n");
        }
    }
    #endif

    FP_T final_thickness = project_thickness(tvalues);

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 3] Thickness Calculation: Final_Thickness = %f\n", final_thickness);
        printf("--- END TRACE ---\n\n");
    }
    #endif

    return final_thickness;
}

// __device__ FP_T traceRay(
//     const Ray &ray, const Tree &tree,
//     FP_T4 *__restrict__ vertices,
//     FP_T4 *__restrict__ normals,
//     FP_T &epsilon)
// {
//     List<int> candidates, intersected;
//     List<FP_T> tvalues;

//     // This is where the acceleration structure (BVH) is actually useful
//     query(tree, ray, candidates);

//     if (candidates.size() == 0)
//     {
//         return 0.0;
//     }

//     // Test the candidates for actual intersections
//     for (int i = 0; i < candidates.size(); i++)
//     {
//         int primIndex = candidates.get(i) * 3;

//         const FP_T4 V1 = vertices[primIndex];
//         const FP_T4 V2 = vertices[primIndex + 1];
//         const FP_T4 V3 = vertices[primIndex + 2];

//         FP_T t = 0.0, tmax = INFINITY;
//         if (ray.intersects(V1, V2, V3, t, tmax))
//         {
//             tvalues.push_back(t);
//             intersected.push_back(candidates.get(i));
//         }
//     }

//     if (tvalues.size() == 0)
//     {
//         return 0.0;
//     }

//     if (tvalues.size() == 2)
//     {
//         return (fabs)(tvalues.get(1) - tvalues.get(0));
//     }

//     // thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.size(),
//     //     intersected.values);

//     List<FP_T> filtered_tvalues;
//     List<int> filtered_candidates;

//     filtered_tvalues.count = 0;
//     filtered_candidates.count = 0;

//     AreFPValuesClose_Final close(epsilon);

//     if (tvalues.size() > 0) {
//         // Keep the first t-value AND its corresponding candidate
//         filtered_tvalues.push_back(tvalues.get(0));
//         filtered_candidates.push_back(candidates.get(0));

//         for (int i = 1; i < tvalues.size(); ++i) {
//             FP_T current_t = tvalues.get(i);
//             FP_T last_unique_t = filtered_tvalues.back();
//             bool is_close = close(current_t, last_unique_t);

//             if (!is_close) {
//                 // If the t-value is unique, keep BOTH it and its candidate from the same index
//                 filtered_tvalues.push_back(current_t);
//                 filtered_candidates.push_back(candidates.get(i));
//             }
//         }
//     }

//     // return project_thickness(filtered_tvalues); // Your original commented out line
//     // return match_pairs(filtered_intersections, filtered_tvalues, ray, vertices, normals, tree);
//     return match_pairs(filtered_candidates, filtered_tvalues, ray, vertices, normals, tree);
// }

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
    FP_T u, FP_T v,
    const FP_T4& base_pixel_origin, const FP_T4& scaled_U, const FP_T4& scaled_V,
    const FP_T4& W, const FP_T4& scene_bbMin, const FP_T4& scene_bbMax,
    Tree& tree, FP_T4 *__restrict__ vertices, FP_T min_feature_size)
{
    // Calculate the precise origin for this sub-pixel ray
    FP_T4 sample_origin = base_pixel_origin - scaled_U * u - scaled_V * v;
    WatertightRay ray = WatertightRay(sample_origin, W, scene_bbMin, scene_bbMax);
    return traceRay<WatertightRay>(ray, tree, vertices, min_feature_size, 0, 0); // row/col args for debug not used here
}

__device__ FP_T traceAndCache(
    float u, float v,
    FP_T* value_cache, bool* is_cached,
    // (Original traceSubPixel parameters)
    FP_T4 base_pixel_origin, FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 W,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices, FP_T min_feature_size
) {
    // 1. Calculate the integer grid coordinates for the cache lookup.
    int ix = roundf(u * (CACHE_DIM - 1));
    int iy = roundf(v * (CACHE_DIM - 1));
    int index = iy * CACHE_DIM + ix;

    // 2. Check if the value is already in our cache.
    if (is_cached[index]) {
        return value_cache[index]; // Return cached value instantly.
    }

    // 3. If not cached, perform the expensive trace.
    FP_T value = traceSubPixel(u, v, base_pixel_origin, scaled_U, scaled_V, W, 
                               scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
    
    // 4. Store the new value in the cache and mark it as valid.
    value_cache[index] = value;
    is_cached[index] = true;

    return value;
}

__device__ FP_T tracePixelAdaptive(
    // Per-pixel inputs
    FP_T4 base_pixel_origin,
    // Global scene & camera data
    FP_T4 scaled_U, FP_T4 scaled_V, FP_T4 W,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    Tree &tree, FP_T4 *__restrict__ vertices,
    // Control parameters
    FP_T min_feature_size, FP_T abs_tolerance, FP_T rel_tolerance, int depth
) {
    // 1. INITIALIZATION
    Quad quad_stack[MAX_QUADS_PER_PIXEL];
    int stack_ptr = 0;

    FinalQuad final_quads[MAX_QUADS_PER_PIXEL];
    int final_quads_count = 0;
    
    FP_T value_cache[CACHE_SIZE];
    bool is_cached[CACHE_SIZE];

    for(int i = 0; i < CACHE_SIZE; ++i) {
        is_cached[i] = false;
    }

    quad_stack[stack_ptr++] = {0.0f, 0.0f, 1.0f, 0};

    // 2. ADAPTIVE SUBDIVISION LOOP
    while (stack_ptr > 0)
    {
        Quad current_quad = quad_stack[--stack_ptr];

        float u = current_quad.u, v = current_quad.v, s = current_quad.size, hs = s / 2.0f;
        FP_T values[5];
        values[0] = traceAndCache(u,      v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, W, scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        values[1] = traceAndCache(u + s,  v,      value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, W, scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        values[2] = traceAndCache(u,      v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, W, scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        values[3] = traceAndCache(u + s,  v + s,  value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, W, scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        values[4] = traceAndCache(u + hs, v + hs, value_cache, is_cached, base_pixel_origin, scaled_U, scaled_V, W, scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        
        FP_T min_val = values[0], max_val = values[0];
        for (int i = 1; i < 5; ++i) {
            if (values[i] < min_val) min_val = values[i];
            if (values[i] > max_val) max_val = values[i];
        }

        // 3. DECISION
        FP_T threshold = fmaxf(abs_tolerance, rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL))
        {
            if (final_quads_count < MAX_QUADS_PER_PIXEL) {
                FP_T avg_value = (values[0] + values[1] + values[2] + values[3] + values[4]) / 5.0f;
                final_quads[final_quads_count++] = {avg_value, s * s};
            }
        }
        else
        {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u,      v,      hs, next_depth};
            quad_stack[stack_ptr++] = {u + hs, v,      hs, next_depth};
            quad_stack[stack_ptr++] = {u,      v + hs, hs, next_depth};
            quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }

    // 4. FINAL AVERAGING
    FP_T total_value = 0.0f;
    FP_T total_area = 0.0f;
    for (int i = 0; i < final_quads_count; ++i) {
        total_value += final_quads[i].value * final_quads[i].area;
        total_area  += final_quads[i].area;
    }
    
    return (total_area > 0.0f) ? (total_value / total_area) : 0.0f;
}

extern "C" __global__ void project_parallel_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W,
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope, int *left, unsigned *permutation,
    FP_T4 *bboxMin, FP_T4 *bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter, FP_T min_feature_size, int supersampling,
    FP_T abs_tolerance, FP_T rel_tolerance
)
{
    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;
    
    unsigned int totalPixels = N.x * N.y;

    // --- Persistent Thread Loop ---
    // Minor performance improvement because of the low quality BVH
    // TODO: balance workload by quad subdivision, not by ray
    while (true)
    {
        unsigned int index = atomicAdd(globalCounter, 1);
        if (index >= totalPixels) {
            break; 
        }

        // --- Per-Pixel Processing (Now much cleaner) ---
        unsigned int row = index / N.x;
        unsigned int col = index % N.x;
        FP_T4 base_pixel_origin = upperleft_origin - scaled_U * col - scaled_V * row;

        if ((supersampling > 0) && (supersampling <= MAX_DEPTH))
        {
            image[index] = tracePixelAdaptive(
                base_pixel_origin,
                scaled_U, scaled_V, W,
                scene_bbMin, scene_bbMax,
                tree, vertices,
                min_feature_size, abs_tolerance, rel_tolerance, supersampling
            );
        }
        else // Simple, single-ray tracing mode
        {
            image[index] = traceSubPixel(0.5f, 0.5f, base_pixel_origin, scaled_U, scaled_V, W,
                                         scene_bbMin, scene_bbMax, tree, vertices, min_feature_size);
        }
    }
}


// extern "C" __global__ void project_conebeam_kernel(
//     unsigned nb_keys, FP_T *image, uint2 N,
//     FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
//     FP_T4 upperleft_origin, FP_T2 ps,
//     int *rope,
//     int *left,
//     unsigned *permutation, // BVH tree
//     FP_T4 *bboxMin,
//     FP_T4 *bboxMax,
//     FP_T4 *__restrict__ vertices,
//     unsigned *globalCounter,
//     FP_T4 source,
//     FP_T epsilon
// )
// {
//     // Setup the tree structure once per thread block or globally as needed
//     Tree tree;
//     tree.nb_keys = nb_keys;
//     tree.rope = rope;
//     tree.left = left;
//     tree.indices = permutation;
//     tree.bboxMin = bboxMin;
//     tree.bboxMax = bboxMax;

//     FP_T4 scaled_U = U * ps.x;
//     FP_T4 scaled_V = V * ps.y;

//     // Calculate the total number of rays using N.x and N.y
//     unsigned totalRays = N.x * N.y;

//     // Loop until all rays have been processed
//     while (true)
//     {
//         // Atomically fetch the next index to process
//         unsigned index = atomicAdd(globalCounter, 1);
//         if (index >= totalRays)
//             break; // No more rays to process

//         // Convert the 1D index to 2D coordinates for the ray position
//         unsigned row = index / N.x;
//         unsigned col = index % N.x;

//         FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
//         FP_T4 direction = source - pixel_coordinates;
//         Ray ray = Ray(pixel_coordinates, direction);
//         // image[index] = traceRay(ray, tree, vertices, epsilon);
//     }
// }

// extern "C" __global__ void project_conebeam_normals_kernel(
//     unsigned nb_keys, FP_T *image, uint2 N,
//     FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
//     FP_T4 upperleft_origin, FP_T2 ps,
//     int *rope,
//     int *left,
//     unsigned *permutation, // BVH tree
//     FP_T4 *bboxMin,
//     FP_T4 *bboxMax,
//     FP_T4 *__restrict__ vertices,
//     unsigned *globalCounter,
//     FP_T4 *__restrict__ normals,
//     FP_T4 source,
//     FP_T epsilon
// )
// {
//     // Setup the tree structure once per thread block or globally as needed
//     Tree tree;
//     tree.nb_keys = nb_keys;
//     tree.rope = rope;
//     tree.left = left;
//     tree.indices = permutation;
//     tree.bboxMin = bboxMin;
//     tree.bboxMax = bboxMax;

//     FP_T4 scaled_U = U * ps.x;
//     FP_T4 scaled_V = V * ps.y;

//     // Calculate the total number of rays using N.x and N.y
//     unsigned totalRays = N.x * N.y;

//     // Loop until all rays have been processed
//     while (true)
//     {
//         // Atomically fetch the next index to process
//         unsigned index = atomicAdd(globalCounter, 1);
//         if (index >= totalRays)
//             break; // No more rays to process

//         // Convert the 1D index to 2D coordinates for the ray position
//         unsigned row = index / N.x;
//         unsigned col = index % N.x;

//         FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
//         FP_T4 direction = source - pixel_coordinates;
//         Ray ray = Ray(pixel_coordinates, direction);
//         // image[index] = traceRay(ray, tree, vertices, normals, epsilon);
//     }
// }