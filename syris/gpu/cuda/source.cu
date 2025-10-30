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
template<typename RayType>
__device__ FP_T match_pairs(
    const List<HitRecord>& hits, const RayType& ray,
    FP_T4* __restrict__ normals,
    // Debug parameters
    unsigned row, unsigned col)
{
    const FP_T epsilon = ray.m_epsilons.group_abs_epsilon;
    const FP_T rel_epsilon = ray.m_epsilons.group_rel_epsilon;

    // Set up the debug flag
    #ifdef DEBUG
    bool is_debug_thread = (row == debug_row) && (col == debug_col);
    if (is_debug_thread) {
        printf("\t[Stage 3] match pairs starting. hits.size() = %d. Using ABS_EPS=%f, REL_FAC=%f\n",
               hits.size(), epsilon, rel_epsilon);
    }
    #endif

    if (hits.size() < 2) {
        return (0.0);
    }

    FP_T total_thickness = (0.0);
    FP_T t_enter = (0.0);
    int inside_count = 0; 

    int i = 0;
    while (i < hits.size()) {
        const FP_T t_current_event = hits.values[i].t;
        
        // --- ROBUST DYNAMIC EPSILON ---
        // Calculate the tolerance for *this specific t-value*.
        // It's the larger of an absolute tolerance (for t ~ 0)
        // and a relative tolerance (for t > 1).
        const FP_T dynamic_epsilon = fmaxf(
            epsilon,
            rel_epsilon * fabsf(t_current_event)
        );
        // ---

        #ifdef DEBUG
        if (is_debug_thread) {
            printf("\t  Processing event at t = %.8f (i = %d). dynamic_epsilon = %.8g\n",
                   t_current_event, i, dynamic_epsilon);
        }
        #endif

        int net_change = 0;
        int j = i;

        // 1. Process ALL hits within the DYNAMIC tolerance
        while (j < hits.size() && fabsf(hits.values[j].t - t_current_event) < dynamic_epsilon) {
            
            unsigned int tri_idx = hits.values[j].triangle_idx;
            const FP_T4 N = normals[tri_idx];
            FP_T dot_prod = dot(ray.direction, N);

            #ifdef DEBUG
            if (is_debug_thread) {
                printf("\t    -> Hit (j=%d): tri_idx=%u, t=%.8f, dot_prod=%.8f\n",
                       j, tri_idx, hits.values[j].t, dot_prod);
            }
            #endif

            if (dot_prod < 0.0) {
                net_change++; // ENTRY
            } else if (dot_prod > 0.0) {
                net_change--; // EXIT
            }
            
            j++;
        }

        // 2. Interpret the net change for this t-event
        int prev_inside_count = inside_count;

        if (net_change > 0) {
            if (inside_count == 0) {
                t_enter = t_current_event;
            }
            inside_count++; 
        } 
        else if (net_change < 0) {
            inside_count--; 
        }
        
        // 3. Check for thickness-adding transitions
        // We use t_current_event, which is the t-value of the *first*
        // hit in this group. This is more robust than using the last.
        if (prev_inside_count > 0 && inside_count == 0) {
            total_thickness += (t_current_event - t_enter);
        }

        // 4. Handle invalid states
        if (inside_count < 0) {
            inside_count = 0;
        }

        #ifdef DEBUG
        if (is_debug_thread) {
            printf("\t  Event Summary: net_change=%d, prev_inside=%d, inside_count=%d, t_enter=%.8f, total_thickness=%.8f\n",
                   net_change, prev_inside_count, inside_count, t_enter, total_thickness);
        }
        #endif

        // 5. Move outer loop index past all processed hits
        i = j;
    }

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 3] match_pairs finished. Final thickness: %.8f\n", total_thickness);
    }
    #endif

    return total_thickness;
}

__device__ void unique_from_sorted_with_epsilon(List<FP_T>& list, const EpsilonParams& epsilons) {
    if (list.count <= 1) {
        return;
    }

    AreFPValuesClose are_close(epsilons.unique_abs_epsilon);

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
    FP_T4 *__restrict__ vertices,
    unsigned row, unsigned col)
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

    unique_from_sorted_with_epsilon(tvalues, ray.m_epsilons);
    
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
template<typename RayType>
__device__ FP_T traceRay(
    const RayType &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    unsigned row, unsigned col)
{
    List<int> candidates;
    // Use our new HitRecord struct
    List<HitRecord> hits; 

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

        if (hit) {
            // Store both t and the triangle index
            hits.push_back(HitRecord(t, original_triangle_idx));
        }

        #ifdef DEBUG
        if (is_debug_thread) {
            printf("\t\t-> Testing candidate triangle %u... Result: %s, t-value: %f\n",
                original_triangle_idx,
                hit ? "HIT" : "MISS",
                hit ? t : 0.0f);
        }
        #endif
    }

    // Need at least 2 hits to form a segment
    if (hits.size() < 2)
    {
        return 0.0;
    }

    // Sort the HitRecord list using our new sort function
    sort_hit(hits.values, hits.size());

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 2] Intersection Test: Found %u actual intersections.\n", hits.size());
        if (hits.size() > 0) {
            printf("\t -> t-values: ");
            for(int i = 0; i < hits.size(); ++i) {
                printf("%f ", hits.get(i).t);
            }
            printf("\n");
        }
    }
    #endif

    // Call the new match_pairs function, passing the normals array
    FP_T final_thickness = match_pairs<RayType>(hits, ray, normals, row, col);

    #ifdef DEBUG
    if (is_debug_thread) {
        printf("\t[Stage 3] Thickness Calculation using normals: ret = %f\n", final_thickness);
        printf("---- END TRACE ----\n\n");
    }
    #endif

    return final_thickness;
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
    FP_T norm = rnorm3df(direction.x, direction.y, direction.z);
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
    FP_T norm = rnorm3df(direction.x, direction.y, direction.z);
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
    FinalQuad final_quads[MAX_QUADS_PER_PIXEL]; int final_quads_count = 0;
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
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            if (final_quads_count < MAX_QUADS_PER_PIXEL) {
                FP_T spread_low  = values[2] - values[0];
                FP_T spread_high = values[4] - values[2];
                // FP_T avg_value = (spread_low < spread_high) ? (values[0] + values[1] + values[2]) / 3.0 : (values[2] + values[3] + values[4]) / 3.0;
                final_quads[final_quads_count++] = {values[2], s * s};
                // final_quads[final_quads_count++] = {avg_value, s * s};
            }
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    FP_T total_value = 0.0, total_area = 0.0;
    for (int i = 0; i < final_quads_count; ++i) { total_value += final_quads[i].value * final_quads[i].area; total_area  += final_quads[i].area; }
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
    FinalQuad final_quads[MAX_QUADS_PER_PIXEL]; int final_quads_count = 0;
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
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            if (final_quads_count < MAX_QUADS_PER_PIXEL) {
                FP_T spread_low  = values[2] - values[0];
                FP_T spread_high = values[4] - values[2];
                // FP_T avg_value = (spread_low < spread_high) ? (values[0] + values[1] + values[2]) / 3.0 : (values[2] + values[3] + values[4]) / 3.0;
                final_quads[final_quads_count++] = {values[2], s * s};
                // final_quads[final_quads_count++] = {avg_value, s * s};
            }
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    FP_T total_value = 0.0, total_area = 0.0;
    for (int i = 0; i < final_quads_count; ++i) { total_value += final_quads[i].value * final_quads[i].area; total_area  += final_quads[i].area; }
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
    FinalQuad final_quads[MAX_QUADS_PER_PIXEL]; int final_quads_count = 0;
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
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            if (final_quads_count < MAX_QUADS_PER_PIXEL) {
                FP_T spread_low  = values[2] - values[0];
                FP_T spread_high = values[4] - values[2];
                // FP_T avg_value = (spread_low < spread_high) ? (values[0] + values[1] + values[2]) / 3.0 : (values[2] + values[3] + values[4]) / 3.0;
                final_quads[final_quads_count++] = {values[2], s * s};
                // final_quads[final_quads_count++] = {avg_value, s * s};
            }
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    FP_T total_value = 0.0, total_area = 0.0;
    for (int i = 0; i < final_quads_count; ++i) { total_value += final_quads[i].value * final_quads[i].area; total_area  += final_quads[i].area; }
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
    FinalQuad final_quads[MAX_QUADS_PER_PIXEL]; int final_quads_count = 0;
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
        sort(values, 5);
        FP_T min_val = values[0], max_val = values[4];
        FP_T threshold = fmax(sampling_params.abs_tolerance, sampling_params.rel_tolerance * max_val);
        if ((max_val - min_val < threshold) || (current_quad.depth >= sampling_params.max_depth) || (stack_ptr + 4 > MAX_QUADS_PER_PIXEL)) {
            if (final_quads_count < MAX_QUADS_PER_PIXEL) {
                FP_T spread_low  = values[2] - values[0];
                FP_T spread_high = values[4] - values[2];
                // FP_T avg_value = (spread_low < spread_high) ? (values[0] + values[1] + values[2]) / 3.0 : (values[2] + values[3] + values[4]) / 3.0;
                final_quads[final_quads_count++] = {values[2], s * s};
                // final_quads[final_quads_count++] = {avg_value, s * s};
            }
        } else {
            int next_depth = current_quad.depth + 1;
            quad_stack[stack_ptr++] = {u, v, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v, hs, next_depth};
            quad_stack[stack_ptr++] = {u, v + hs, hs, next_depth}; quad_stack[stack_ptr++] = {u + hs, v + hs, hs, next_depth};
        }
    }
    FP_T total_value = 0.0, total_area = 0.0;
    for (int i = 0; i < final_quads_count; ++i) { total_value += final_quads[i].value * final_quads[i].area; total_area  += final_quads[i].area; }
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
            FP_T4 direction = source - sample_origin;
            FP_T norm = rnorm3df(direction.x, direction.y, direction.z);
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
            FP_T norm = rnorm3df(direction.x, direction.y, direction.z);
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