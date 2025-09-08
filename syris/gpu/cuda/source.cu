#include "Commons.cuh"
#include "Ray.cuh"

#define SENTINEL -1
#define INVALID -1

typedef unsigned int morton_t;
typedef int delta_t;

struct Hit
{
    FP_T t;
    int primID;
    // The normal is no longer needed here if we pass the global array,
    // but storing it can be useful for debugging. Let's keep it simple for now.
};

// Custom comparator for sorting Hits by t-value
struct HitComparator {
    __device__ bool operator()(const Hit& a, const Hit& b) const {
        return a.t < b.t;
    }
};

typedef struct
{
    unsigned int nb_keys;
    morton_t *keys;
    unsigned int *indices;
    int *entered;
    int *rope;
    int *left;
    FP_T4 *bboxMin;
    FP_T4 *bboxMax;
} Tree;

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
    bbMinOutput->x = FP_MATH(fmin)(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = FP_MATH(fmin)(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = FP_MATH(fmin)(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = FP_MATH(fmax)(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = FP_MATH(fmax)(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = FP_MATH(fmax)(bbMaxInput.z, bbMaxOutput->z);
}

__device__ int delta(const Tree &tree, const int index)
{
    constexpr int MAX = ::cuda::std::numeric_limits<int>::max();
    constexpr int MIN = ::cuda::std::numeric_limits<int>::min();

    if (index < 0 || index >= tree.nb_keys - 1)
    {
        return MAX;
    }

    // TODO: augment the function if the codes are the same
    unsigned int a = tree.keys[index];
    unsigned int b = tree.keys[index + 1];
    int x = a ^ b;
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
            // __syncthreads();
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
            // __syncthreads();
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

__device__ void query(const Tree &tree, const Ray &ray, List<int> &candidates)
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
    constexpr FP_T DEDUP_EPSILON = FP_CONST(1e-6);
    FP_T result = FP_CONST(0.0);

    int i = 0;
    while (i < tvalues.size())
    {
        int j = i + 1;
        while (j < tvalues.size() && are_close(tvalues.values[j], tvalues.values[i], DEDUP_EPSILON))
        {
            j++;
        }
        
        if (i < tvalues.size() && j < tvalues.size())
        {
            result += FP_MATH(fabs)(tvalues.values[j] - tvalues.values[i]);
        }
        i = j + 1;
    }

    return result;
}

template <typename T>
__device__ T sum(List<T> const &list)
{
    T result = static_cast<T>(0);
    for (int i = 0; i < list.size(); i++)
    {
        result += list.values[i];
    }
    return result;
}

inline __device__ FP_T dot(const FP_T4 &a, const FP_T4 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// __device__ FP_T matchOuterPairs(
//     const List<int> &candidates, const List<FP_T> &tvalues, const Ray &ray,
//     FP_T4 *__restrict__ vertices,
//     FP_T4 *__restrict__ normals,
//     const Tree &tree)
// {
//     FP_T thickness = FP_CONST(0.0);
//     FP_T inner = FP_CONST(0.0);
//     int counter = 0;

//     constexpr FP_T DOT_PRODUCT_EPSILON = FP_CONST(1e-6);

//     for (int i = 0; i < tvalues.size(); i++)
//     {
//         unsigned primIndex = candidates.values[i];
//         FP_T dot_product = dot(ray.getDirection(), normals[primIndex]);
        
//         if (is_null<FP_T>(dot_product, DOT_PRODUCT_EPSILON))
//         {
//             continue;
//         }

//         bool is_neg = dot(ray.getDirection(), normals[primIndex]) < 0;
//         if (!is_neg && counter == 0)
//         {
//             continue;
//         }

//         if (is_neg)
//         {
//             if (counter++ == 0)
//                 inner = tvalues.values[i];
//         }
//         else
//         {
//             if (--counter == 0)
//             {
//                 thickness += tvalues.values[i] - inner;
//             }
//         }
//     }
//     return thickness;
// }

__device__ FP_T matchOuterPairs(
    const List<int> &candidates, const List<FP_T> &tvalues, const Ray &ray,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    const Tree &tree)
{
    FP_T total_thickness = FP_CONST(0.0);

    // Iterate through the hits in pairs (entry, exit)
    for (int i = 0; i < tvalues.size(); i += 2)
    {
        // Ensure we have a complete pair to process. This check is vital.
        if (i + 1 < tvalues.size())
        {
            unsigned entry_primID = candidates.get(i);
            unsigned exit_primID = candidates.get(i + 1);

            FP_T entry_dot = dot(ray.getDirection(), normals[entry_primID]);
            FP_T exit_dot = dot(ray.getDirection(), normals[exit_primID]);

            // Sanity Check: A valid pair should be an ENTRY (dot < 0)
            // followed by an EXIT (dot > 0).
            if (entry_dot < 0 && exit_dot > 0)
            {
                FP_T entry_t = tvalues.get(i);
                FP_T exit_t = tvalues.get(i + 1);
                total_thickness += (exit_t - entry_t);
            }
        }
    }
    return total_thickness;
}


struct AreFPValuesClose {
    const FP_T relative_epsilon;

    __device__ AreFPValuesClose(FP_T ep) : relative_epsilon(ep) {}

    __device__ bool operator()(FP_T a, FP_T b) const {
        // A robust relative comparison
        return FP_MATH(fabs)(a - b) <= relative_epsilon * FP_MATH(fmax)(FP_CONST(1.0), FP_MATH(fmax)(FP_MATH(fabs)(a), FP_MATH(fabs)(b)));
    }
};

struct AreTValuesAbsolutelyClose {
    const FP_T t_epsilon; // This will be the value calculated in Python

    __device__ AreTValuesAbsolutelyClose(FP_T ep) : t_epsilon(ep) {}

    __device__ bool operator()(FP_T a, FP_T b) const {
        return FP_MATH(fabs)(a - b) <= t_epsilon;
    }
};

__device__ FP_T traceRay(
    const Ray &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices, FP_T &epsilon)
{
    List<int> candidates, intersected;
    List<FP_T> tvalues;

    // This is where the acceleration structure (BVH) is actually useful
    query(tree, ray, candidates);

    if (candidates.size() == 0)
    {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (unsigned i = 0; i < candidates.size(); i++) // Changed to unsigned to match List::get
    {
        int primIndex = candidates.get(i) * 3;

        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0;
        if (ray.intersects(V1, V2, V3, t))
        {
            tvalues.push_back(t);
            intersected.push_back(candidates.get(i));
        }
    }

    if (tvalues.size() == 0 || tvalues.size() == 1)
    {
        return 0.0;
    }

    thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.size(),
        intersected.values);

    // Filter duplicates
    List<int> filtered_intersections;
    List<FP_T> filtered_tvalues;

    thrust::pair<FP_T*, int*> new_ends;

    new_ends = thrust::unique_by_key_copy(
        thrust::seq,                             // Explicit sequential execution policy
        tvalues.values,                          // Input keys: start
        tvalues.values + tvalues.size(),         // Input keys: end
        intersected.values,                      // Input values: start (must match key range)
        filtered_tvalues.values,                 // Output keys: destination
        filtered_intersections.values,           // Output values: destination
        AreFPValuesClose(epsilon)                // Custom predicate for "equality"
    );

    // Update the counts in your filtered lists
    filtered_tvalues.count = new_ends.first - filtered_tvalues.values;
    filtered_intersections.count = new_ends.second - filtered_intersections.values;

    return project_thickness(filtered_tvalues);
}


__device__ FP_T traceRay(
    const Ray &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    FP_T &epsilon)
{
    List<int> candidates, intersected;
    List<FP_T> tvalues;

    // This is where the acceleration structure (BVH) is actually useful
    query(tree, ray, candidates);

    if (candidates.size() == 0)
    {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.size(); i++)
    {
        int primIndex = candidates.get(i) * 3;

        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0, tmax = INFINITY;
        if (ray.intersects(V1, V2, V3, t, tmax))
        {
            tvalues.push_back(t);
            intersected.push_back(candidates.get(i));
        }
    }

    if (tvalues.size() == 0)
    {
        return 0.0;
    }

    if (tvalues.size() == 2)
    {
        return FP_MATH(fabs)(tvalues.get(1) - tvalues.get(0));
    }

    thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.size(),
        intersected.values);

    // Filter duplicates
    List<int> filtered_intersections;
    List<FP_T> filtered_tvalues;

    thrust::pair<FP_T*, int*> new_ends;

    new_ends = thrust::unique_by_key_copy(
        thrust::seq,                             // Explicit sequential execution policy
        tvalues.values,                          // Input keys: start
        tvalues.values + tvalues.size(),         // Input keys: end
        intersected.values,                      // Input values: start (must match key range)
        filtered_tvalues.values,                 // Output keys: destination
        filtered_intersections.values,           // Output values: destination
        AreFPValuesClose(epsilon)          // Custom predicate for "equality"
    );

    // Update the counts in your filtered lists
    filtered_tvalues.count = new_ends.first - filtered_tvalues.values;
    filtered_intersections.count = new_ends.second - filtered_intersections.values;

    if (filtered_tvalues.size() == 0) return 0.0; // or TRACE_OK

    if (filtered_tvalues.size() % 2 != 0) {
        return 0; // This is a true error
    }

    // return project_thickness(filtered_tvalues); // Your original commented out line
    return matchOuterPairs(filtered_intersections, filtered_tvalues, ray, vertices, normals, tree);
}

__device__ FP_T matchPairs_Winding(
    const List<Hit> &hits,
    const Ray &ray,
    FP_T4 *__restrict__ normals)
{
    FP_T total_thickness = FP_CONST(0.0);
    FP_T entry_t = FP_CONST(0.0);
    int winding_counter = 0;

    for (int i = 0; i < hits.size(); i++)
    {
        const Hit& current_hit = hits.get(i);
        FP_T dot_product = dot(ray.getDirection(), normals[current_hit.primID]);

        // Ignore grazing angles, which are numerically unstable.
        if (FP_MATH(fabs)(dot_product) < FP_CONST(1e-7))
        {
            continue;
        }

        bool is_entry = dot_product < 0;

        if (is_entry)
        {
            // If this is the FIRST entry into any surface, record the t value.
            if (winding_counter == 0)
            {
                entry_t = current_hit.t;
            }
            winding_counter++;
        }
        else // Is an exit
        {
            winding_counter--;
            // If this exit brings us completely OUTSIDE all surfaces, add the segment to the total thickness.
            if (winding_counter == 0 && entry_t > FP_CONST(0.0))
            {
                total_thickness += (current_hit.t - entry_t);
                entry_t = FP_CONST(0.0); // Reset for the next segment
            }
        }
    }
    return total_thickness;
}

__device__ FP_T traceRay_Robust(
    const Ray &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals)
{
    List<int> candidates;
    query(tree, ray, candidates);

    if (candidates.size() == 0)
    {
        return 0.0;
    }

    // 1. Collect all valid intersections into a single list of Hits
    List<Hit> hits;
    for (int i = 0; i < candidates.size(); i++)
    {
        int primID = candidates.get(i);
        int primIndex = primID * 3;

        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0;
        if (ray.intersects(V1, V2, V3, t))
        {
            // Only consider hits in front of the ray
            if (t > 0) {
                 hits.push_back({t, primID});
            }
        }
    }

    if (hits.size() < 2)
    {
        return 0.0;
    }

    // 2. Sort all hits by their t-value
    thrust::sort(thrust::seq, hits.values, hits.values + hits.size(), HitComparator());

    // 3. Manually filter duplicates using a dynamic, angle-aware epsilon
    List<Hit> filtered_hits;
    if (hits.size() > 0)
    {
        filtered_hits.push_back(hits.get(0)); // Always accept the first hit

        for (int i = 1; i < hits.size(); ++i)
        {
            const Hit& current_hit = hits.get(i);
            const Hit& prev_filtered_hit = filtered_hits.back();

            // Calculate the dynamic epsilon based on the PREVIOUS accepted hit's normal
            FP_T dot_product = FP_MATH(fabs)(dot(ray.getDirection(), normals[prev_filtered_hit.primID]));

            // Prevent division by zero and handle grazing angles robustly
            // A larger clamp (e.g., 1e-5) makes the filter more aggressive
            dot_product = FP_MATH(fmax)(dot_product, FP_CONST(1e-9));

            FP_T t_epsilon = FP_CONST(1e-9) / dot_product;

            // If the current hit is sufficiently far from the last one, accept it.
            if ((current_hit.t - prev_filtered_hit.t) > t_epsilon)
            {
                filtered_hits.push_back(current_hit);
            }
        }
    }

    if (filtered_hits.size() < 2)
    {
        return 0.0;
    }

    // 4. Calculate thickness using the robust winding number algorithm
    return matchPairs_Winding(filtered_hits, ray, normals);
}


__device__ FP_T traceRay_DEBUG(
    const Ray &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    FP_T &epsilon)
{
    List<int> candidates, intersected;
    List<FP_T> tvalues;

    // This is where the acceleration structure (BVH) is actually useful
    query(tree, ray, candidates);

    if (candidates.size() == 0)
    {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.size(); i++)
    {
        int primIndex = candidates.get(i) * 3;

        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0, tmax = INFINITY;
        if (ray.intersects(V1, V2, V3, t, tmax))
        {
            tvalues.push_back(t);
            intersected.push_back(candidates.get(i));
        }
    }

    if (tvalues.size() == 0)
    {
        return 0.0;
    }

    if (tvalues.size() == 2)
    {
        return FP_MATH(fabs)(tvalues.get(1) - tvalues.get(0));
    }

    thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.size(),
        intersected.values);

    // Filter duplicates
    List<int> filtered_intersections;
    List<FP_T> filtered_tvalues;

    thrust::pair<FP_T*, int*> new_ends;

    new_ends = thrust::unique_by_key_copy(
        thrust::seq,                             // Explicit sequential execution policy
        tvalues.values,                          // Input keys: start
        tvalues.values + tvalues.size(),         // Input keys: end
        intersected.values,                      // Input values: start (must match key range)
        filtered_tvalues.values,                 // Output keys: destination
        filtered_intersections.values,           // Output values: destination
        AreFPValuesClose(epsilon)          // Custom predicate for "equality"
    );

    // Update the counts in your filtered lists
    filtered_tvalues.count = new_ends.first - filtered_tvalues.values;
    filtered_intersections.count = new_ends.second - filtered_intersections.values;

    printf("--- DEBUG for PIXEL (%u, %u) ---\n", 1020, 439); // Assuming you can get col/row here
    printf("Found %d unique intersections to pair:\n", filtered_intersections.size());

    // --- The key new printout ---
    for (int i = 0; i < filtered_intersections.size(); i++) {
        unsigned primIndex = filtered_intersections.get(i);
        FP_T t_val = filtered_tvalues.get(i);
        FP_T4 normal = normals[primIndex];
        FP_T dot_product = dot(ray.getDirection(), normal);
        
        printf("  Hit %d: t=%.17f, primID=%u, Normal=(%.3f, %.3f, %.3f), Dot=%.6f, Type=%s\n",
               i,
               t_val,
               primIndex,
               normal.x, normal.y, normal.z,
               dot_product,
               (dot_product < 0 ? "ENTRY" : "EXIT")
        );
    }

    if (filtered_tvalues.size() == 0) return 0.0; // or TRACE_OK

    if (filtered_tvalues.size() % 2 != 0) {
        return 0; // This is a true error
    }

    FP_T final_thickness = matchOuterPairs(filtered_intersections, filtered_tvalues, ray, vertices, normals, tree);
    printf("Final calculated thickness: %.17f\n", final_thickness);
    printf("----------------------------------\n");

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
    unsigned int const nb_keys, FP_T4 const *vertices, unsigned int *keys,
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
        morton_t mortonCode = calculateMortonCode(normalizedCentroid);

        // Store the morton code
        keys[index] = mortonCode;

        index += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void growTreeKernel(
    unsigned int nb_keys, unsigned int *keys, unsigned int *permutation,
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

extern "C" __global__ void project_parallel_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter,
    FP_T epsilon
)
{
    // Setup the tree structure once per thread block or globally as needed
    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;

    // Calculate the total number of rays using N.x and N.y
    unsigned totalRays = N.x * N.y;

    // Loop until all rays have been processed
    while (true)
    {
        // Atomically fetch the next index to process
        unsigned index = atomicAdd(globalCounter, 1);
        if (index >= totalRays)
            break; // No more rays to process

        // Convert the 1D index to 2D coordinates for the ray position
        unsigned row = index / N.x;
        unsigned col = index % N.x;

        FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        Ray ray = Ray(pixel_coordinates, W);
        image[index] = traceRay(ray, tree, vertices, epsilon);
    }
}

extern "C" __global__ void project_parallel_normals_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter,
    FP_T4 *__restrict__ normals,
    FP_T epsilon
)
{
    // Setup the tree structure once per thread block or globally as needed
    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;
    
    // Calculate the total number of rays using N.x and N.y
    unsigned totalRays = N.x * N.y;

    while (true)
    {
        // Atomically fetch the next index to process
        unsigned index = atomicAdd(globalCounter, 1);
        if (index >= totalRays)
            break; // No more rays to process

        // Convert the 1D index to 2D coordinates for the ray position
        unsigned row = index / N.x;
        unsigned col = index % N.x;

        // if (col == 439 && row == 1020) {
        //     FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        //     Ray ray = Ray(pixel_coordinates, W);

        //     // Call a special debug version of traceRay
        //     image[index] = traceRay_DEBUG(ray, tree, vertices, normals, epsilon);
        // } else {
        //     // Normal execution for all other pixels
        FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        Ray ray = Ray(pixel_coordinates, W);
        image[index] = traceRay_Robust(ray, tree, vertices, normals);
        // }
    }
}

extern "C" __global__ void project_conebeam_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter,
    FP_T4 source,
    FP_T epsilon
)
{
    // Setup the tree structure once per thread block or globally as needed
    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;

    // Calculate the total number of rays using N.x and N.y
    unsigned totalRays = N.x * N.y;

    // Loop until all rays have been processed
    while (true)
    {
        // Atomically fetch the next index to process
        unsigned index = atomicAdd(globalCounter, 1);
        if (index >= totalRays)
            break; // No more rays to process

        // Convert the 1D index to 2D coordinates for the ray position
        unsigned row = index / N.x;
        unsigned col = index % N.x;

        FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        FP_T4 direction = source - pixel_coordinates;
        Ray ray = Ray(pixel_coordinates, direction);
        image[index] = traceRay(ray, tree, vertices, epsilon);
    }
}

extern "C" __global__ void project_conebeam_normals_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter,
    FP_T4 *__restrict__ normals,
    FP_T4 source,
    FP_T epsilon
)
{
    // Setup the tree structure once per thread block or globally as needed
    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;

    // Calculate the total number of rays using N.x and N.y
    unsigned totalRays = N.x * N.y;

    // Loop until all rays have been processed
    while (true)
    {
        // Atomically fetch the next index to process
        unsigned index = atomicAdd(globalCounter, 1);
        if (index >= totalRays)
            break; // No more rays to process

        // Convert the 1D index to 2D coordinates for the ray position
        unsigned row = index / N.x;
        unsigned col = index % N.x;

        FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        FP_T4 direction = source - pixel_coordinates;
        Ray ray = Ray(pixel_coordinates, direction);
        image[index] = traceRay(ray, tree, vertices, normals, epsilon);
    }
}