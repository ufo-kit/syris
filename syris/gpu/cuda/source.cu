#include "Commons.cuh"
#include "Ray.cuh"

#define SENTINEL -1
#define INVALID -1

typedef unsigned int morton_t;
typedef int delta_t;

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
    bbMinOutput->x = fminf(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = fminf(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = fminf(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = fmaxf(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = fmaxf(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = fmaxf(bbMaxInput.z, bbMaxOutput->z);
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
        int right_child;
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

            right_child = apetrei_parent + 1;

            // Memory sync
            __threadfence();

            if (right_child != range_right)
            {
                right_child = toInternalRepresentation(tree, right_child);
            }

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

            // Memory sync
            __threadfence();

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
                if (candidates.count == MAX_COLLISIONS)
                {
                    return;
                }
                candidates.values[candidates.count++] = current_node;
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

__device__ FP_T computeThickness(List<FP_T> &tvalues)
{
    FP_T result = 0.0;
    FP_T epsilon = 1e-6;
    int i = 0, j = 1;
    if (tvalues.count == 0)
    {
        return 0.0;
    }

    FP_T t1 = tvalues.values[i];
    while (j < tvalues.count)
    {
        FP_T t2 = tvalues.values[j];
        FP_T d = fabsf(t2 - t1);
        if (d > epsilon)
        {
            result += d;
            t1 = t2;
            j++;
        }
        j++;
    }

    return result;
}

__device__ FP_T project_thickness(List<FP_T> &tvalues)
{
    int i, j;
    FP_T result = 0.0;

    i = 0;
    while (i < tvalues.count)
    {
        j = i + 1;
        while (j < tvalues.count && fabsf(tvalues.values[j] - tvalues.values[i]) < 1e-6)
        {
            j++;
        }
        if (i < tvalues.count && j < tvalues.count)
        {
            result += fabsf(tvalues.values[j] - tvalues.values[i]);
        }
        i = j + 1;
    }

    return result;
}

__device__ FP_T sumTvalues(List<FP_T> &tvalues)
{
    FP_T result = 0.0;
    for (int i = 0; i < tvalues.count; i++)
    {
        result += tvalues.values[i];
    }
    return result;
}

inline __device__ FP_T dot(const FP_T4 &a, const FP_T4 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ FP_T matchOuterPairs(
    const List<int> &candidates, const List<FP_T> &tvalues, const Ray &ray,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals,
    const Tree &tree)
{
    FP_T thickness = 0.0, inner = 0.0;
    int counter = 0;

    for (int i = 0; i < tvalues.count; i++)
    {
        unsigned primIndex = candidates.values[i];
        FP_T dot_product = dot(ray.getDirection(), normals[primIndex]);
        if (is_null(dot_product))
        {
            continue;
        }

        bool is_neg = dot(ray.getDirection(), normals[primIndex]) < 0;
        if (!is_neg && counter == 0)
        {
            continue;
        }

        if (is_neg)
        {
            if (counter++ == 0)
                inner = tvalues.values[i];
        }
        else
        {
            if (--counter == 0)
            {
                thickness += tvalues.values[i] - inner;
            }
        }
    }
    return thickness;
}

struct FloatEpsilonComparator
{
    __device__ bool operator()(FP_T a, FP_T b) const {
        return is_null(a - b);
    }
};
__device__ FP_T traceParallelRay(
    const Ray &ray, const Tree &tree,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals)
{

    List<int> candidates;
    List<FP_T> tvalues;
    candidates.count = 0;
    tvalues.count = 0;

    // This is where the acceleration structure (BVH) is actually usefull
    query(tree, ray, candidates);

    if (candidates.count == 0)
    {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++)
    {
        int primIndex = candidates.values[i] * 3;

        const FP_T4 V1 = vertices[primIndex];
        const FP_T4 V2 = vertices[primIndex + 1];
        const FP_T4 V3 = vertices[primIndex + 2];

        FP_T t = 0.0, tmax = INFINITY;
        if (ray.intersects(V1, V2, V3, t, tmax))
        {
            candidates.values[tvalues.count] = candidates.values[i];
            tvalues.values[tvalues.count++] = t;
        }
    }

    if (tvalues.count < 2)
    {
        return 0.0;
    }
    // if (tvalues.count == 2) {
    //     return fabsf(tvalues.values[1] - tvalues.values[0]);
    // }

    int index[MAX_COLLISIONS];
    thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.count, candidates.values);

    // Remove duplicates
    List<FP_T> new_tvalues;
    new_tvalues.count = 1;
    new_tvalues.values[0] = tvalues.values[0];
    List<int> new_candidates;
    new_candidates.count = 1;
    new_candidates.values[0] = candidates.values[0];
    for (int i = 1; i < tvalues.count; i++)
    {
        if (!is_null(tvalues.values[i] - tvalues.values[i - 1], 1e-9))
        {
            new_tvalues.values[new_tvalues.count] = tvalues.values[i];
            new_candidates.values[new_candidates.count] = candidates.values[i];
            new_tvalues.count++;
            new_candidates.count++;
        }
    }
    // int newCount = thrust::unique_by_key(
    //     thrust::device,
    //     tvalues.values,
    //     tvalues.values + tvalues.count,
    //     candidates.values,
    //     FloatEpsilonComparator()
    //   ).first - tvalues.values;

    // tvalues.count = newCount;
    // candidates.count = newCount;

    // compute the thickness
    // return matchOuterPairs (new_candidates, new_tvalues, ray, vertices, normals, tree);
    // FP_T val = project_thickness(new_tvalues);
    // FP_T val = computeThickness(tvalues);
    return new_tvalues.count;
    // FP_T val = candidates.count;
    // return sumTvalues(new_tvalues);
}

// __device__ FP_T traceRay (
//         Ray &ray, Tree &tree,
//         FP_T4* __restrict__ vertices,
//         FP_T4* __restrict__ normals) {
//     List candidates;
//     candidates.count = 0;
//     memset(candidates.values, 0, MAX_COLLISIONS * sizeof(int));

//     List tvalues;
//     tvalues.count = 0;
//     memset(tvalues.values, 0, MAX_COLLISIONS * sizeof(FP_T));

//     // This is where the acceleration structure (BVH) is actually usefull
//     query(tree, ray, candidates);

//     if (candidates.count == 0) {
//         return 0.0;
//     }

//     // Test the candidates for actual intersections
//     for (int i = 0; i < candidates.count; i++) {
//         int primIndex = candidates.values[i]*3;

//         FP_T4 V1 = vertices[primIndex];
//         FP_T4 V2 = vertices[primIndex + 1];
//         FP_T4 V3 = vertices[primIndex + 2];

//         FP_T t;
//         if (ray.intersects(V1, V2, V3, t)) {
//             candidates.values[tvalues.count] = candidates.values[i];
//             tvalues.values[tvalues.count++] = t;
//         }
//     }

//     if (tvalues.count < 2) {
//         return 0.0;
//     }

//     // printf ("Tvalues count: %d\n", tvalues.count);

//     // arg sort the tvalues
//     int index[MAX_COLLISIONS];
//     thrust::stable_sort_by_key(thrust::seq, tvalues.values, tvalues.values + tvalues.count, candidates.values);

//     // compute the thickness
//     FP_T val = matchOuterPairs (candidates, tvalues, ray, vertices, normals, tree);
//     // FP_T val = sumTvalues(tvalues);
//     return val;
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

extern "C" __global__ void projectParallelKernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, float2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals // ray casting
)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N.x == 0 || N.y == 0)
    {
        return;
    }

    if (gid >= N.x * N.y)
    {
        return;
    }

    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    FP_T4 scaled_U = U * ps.x;
    FP_T4 scaled_V = V * ps.y;

    for (int i = gid; i < N.x * N.y; i += blockDim.x * gridDim.x)
    {
        unsigned row = i / N.x;
        unsigned col = i % N.x;

        // U, V are the scaled basis vectors for the image plane
        FP_T4 pixel_coordinates = upperleft_origin - scaled_U * col - scaled_V * row;
        Ray ray = Ray(pixel_coordinates, W);
        image[row * N.x + col] += traceParallelRay(ray, tree, vertices, normals);
    }
}

extern "C" __global__ void projectPerspectiveKernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T4 ray_origin, float2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 *__restrict__ vertices,
    FP_T4 *__restrict__ normals // ray casting
)
{

    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (N.x == 0 || N.y == 0)
    {
        return;
    }

    if (gid_x >= N.x || gid_y >= N.y)
    {
        return;
    }

    //     Tree tree;
    //     tree.nb_keys = nb_keys;
    //     tree.rope = rope;
    //     tree.left = left;
    //     tree.indices = permutation;
    //     tree.bboxMin = bboxMin;
    //     tree.bboxMax = bboxMax;

    //     FP_T4 scaled_U = U * ps.x;
    //     FP_T4 scaled_V = V * ps.y;

    //     for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
    //         for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {
    //             // U, V are the scaled basis vectors for the image plane
    //             FP_T4 pixel_coordinates = upperleft_origin - scaled_U * i - scaled_V * j;
    //             FP_T4 direction = pixel_coordinates - ray_origin;
    //             Ray ray = Ray(ray_origin, direction);
    //             // ray.print();

    //             FP_T thickness = traceRay (ray, tree, vertices, normals);

    //             image[j * N.x + i] = thickness;
    //         }
    //     }
}