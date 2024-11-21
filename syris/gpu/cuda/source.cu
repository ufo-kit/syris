#include "Commons.cuh"
#include "Ray.cuh"

#define SENTINEL -1
#define INVALID -1

#define INT_MAX 2147483647
#define INT_MIN -2147483648

typedef unsigned int morton_t;
typedef int delta_t;

typedef struct {
    unsigned int nb_keys;
    morton_t *keys;
    unsigned int *indices;
    int *entered;
    int *rope;
    int *left;
    float4 *bboxMin;
    float4 *bboxMax;
} Tree;

__device__ bool isLeaf(Tree &tree, unsigned int index) {
    return index < tree.nb_keys;
}

__device__ unsigned int toInternalRepresentation(Tree &tree, unsigned int index) {
    return index + tree.nb_keys;
}

__device__ void growBox(float4 &bbMinInput, float4 &bbMaxInput, float4 *bbMinOutput, float4 *bbMaxOutput) {
    bbMinOutput->x = fminf(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = fminf(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = fminf(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = fmaxf(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = fmaxf(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = fmaxf(bbMaxInput.z, bbMaxOutput->z);
}

__device__ int delta(Tree &tree, int index) {

    if (index < 0 || index >= tree.nb_keys - 1) {
        return INT_MAX;
    }
    
    // TODO: augment the function if the codes are the same
    unsigned int a = tree.keys[index];
    unsigned int b = tree.keys[index + 1];
    int x = a ^ b;
    return x + (!x) * (INT_MIN + (index ^ (index + 1))) - 1; // 
}

__device__ void setRope(Tree &tree, unsigned int skip_index, int range_right, delta_t delta_right) {
    int rope;

    if (range_right != tree.nb_keys - 1) {
        int r = range_right + 1;
        rope = delta_right < delta(tree, r) ? r : toInternalRepresentation(tree, r);
    }
    else {
        rope = SENTINEL;
    }
    tree.rope[skip_index] = rope;
}

__device__ void setLeftChild (Tree &tree, unsigned int parent, unsigned int left_child) {
    tree.left[parent] = left_child;
}

__device__ void setBBMin (Tree &tree, unsigned int parent, float4 bbMin) {
    tree.bboxMin[parent] = bbMin;
}

__device__ void setBBMax (Tree &tree, unsigned int parent, float4 bbMax) {
    tree.bboxMax[parent] = bbMax;
}

__device__ float4 getBBMin (Tree &tree, unsigned int index) {
    return tree.bboxMin[index];
}

__device__ float4 getBBMax (Tree &tree, unsigned int index) {
    return tree.bboxMax[index];
}

__device__ int getRope (Tree &tree, unsigned int index) {
    return tree.rope[index];
}

__device__ int getLeftChild (Tree &tree, unsigned int index) {
    return tree.left[index];
}

__device__ void updateParents(Tree &tree, int i) {
    int range_left = i;
    int range_right = i;
    delta_t delta_left = delta(tree, i - 1);
    delta_t delta_right = delta(tree, i);

    float4 bbMinCurrent = getBBMin (tree, i);
    float4 bbMaxCurrent = getBBMax (tree, i);

    setRope(tree, i, range_right, delta_right);

    unsigned const root = toInternalRepresentation(tree, 0);

    do {
        int left_child;
        int right_child;
        if (delta_right < delta_left) {
            const int apetrei_parent = range_right;

            range_right = atomicCAS (&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_left);

            if (range_right == INVALID) {
                return;
            }
            delta_right = delta(tree, range_right);

            left_child = i;

            right_child = apetrei_parent + 1;

            // Memory sync
            __threadfence();

            if (right_child != range_right) {
                right_child = toInternalRepresentation(tree, right_child);
            }

            float4 bbMinRight = getBBMin (tree, right_child);
            float4 bbMaxRight = getBBMax (tree, right_child);
            growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
        }
        else {
            int const apetrei_parent = range_left - 1;
            range_left = atomicCAS (&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_right);

            if (range_left == INVALID){
                return;
            }

            delta_left = delta(tree, range_left - 1);

            left_child = apetrei_parent;
            bool const left_is_leaf = (left_child == range_left);

            // Memory sync
            __threadfence();
            
            if (!left_is_leaf) {
                left_child = toInternalRepresentation(tree, left_child);
            }

            float4 bbMinLeft = getBBMin (tree, left_child);
            float4 bbMaxLeft = getBBMax (tree, left_child);
            growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
        }

        int karras_parent = delta_right < delta_left ? range_right : range_left;
        karras_parent = toInternalRepresentation(tree, karras_parent);

        setLeftChild(tree, karras_parent, left_child);
        setBBMin(tree, karras_parent, bbMinCurrent);
        setBBMax(tree, karras_parent, bbMaxCurrent);
        setRope(tree, karras_parent, range_right, delta_right);

        i = karras_parent;
    }
    while (i != root);
    
    return;
}


__device__ void updateParentsVoxelgrid(Tree &tree, int i, int j, int k) {
    int range_left = i;
    int range_right = i;
    delta_t delta_left = delta(tree, i - 1);
    delta_t delta_right = delta(tree, i);

    float4 bbMinCurrent = getBBMin (tree, i);
    float4 bbMaxCurrent = getBBMax (tree, i);

    setRope(tree, i, range_right, delta_right);

    unsigned const root = toInternalRepresentation(tree, 0);

    do {
        int left_child;
        int right_child;
        if (delta_right < delta_left) {
            const int apetrei_parent = range_right;

            range_right = atomicCAS (&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_left);

            if (range_right == INVALID) {
                return;
            }
            delta_right = delta(tree, range_right);

            left_child = i;

            right_child = apetrei_parent + 1;

            // Memory sync
            __threadfence();

            if (right_child != range_right) {
                right_child = toInternalRepresentation(tree, right_child);
            }

            float4 bbMinRight = getBBMin (tree, right_child);
            float4 bbMaxRight = getBBMax (tree, right_child);
            growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
        }
        else {
            int const apetrei_parent = range_left - 1;
            range_left = atomicCAS (&(tree.entered[toInternalRepresentation(tree, apetrei_parent)]), INVALID, range_right);

            if (range_left == INVALID){
                return;
            }

            delta_left = delta(tree, range_left - 1);

            left_child = apetrei_parent;
            bool const left_is_leaf = (left_child == range_left);

            // Memory sync
            __threadfence();
            
            if (!left_is_leaf) {
                left_child = toInternalRepresentation(tree, left_child);
            }

            float4 bbMinLeft = getBBMin (tree, left_child);
            float4 bbMaxLeft = getBBMax (tree, left_child);
            growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
        }

        int karras_parent = delta_right < delta_left ? range_right : range_left;
        karras_parent = toInternalRepresentation(tree, karras_parent);

        setLeftChild(tree, karras_parent, left_child);
        setBBMin(tree, karras_parent, bbMinCurrent);
        setBBMax(tree, karras_parent, bbMaxCurrent);
        setRope(tree, karras_parent, range_right, delta_right);

        i = karras_parent;
    }
    while (i != root);
    
    return;
}


__device__ void query (Tree &tree, Ray &ray, CandidateList &candidates) {
    int current_node = toInternalRepresentation(tree, 0);
    
    do {
        float4 bbMax = getBBMax (tree, current_node);
        float4 bbMin = getBBMin (tree, current_node);
        if (ray.intersects(bbMin, bbMax)) {
            if (isLeaf(tree, current_node)) {
                if (candidates.count == MAX_COLLISIONS) {
                    return;
                }
                candidates.collisions[candidates.count++] = current_node;
                current_node = getRope(tree, current_node);
            }
            else {
                current_node = getLeftChild(tree, current_node);
            }
        }
        else {
            current_node = getRope(tree, current_node);
        }
    }
    while (current_node != SENTINEL);
}

__device__ float4 phi (int i, int j, float2 D, uint2 N) {
    float delta_x = D.x / (N.x-1);
    float delta_y = D.y / (N.y-1);
    float Dx2 = D.x / 2;
    float Dy2 = D.y / 2;

    float x = -Dx2 + i * delta_x;
    float y = -Dy2 + j * delta_y;
    return make_float4(x, y, 0.0, 0);
}

__device__ float sumTvalues (CollisionList &t_values) {
    float thickness = 0;
    for (int i = 0; i < t_values.count; i++) {
        thickness += t_values.collisions[i];
    }
    return thickness;
}
//
__device__ float computeThickness(CollisionList &tvalues) {
    float result = 0.0;
    float epsilon = 1e-6;
    int i = 0, j = 1;
    if (tvalues.count == 0) {
        return 0.0;
    }
    
    float t1 = tvalues.collisions[i];
    while (j < tvalues.count) {
        float t2 = tvalues.collisions[j];
        float d = fabsf (t2 - t1);
        if (d > epsilon){
            result += d;
            t1 = t2;
            j++;
        }
        j++;
    }

    return result;
}

inline __device__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// printf ("ID -> %d \nT -> \n%f %f %f \n%f %f %f \n%f %f %f\n", primIndex,
//             vertices[primIndex*3].x, vertices[primIndex*3].y, vertices[primIndex*3].z,
//             vertices[primIndex*3+1].x, vertices[primIndex*3+1].y, vertices[primIndex*3+1].z,
//             vertices[primIndex*3+2].x, vertices[primIndex*3+2].y, vertices[primIndex*3+2].z);

__device__ float matchOuterPairs (
        CandidateList &candidates, CollisionList &tvalues, Ray &ray,
        float4* __restrict__ vertices,
        float4* __restrict__ normals,
        Tree &tree) {
    float thickness = 0.0, inner = 0.0;
    int counter = 0;

    if (tvalues.count == 1) {
        return tvalues.collisions[0];
    }

    for (int i = 0; i < tvalues.count; i++) {
        unsigned primIndex = candidates.collisions[i];
        bool is_neg = dot(ray.getDirection(), normals[primIndex]) < 0;
        if (!is_neg && counter == 0) {
            continue;
        }

        if (is_neg) {
            if (counter++ == 0)
                inner = tvalues.collisions[i];
        }
        else {
            if (--counter == 0) {
                thickness += tvalues.collisions[i] - inner;
            }
        }

    }
    return thickness;
}

__device__ float getNbCandidate (Ray &ray, Tree &tree, float4 *vertices, float4 *normals) {
    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));

    // This is where the acceleration structure (BVH) is actually usefull
    query(tree, ray, candidates);
    
    return candidates.count;
}

__device__ float getSumedTvalues (Ray &ray, Tree &tree, float4 *vertices) {
    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));

    CollisionList tvalues;
    tvalues.count = 0;
    memset(tvalues.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    query(tree, ray, candidates);

    if (candidates.count == 0) {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;
        
        float4 V1 = vertices[primIndex];
        float4 V2 = vertices[primIndex + 1];
        float4 V3 = vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            candidates.collisions[tvalues.count] = candidates.collisions[i];
            tvalues.collisions[tvalues.count++] = t;
        }
    }

    return sumTvalues(tvalues);
}


__device__ float traceParallelRay (
        Ray &ray, Tree &tree, 
        float4* __restrict__ vertices,
        float4* __restrict__ normals) {
    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));

    CollisionList tvalues;
    tvalues.count = 0;
    memset(tvalues.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    query(tree, ray, candidates);

    if (candidates.count == 0) {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;
        
        float4 V1 = vertices[primIndex];
        float4 V2 = vertices[primIndex + 1];
        float4 V3 = vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            candidates.collisions[tvalues.count] = candidates.collisions[i];
            tvalues.collisions[tvalues.count++] = t;
        }
    }

    if (tvalues.count < 2) {
        return 0.0;
    }

    // printf ("Tvalues count: %d\n", tvalues.count);

    // arg sort the tvalues
    int index[MAX_COLLISIONS];
    thrust::stable_sort_by_key(thrust::seq, tvalues.collisions, tvalues.collisions + tvalues.count, candidates.collisions);

    // compute the thickness
    float val = matchOuterPairs (candidates, tvalues, ray, vertices, normals, tree);
    // float val = sumTvalues(tvalues);
    return val;
}

__device__ float traceRay (
        Ray &ray, Tree &tree, 
        float4* __restrict__ vertices,
        float4* __restrict__ normals) {
    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));

    CollisionList tvalues;
    tvalues.count = 0;
    memset(tvalues.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    query(tree, ray, candidates);

    if (candidates.count == 0) {
        return 0.0;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;
        
        float4 V1 = vertices[primIndex];
        float4 V2 = vertices[primIndex + 1];
        float4 V3 = vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            candidates.collisions[tvalues.count] = candidates.collisions[i];
            tvalues.collisions[tvalues.count++] = t;
        }
    }

    if (tvalues.count < 2) {
        return 0.0;
    }

    // printf ("Tvalues count: %d\n", tvalues.count);

    // arg sort the tvalues
    int index[MAX_COLLISIONS];
    thrust::stable_sort_by_key(thrust::seq, tvalues.collisions, tvalues.collisions + tvalues.count, candidates.collisions);

    // compute the thickness
    float val = matchOuterPairs (candidates, tvalues, ray, vertices, normals, tree);
    // float val = sumTvalues(tvalues);
    return val;
}

extern "C" __global__ void calculateBbBoxKernel (float4 *vertices, float4 *bbMin, float4 *bbMax, unsigned int nb_keys) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nb_keys) {
        float4 V1 = vertices[tid * 3];
        float4 V2 = vertices[tid * 3 + 1];
        float4 V3 = vertices[tid * 3 + 2];
        calculateTriangleBoundingBox (V1, V2, V3, bbMin[tid], bbMax[tid]);

        tid += blockDim.x * gridDim.x;
    }
    
}


extern "C" __global__ void projectTriangleCentroid(
    unsigned int const nb_keys, float4 const *vertices, unsigned int *keys,
    float4 *bbMin, float4 *bbMax, float4 const scene_bbMin, float4 const scene_bbMax) {

    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;


    while (index < nb_keys) {
        // Get the triangle vertices
        float4 V1 = vertices[index * 3];
        float4 V2 = vertices[index * 3 + 1];
        float4 V3 = vertices[index * 3 + 2];

        // Calculate the bounding box of the triangle
        calculateTriangleBoundingBox(V1, V2, V3, bbMin[index], bbMax[index]);

        // Calculate the centroid of the AABB
        float4 centroid = getBoundingBoxCentroid(bbMin[index], bbMax[index]);
        
        float4 normalizedCentroid = normalize(centroid, scene_bbMin, scene_bbMax);

        // Calculate the morton code of the triangle
        morton_t mortonCode = calculateMortonCode(normalizedCentroid);

        // Store the morton code
        keys[index] = mortonCode;

        index += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void growTreeKernel (
    unsigned int nb_keys, unsigned int *keys, unsigned int *permutation, 
    int *rope, int *left, int *entered,
    float4 *bboxMin, float4 *bboxMax) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys) {
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

// extern "C" __global__ void growVoxelgridTreeKernel (
//     uint4 N, float4 L // voxelgrid parameters
//     unsigned int *keys, unsigned int *permutation, // tree parameters
//     int *rope, int *left, int *entered,
//     ) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;

//     while (index < nb_keys) {
//         Tree tree;
//         tree.nb_keys = N.x * N.y * N.z;
//         tree.keys = keys;
//         tree.indices = permutation;
//         tree.entered = entered;
//         tree.rope = rope;
//         tree.left = left;

//         updateParents(tree, index);
//         index += blockDim.x * gridDim.x;
//     }
// }

// extern "C" __global__ void projectNbCandidatesKernel (
//     unsigned int nb_keys,
//     float *image, uint2 N, float2 D, // image parameters
//     float4 U, float4 V, float4 W, float4 origin, // projection basis and origin
//     int *rope, int *left, unsigned *permutation,  // BVH tree
//     float4 *bboxMin, float4 *bboxMax, float4 *vertices, float4 *normals // ray casting
//     ) {
   
//     int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (N.x == 0 || N.y == 0) {
//         return;
//     }

//     if (gid_x >= N.x || gid_y >= N.y) {
//         return;
//     }

//     Tree tree;
//     tree.nb_keys = nb_keys;
//     tree.rope = rope;
//     tree.left = left;
//     tree.indices = permutation;
//     tree.bboxMin = bboxMin;
//     tree.bboxMax = bboxMax;

//     for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
//         for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {

//             float4 point_local_basis = phi(i, j, D, N);
//             float4 point_new_basis = origin + U * point_local_basis.x + V * point_local_basis.y;
//             Ray ray = Ray(point_new_basis, W);

//             float thickness = getNbCandidate (ray, tree, vertices, normals);

//             image[j * N.x + i] = thickness;
//         }
//     }
// }

// extern "C" __global__ void projectTvaluesKernel (
//     unsigned int nb_keys,
//     float *image, uint2 N, float2 D, // image parameters
//     float4 U, float4 V, float4 W, float4 origin, // projection basis and origin
//     int *rope, int *left, unsigned *permutation,  // BVH tree
//     float4 *bboxMin, float4 *bboxMax, float4 *vertices // ray casting
//     ) {
   
//     int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (N.x == 0 || N.y == 0) {
//         return;
//     }

//     if (gid_x >= N.x || gid_y >= N.y) {
//         return;
//     }

//     Tree tree;
//     tree.nb_keys = nb_keys;
//     tree.rope = rope;
//     tree.left = left;
//     tree.indices = permutation;
//     tree.bboxMin = bboxMin;
//     tree.bboxMax = bboxMax;

//     for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
//         for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {

//             float4 point_local_basis = phi(i, j, D, N);
//             float4 point_new_basis = origin + U * point_local_basis.x + V * point_local_basis.y;
//             Ray ray = Ray(point_new_basis, W);

//             float thickness = getSumedTvalues (ray, tree, vertices);

//             image[j * N.x + i] = thickness;
//         }
//     }
// }

extern "C" __global__ void projectParallelKernel (
    unsigned nb_keys, float* image, uint2 N,
    float4 U, float4 V, float4 W,  // projection basis and origin
    float4 upperleft_origin, float2 ps,
    int* rope,
    int* left,
    unsigned* permutation,  // BVH tree
    float4* bboxMin,
    float4* bboxMax,
    float4* __restrict__ vertices,
    float4* __restrict__ normals // ray casting
    ) {
   
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (N.x == 0 || N.y == 0) {
        return;
    }

    if (gid_x >= N.x || gid_y >= N.y) {
        return;
    }

    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    float4 scaled_U = U * ps.x;
    float4 scaled_V = V * ps.y;

    for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
        for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {
            // U, V are the scaled basis vectors for the image plane
            float4 pixel_coordinates = upperleft_origin - scaled_U * i - scaled_V * j;
            Ray ray = Ray(pixel_coordinates, W);
            image[j * N.x + i] = traceParallelRay (ray, tree, vertices, normals);
        }
    }
}

// extern "C" __global__ void projectGivenRaysKernel (
//     unsigned int nb_keys, unsigned int nb_rays,
//     float *ray_retvals, // projected thicknesses
//     float4 *origins, // ray origins
//     int *rope, int *left, unsigned *permutation,  // BVH tree
//     float4 *bboxMin, float4 *bboxMax, float4 *vertices, float4 *normals // ray casting
//     ) {
   
//     int gid_x = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid_x >= nb_rays) {
//         return;
//     }

//     Tree tree;
//     tree.nb_keys = nb_keys;
//     tree.rope = rope;
//     tree.left = left;
//     tree.indices = permutation;
//     tree.bboxMin = bboxMin;
//     tree.bboxMax = bboxMax;

//     float4 W = make_float4(0.0, 0.0, -1.0, 0.0);

//     for (int i = gid_x; i < nb_rays; i += blockDim.x * gridDim.x) {        
//         Ray ray = Ray(origins[i], W);
//         ray_retvals[i] = traceParallelRay (ray, tree, vertices, normals);
//     }
// }

extern "C" __global__ void projectPerspectiveKernel (
    unsigned nb_keys, float* image, uint2 N,
    float4 U, float4 V, float4 W,  // projection basis and origin
    float4 upperleft_origin, float4 ray_origin, float2 ps, 
    int* rope,
    int* left,
    unsigned* permutation,  // BVH tree
    float4* bboxMin,
    float4* bboxMax,
    float4* __restrict__ vertices,
    float4* __restrict__ normals // ray casting
    ) {
   
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (N.x == 0 || N.y == 0) {
        return;
    }

    if (gid_x >= N.x || gid_y >= N.y) {
        return;
    }

    Tree tree;
    tree.nb_keys = nb_keys;
    tree.rope = rope;
    tree.left = left;
    tree.indices = permutation;
    tree.bboxMin = bboxMin;
    tree.bboxMax = bboxMax;

    float4 scaled_U = U * ps.x;
    float4 scaled_V = V * ps.y;
    

    for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
        for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {
            // U, V are the scaled basis vectors for the image plane
            float4 pixel_coordinates = upperleft_origin - scaled_U * i - scaled_V * j;
            float4 direction = pixel_coordinates - ray_origin;
            Ray ray = Ray(ray_origin, direction);
            // ray.print();

            float thickness = traceRay (ray, tree, vertices, normals);

            image[j * N.x + i] = thickness;
        }
    }
}