#pragma once
// #include <cstdio>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <cmath>

#define MAX_COLLISIONS 256
#define EPSILON 0.00001

typedef struct {
    int collisions[MAX_COLLISIONS];
    int count;
} CandidateList;

typedef struct {
    float collisions[MAX_COLLISIONS];
    int count;
} CollisionList;

// // #include <vector_types.h>

// Uncomment this to enforce warp synchronization
#define SAFE_WARP_SYNCHRONY

// Synchronize warp. This protects the code from future compiler optimization that 
// involves instructions reordering, possibly leading to race conditions. 
// __syncthreads() could be used instead, at a slight performance penalty
#ifdef SAFE_WARP_SYNCHRONY
#define WARP_SYNC \
do { \
    int _sync = 0; \
    __shfl(_sync, 0); \
} while (0)
#else
#define WARP_SYNC \
do { \
} while (0)
#endif

#define WARP_SIZE 32

// Get the global warp index
#define GLOBAL_WARP_INDEX static_cast<int>((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE)

// Get the block warp index
#define WARP_INDEX static_cast<int>(threadIdx.x / WARP_SIZE)

// Get a pointer to the beginning of a warp area in an array that stores a certain number of 
// elements for each warp
#define WARP_ARRAY(source, elementsPerWarp) ((source) + WARP_INDEX * (elementsPerWarp))

// Calculate the index of a value in an array that stores a certain number of elements for each 
// warp
#define WARP_ARRAY_INDEX(index, elementsPerWarp) (WARP_INDEX * (elementsPerWarp) + (index))

// Index of the thread in the warp, from 0 to WARP_SIZE-1
#define THREAD_WARP_INDEX (threadIdx.x & (WARP_SIZE - 1))

// Read a vector of 3 elements using shuffle operations
#define SHFL_float4(destination, source, index) \
do { \
    (destination).x = __shfl((source)[0], (index)); \
    (destination).y = __shfl((source)[1], (index)); \
    (destination).z = __shfl((source)[2], (index)); \
} while (0);


__device__ float4 operator+(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

__device__ float4 operator-(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

__device__ float4 operator*(const float4& lhs, const float& rhs) {
    return make_float4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}

__device__ float4 operator/(const float4& lhs, const float& rhs) {
    return make_float4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

__device__ float4 operator*(const float& lhs, const float4& rhs) {
    return make_float4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

__device__ float4 operator/(const float& lhs, const float4& rhs) {
    return make_float4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

__device__ float4 operator*(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}


__device__ inline float xor_signmask(float x, int y)
{
    return (float)(int(x) ^ y);
}

__device__ inline float4 sub4(float4 a, float4 b)
{
    float4 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    c.w = 0;
    return c;
}

__device__ inline float4 abs4(float4 a)
{
    float4 c;
    c.x = fabs(a.x);
    c.y = fabs(a.y);
    c.z = fabs(a.z);
    c.w = fabs(a.w);
    return c;
}

__device__ inline float4 min4(float4 a, float4 b)
{
    float4 c;
    c.x = fminf(a.x, b.x);
    c.y = fminf(a.y, b.y);
    c.z = fminf(a.z, b.z);
    c.w = fminf(a.w, b.w);
    return c;
}

__device__ inline float4 max4(float4 a, float4 b)
{
    float4 c;
    c.x = fmaxf(a.x, b.x);
    c.y = fmaxf(a.y, b.y);
    c.z = fmaxf(a.z, b.z);
    c.w = fmaxf(a.w, b.w);
    return c;
}

__device__ inline float4 cross4 (float4 a, float4 b) // cross product between two 3D vectors
{ 
    float4 c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}

__device__ inline float dot4(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float4 normalize4(float4 a)
{
    float invLen = rsqrtf(dot4(a, a));
    float norm = 1.0f / invLen;
    return make_float4(a.x * invLen, a.y * invLen, a.z * invLen, norm);
}


__device__ inline int maxDimIndex(const float4 &D)
{
    if (D.x > D.y)
    {
        if (D.x > D.z)
        {
            return 0;
        }
        else
        {
            return 2;
        }
    }
    else
    {
        if (D.y > D.z)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
}



__device__ inline float4 permuteVectorAlongMaxDim(float4 v, unsigned int shift)
{
    float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    switch (shift)
    {
    case 0:
        c.x = v.y;
        c.y = v.z;
        c.z = v.x;
        c.w = 0;
        break;
    case 1:
        c.x = v.z;
        c.y = v.x;
        c.z = v.y;
        c.w = 0;
        break;
    case 2:
        c.x = v.x;
        c.y = v.y;
        c.z = v.z;
        c.w = 0;
        break;
    }

    // printf("Permuted: %f %f %f\n", c.x, c.y, c.z);

    return c;
}


// namespace BVHRT
// {

/// <summary> Checks if the specified index corresponds to an internal node. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">             Node index. </param>
/// <param name="numberOfTriangles"> Number of triangles contained in the BVH. </param>
///
/// <returns> true if the index corresponds to an internal node, false otherwise. </returns>
__forceinline__  __device__ bool isInternalNode(unsigned int index,
        unsigned int numberOfTriangles)
{
    return (index < numberOfTriangles - 1);
}

/// <summary> Checks if the specified index corresponds to a leaf node. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">             Node index. </param>
/// <param name="numberOfTriangles"> Number of triangles contained in the BVH. </param>
///
/// <returns> true if the index corresponds to a leaf node, false otherwise. </returns>
__forceinline__  __device__ bool isLeaf(unsigned int index, unsigned int numberOfTriangles)
{
    return !isInternalNode(index, numberOfTriangles);
}

/// <summary> Calculates the surface area of a bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin"> The bounding box minimum. </param>
/// <param name="bbMax"> The bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__  __device__ float calculateBoundingBoxSurfaceArea(float4 bbMin,
        float4 bbMax)
{
    float4 size;
    size.x = bbMax.x - bbMin.x;
    size.y = bbMax.y - bbMin.y;
    size.z = bbMax.z - bbMin.z;
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the bounding box surface area. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin"> The bounding box minimum. </param>
/// <param name="bbMax"> The bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__  __device__ float calculateBoundingBoxSurfaceArea(const float* bbMin,
        const float* bbMax)
{
    float4 size;
    size.x = bbMax[0] - bbMin[0];
    size.y = bbMax[1] - bbMin[1];
    size.z = bbMax[2] - bbMin[2];
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the union of two bounding boxes and returns the union box surface area. 
///           </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin1"> First bounding box minimum. </param>
/// <param name="bbMax1"> First bounding box maximum. </param>
/// <param name="bbMin2"> Second bounding box minimum. </param>
/// <param name="bbMax2"> Second bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__  __device__ float calculateBoundingBoxAndSurfaceArea(const float4 bbMin1,
        const float4 bbMax1, const float4 bbMin2, const float4 bbMax2)
{
    float4 size;
    size.x = max(bbMax1.x, bbMax2.x) - min(bbMin1.x, bbMin2.x);
    size.y = max(bbMax1.y, bbMax2.y) - min(bbMin1.y, bbMin2.y);
    size.z = max(bbMax1.z, bbMax2.z) - min(bbMin1.z, bbMin2.z);
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Loads a triangle from the vertices array. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">    The node index. </param>
/// <param name="vertices"> The array of vertices. </param>
/// <param name="vertex1">  [out] The first vertex. </param>
/// <param name="vertex2">  [out] The second vertex. </param>
/// <param name="vertex3">  [out] The third vertex. </param>
__forceinline__  __device__ void loadTriangle(int index, const float4* vertices, 
        float4* vertex1, float4* vertex2, float4* vertex3)
{
    *vertex1 = vertices[index * 3];
    *vertex2 = vertices[index * 3 + 1];
    *vertex3 = vertices[index * 3 + 2];
}

/// <summary> Calculates the triangle bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="vertex1">        The first vertex. </param>
/// <param name="vertex2">        The second vertex. </param>
/// <param name="vertex3">        The third vertex. </param>
/// <param name="boundingBoxMin"> [out] The bounding box minimum. </param>
/// <param name="boundingBoxMax"> [out] The bounding box maximum. </param>
__forceinline__  __device__ void calculateTriangleBoundingBox(
    float4 const &vertex1, float4 const &vertex2, float4 const &vertex3, float4 &boundingBoxMin, float4 &boundingBoxMax)
{
    boundingBoxMin.x = min(vertex1.x, vertex2.x);
    boundingBoxMin.x = min(boundingBoxMin.x, vertex3.x);
    boundingBoxMax.x = max(vertex1.x, vertex2.x);
    boundingBoxMax.x = max(boundingBoxMax.x, vertex3.x);

    boundingBoxMin.y = min(vertex1.y, vertex2.y);
    boundingBoxMin.y = min(boundingBoxMin.y, vertex3.y);
    boundingBoxMax.y = max(vertex1.y, vertex2.y);
    boundingBoxMax.y = max(boundingBoxMax.y, vertex3.y);

    boundingBoxMin.z = min(vertex1.z, vertex2.z);
    boundingBoxMin.z = min(boundingBoxMin.z, vertex3.z);
    boundingBoxMax.z = max(vertex1.z, vertex2.z);
    boundingBoxMax.z = max(boundingBoxMax.z, vertex3.z);
}


__device__ inline float4 getBoundingBoxCentroid(float4 bboxMin, float4 bboxMax)
{
    float4 centroid;

    centroid.x = (bboxMin.x + bboxMax.x) / 2.0f;
    centroid.y = (bboxMin.y + bboxMax.y) / 2.0f;
    centroid.z = (bboxMin.z + bboxMax.z) / 2.0f;

    return centroid;
}

// --- Vector operations --------------------------------------------------------------------------

/// <summary> Gets the coordinate from the specified vector type using its index. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> Source vector. </param>
/// <param name="index">  Coordinate index. </param>
///
/// <returns> The coordinate value. </returns>
__forceinline__  __device__ float getCoordinate(float4 source, int index)
{
    if (index == 0)
    {
        return source.x;
    }
    else if (index == 1)
    {
        return source.y;
    }
    else
    {
        return source.z;
    }
}

/// <summary> Sets the coordinate from the specified vector type using its index. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> [out] Source vector. </param>
/// <param name="index">  Coordinate index. </param>
/// <param name="value">  Value. </param>
__forceinline__  __device__ void setCoordinate(float4* source, int index, float value)
{
    if (index == 0)
    {
        source->x = value;
    }
    else if (index == 1)
    {
        source->y = value;
    }
    else
    {
        source->z = value;
    }
}

/// <summary> Calculates v1 - v2. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="v1"> v1. </param>
/// <param name="v2"> v2. </param>
///
/// <returns> v1 - v2. </returns>
__forceinline__  __device__ float4 subtract(float4 v1, float4 v2)
{
    float4 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return result;
}

/// <summary> Calculates the cross product between v1 and v2. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="v1"> v1. </param>
/// <param name="v2"> v2. </param>
///
/// <returns> v1 x v2. </returns>
__forceinline__  __device__ float4 cross(float4 v1, float4 v2)
{
    float4 result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}

/// <summary> Converts a float4 to float4. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> The source vector. </param>
///
/// <returns> A float4. </returns>
__forceinline__  __device__ float4 float4Fromfloat4(float4 source)
{
    float4 temp;
    temp.x = source.x;
    temp.y = source.y;
    temp.z = source.z;
    // we do not care about w

    return temp;
}

/// <summary> Converts a float4 to an array of floats. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 01/21/2015. </remarks>
///
/// <param name="source"> The source vector. </param>
/// <param name="destination"> The destination array. </param>
__forceinline__  __device__ void floatArrayFromFloat4(float4 source, float* destination)
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
}

/// <summary> Converts an array of floats to a float4. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 01/21/2015. </remarks>
///
/// <param name="source"> The source array. </param>
/// <param name="destination"> The destination vector. </param>
__forceinline__  __device__ void float4FromFromFloatArray(const float* source, 
        float4& destination)
{
    destination.x = source[0];
    destination.y = source[1];
    destination.z = source[2];
}

// --- Morton codes -------------------------------------------------------------------------------

/// <summary> Normalizes a position using the specified bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point">          The point. </param>
/// <param name="boundingBoxMin"> The bounding box minimum. </param>
/// <param name="boundingBoxMax"> The bounding box maximum. </param>
///
/// <returns> Normalized position. </returns>
__forceinline__  __device__ float4 normalize(float4 point, float4 boundingBoxMin,
        float4 boundingBoxMax)
{
    float4 normalized;
    normalized.x = (point.x - boundingBoxMin.x) / (boundingBoxMax.x - boundingBoxMin.x);
    normalized.y = (point.y - boundingBoxMin.y) / (boundingBoxMax.y - boundingBoxMin.y);
    normalized.z = (point.z - boundingBoxMin.z) / (boundingBoxMax.z - boundingBoxMin.z);
    return normalized;
}

/// <summary> Un-normalizes a position using the specified bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="normalized"> The normalized value. </param>
/// <param name="bboxMin">    The bounding box minimum. </param>
/// <param name="bboxMax">    The bounding box maximum. </param>
///
/// <returns> The un-normalized value. </returns>
__forceinline__  __device__ float4 denormalize(float4 normalized, float4 bboxMin,
        float4 bboxMax)
{
    float4 point;
    point.x = bboxMin.x + (bboxMax.x - bboxMin.x) * normalized.x;
    point.y = bboxMin.y + (bboxMax.y - bboxMin.y) * normalized.y;
    point.z = bboxMin.z + (bboxMax.z - bboxMin.z) * normalized.z;
    return point;
}

/// <summary> Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The expanded value. </returns>
__forceinline__  __device__ unsigned int expandBits(unsigned int value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

/// <summary> Calculates the point morton code using 30 bits. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The calculated morton code. </returns>
// __forceinline__  __device__ unsigned int calculateMortonCode(float4 point)
// {
//     // Discretize the unit cube into a 10 bit integer
//     uint3 discretized;
//     discretized.x = (unsigned int)min(max(point.x * 1024.0f, 0.0f), 1023.0f);
//     discretized.y = (unsigned int)min(max(point.y * 1024.0f, 0.0f), 1023.0f);
//     discretized.z = (unsigned int)min(max(point.z * 1024.0f, 0.0f), 1023.0f);

//     discretized.x = expandBits(discretized.x);
//     discretized.y = expandBits(discretized.y);
//     discretized.z = expandBits(discretized.z);

//     return discretized.x * 4 + discretized.y * 2 + discretized.z;
// }

template <int N>__forceinline__  __device__ unsigned int expandBitsBy (unsigned int)
{
    static_assert(0 <= N && N < 10,
                "expandBitsBy can only be used with values 0-9");

    return 0; 
}

template <>__forceinline__  __device__ unsigned int expandBitsBy<0> (unsigned int x)
{
    return x;
}

template <>__forceinline__  __device__ unsigned int expandBitsBy<1> (unsigned int x)
{
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

template <>__forceinline__  __device__ unsigned int expandBitsBy<2> (unsigned int x)
{
    x &= 0x000003ffu;
    x = (x ^ (x << 16)) & 0xff0000ffu;
    x = (x ^ (x << 8)) & 0x0300f00fu;
    x = (x ^ (x << 4)) & 0x030c30c3u;
    x = (x ^ (x << 2)) & 0x09249249u;
    return x;
}

template <>__forceinline__  __device__ unsigned int expandBitsBy<3> (unsigned int x)
{
    x &= 0xffu;
    x = (x | x << 16) & 0xc0003fu;
    x = (x | x << 8) & 0xc03807u;
    x = (x | x << 4) & 0x8430843u;
    x = (x | x << 2) & 0x9090909u;
    x = (x | x << 1) & 0x11111111u;
    return x;
}
__forceinline__  __device__ unsigned int calculateMortonCode(float4 point)
{
    // Discretize the unit cube into a 10 bit integer
    constexpr unsigned N = 1u << 10;

    float p[3] = {point.x, point.y, point.z};

    unsigned r = 0;
    for (int d = 0; d < 3; ++d)
    {
        auto x = min (max (p[d] * N, 0.0f), (float)(N - 1));
        r += (expandBitsBy<2>((unsigned int)x) << (3 - d - 1));
    }
    return r;
}

/// <summary> Compact bits from the specified 30-bit value, using only one bit at every 3 from the
///           original value and forming a 10-bit value. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The compacted value. </returns>
__forceinline__  __device__ unsigned int compactBits(unsigned int value)
{
    unsigned int compacted = value;
    compacted &= 0x09249249;
    compacted = (compacted ^ (compacted >> 2)) & 0x030c30c3;
    compacted = (compacted ^ (compacted >> 4)) & 0x0300f00f;
    compacted = (compacted ^ (compacted >> 8)) & 0xff0000ff;
    compacted = (compacted ^ (compacted >> 16)) & 0x000003ff;
    return compacted;
}

/// <summary> Decodes the 'x' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCodeX(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 2);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'y' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCodeY(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 1);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'z' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCodeZ(unsigned int value)
{
    unsigned int expanded = compactBits(value);

    return expanded / 1024.0f;
}

/// <summary> Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The expanded value. </returns>
__forceinline__  __device__ unsigned long long int expandBits64(
        unsigned long long int value)
{
    unsigned long long int expanded = value;
    expanded &= 0x1fffff;
    expanded = (expanded | expanded << 32) & 0x1f00000000ffff;
    expanded = (expanded | expanded << 16) & 0x1f0000ff0000ff;
    expanded = (expanded | expanded << 8) & 0x100f00f00f00f00f;
    expanded = (expanded | expanded << 4) & 0x10c30c30c30c30c3;
    expanded = (expanded | expanded << 2) & 0x1249249249249249;

    return expanded;
}

/// <summary> Calculates the point morton code using 63 bits. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The 63-bit morton code. </returns>
__forceinline__  __device__ unsigned long long int calculateMortonCode64(float4 point)
{
    // Discretize the unit cube into a 10 bit integer
    unsigned long long int discretized[3];
    discretized[0] = (unsigned long long int)min(max(point.x * 2097152.0f, 0.0f), 2097151.0f);
    discretized[1] = (unsigned long long int)min(max(point.y * 2097152.0f, 0.0f), 2097151.0f);
    discretized[2] = (unsigned long long int)min(max(point.z * 2097152.0f, 0.0f), 2097151.0f);

    discretized[0] = expandBits64(discretized[0]);
    discretized[1] = expandBits64(discretized[1]);
    discretized[2] = expandBits64(discretized[2]);

    return discretized[0] * 4 + discretized[1] * 2 + discretized[2];
}

/// <summary> Compact bits from the specified 63-bit value, using only one bit at every 3 from the
///           original value and forming a 21-bit value. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The compacted value. </returns>
__forceinline__  __device__ unsigned long long int compactBits64(
        unsigned long long int value)
{
    unsigned long long int compacted = value;

    compacted &= 0x1249249249249249;
    compacted = (compacted | compacted >> 2) & 0x10c30c30c30c30c3;
    compacted = (compacted | compacted >> 4) & 0x100f00f00f00f00f;
    compacted = (compacted | compacted >> 8) & 0x1f0000ff0000ff;
    compacted = (compacted | compacted >> 16) & 0x1f00000000ffff;
    compacted = (compacted | compacted >> 32) & 0x1fffff;

    return compacted;
}

/// <summary> Decodes the 'x' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCode64X(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 2);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'y' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCode64Y(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 1);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'z' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ float decodeMortonCode64Z(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value);

    return expanded / 2097152.0f;
}

/// <summary> Expands the group bounding box using the specified new bounding box coordinates. 
///           </summary>
///
/// <remarks> Leonardo, 01/22/2015. </remarks>
///
/// <param name="groupBbMin"> Group bounding box minimum values. </param>
/// <param name="groupBbMax"> Group bounding box maximum values. </param>
/// <param name="newBbMin"> New bounding box minimum values. </param>
/// <param name="newBbMax"> New bounding box maximum values. </param>
__forceinline__  __device__ void expandBoundingBox(float4& groupBbMin, float4& groupBbMax, 
        const float4& newBbMin, const float4& newBbMax)
{
    groupBbMin.x = min(newBbMin.x, groupBbMin.x);
    groupBbMin.y = min(newBbMin.y, groupBbMin.y);
    groupBbMin.z = min(newBbMin.z, groupBbMin.z);

    groupBbMax.x = max(newBbMax.x, groupBbMax.x);
    groupBbMax.y = max(newBbMax.y, groupBbMax.y);
    groupBbMax.z = max(newBbMax.z, groupBbMax.z);
}

// --- Arithmetic sequence operations -------------------------------------------------------------

/// <summary> Calculates the sum of an arithmetic sequence. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements in the sequence. </param>
/// <param name="firstElement"> First element in the sequence. </param>
/// <param name="lastElement"> Last element in the sequence. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__  __device__ int sumArithmeticSequence(int numberOfElements, 
        int firstElement, int lastElement)
{
    return numberOfElements * (firstElement + lastElement) / 2;
}

// }


// Device implementations
__device__ inline float device_int_as_float(int i)
{
    return __int_as_float(i);
}

__device__ inline int device_float_as_int(float f)
{
    return __float_as_int(f);
}

__device__ inline float device_xorf(float x, int y)
{
    return __int_as_float(__float_as_int(x) ^ y);
}

__device__ inline int device_sign_mask(float x)
{
    return __float_as_int(x) & 0x80000000;
}

// Host implementations
inline float host_int_as_float(int i)
{
    return *(float*)(&i);
}

inline int host_float_as_int(float f)
{
    return *(int*)(&f);
}

inline float host_xorf(float x, int y)
{
    return host_int_as_float(host_float_as_int(x) ^ y);
}

inline int host_sign_mask(float x)
{
    return host_float_as_int(x) & 0x80000000;
}

// Unified interface for both host and device
__device__ inline float __iaf(int i)
{
#ifdef __CUDA_ARCH__
    return device_int_as_float(i);
#else
    return host_int_as_float(i);
#endif
}

__device__ inline int __fai(float f)
{
#ifdef __CUDA_ARCH__
    return device_float_as_int(f);
#else
    return host_float_as_int(f);
#endif
}

__device__ inline float xorf(float x, int y)
{
#ifdef __CUDA_ARCH__
    return device_xorf(x, y);
#else
    return host_xorf(x, y);
#endif
}

__device__ inline int sign_mask(float x)
{
#ifdef __CUDA_ARCH__
    return device_sign_mask(x);
#else
    return host_sign_mask(x);
#endif
}