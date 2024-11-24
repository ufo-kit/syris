#pragma once
#include <cmath>

#define MAX_COLLISIONS 256
#define EPSILON 0.00001
#define INF INFINITY

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

// Swap two integers
__forceinline__ __device__ void swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
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

__device__ inline float max_component(float4 a)
{
    return fmaxf(fmaxf(a.x, a.y), a.z);
}

__device__ inline float min_component(float4 a)
{
    return fminf(fminf(a.x, a.y), a.z);
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

__forceinline__  __device__ float4 subtract(float4 v1, float4 v2)
{
    float4 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return result;
}

__forceinline__  __device__ float4 cross(float4 v1, float4 v2)
{
    float4 result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}


// --- Morton codes -------------------------------------------------------------------------------

/// <summary> Normalizes a position using the specified bounding box. </summary>
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
__forceinline__  __device__ float decodeMortonCodeX(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 2);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'y' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
__forceinline__  __device__ float decodeMortonCodeY(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 1);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'z' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
__forceinline__  __device__ float decodeMortonCodeZ(unsigned int value)
{
    unsigned int expanded = compactBits(value);

    return expanded / 1024.0f;
}

/// <summary> Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit. </summary>
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
__forceinline__  __device__ float decodeMortonCode64X(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 2);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'y' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
__forceinline__  __device__ float decodeMortonCode64Y(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 1);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'z' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
__forceinline__  __device__ float decodeMortonCode64Z(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value);

    return expanded / 2097152.0f;
}

/// <summary> Expands the group bounding box using the specified new bounding box coordinates.
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