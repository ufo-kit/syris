#pragma once
#include <cuda/std/limits>
#include <thrust/unique.h>


#ifdef __FP_T_D__
    using FP_T = double;
    using FP_T2 = double2;
    using FP_T3 = double3;
    using FP_T4 = double4;

    #define FP_MATH(func) func
    #define FP_CONST(val) val
    #define MAKE_FP_T2(x, y) make_double2(x, y)
    #define MAKE_FP_T4(x, y, z, w) make_double4(x, y, z, w)
#else
    using FP_T = float;
    using FP_T2 = float2;
    using FP_T3 = float3;
    using FP_T4 = float4;

    #define FP_MATH(func) func##f
    #define FP_CONST(val) val##f
    #define MAKE_FP_T2(x, y) make_float2(x, y)
    #define MAKE_FP_T4(x, y, z, w) make_float4(x, y, z, w)
#endif

constexpr unsigned MAX_COLLISIONS = 512;

__forceinline__ __device__ __host__ FP_T4 make_fp_t4(FP_T x, FP_T y, FP_T z, FP_T w)
{
    FP_T4 result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
}

template <typename T>
struct List {
    T values[MAX_COLLISIONS];
    unsigned count = 0;

    __device__ bool push_back(const T& value) {
        if (count < MAX_COLLISIONS) {
            values[count++] = value;
            return true;
        }
        else {
            return false;
        }
    }

    __device__ void push_back_atomic(const T& value) {
        unsigned index = atomicAdd(&count, 1);
        if (index < MAX_COLLISIONS) {
            values[index] = value;
        }
        else {
            // Decrease count if we exceeded the limit to avoid further overflows
            atomicSub(&count, 1);
        }
    }

    __device__ T& back() {
        // Note: Assumes the list is not empty!
        return values[count - 1];
    }

    // Returns a const reference to the last element (for read-only access)
    __device__ const T& back() const {
        // Note: Assumes the list is not empty!
        return values[count - 1];
    }

    __device__ void insert(unsigned index, const T& value) {
        if (index > count || count >= MAX_COLLISIONS) return;
        
        for (unsigned i = count; i > index; --i) {
            values[i] = values[i - 1];
        }

        values[index] = value;
        count++;
    }

    __device__ void remove(unsigned index) {
        if (index >= count || count == 0) return;

        // Shift elements to the left
        for (unsigned i = index; i < count - 1; ++i) {
            values[i] = values[i + 1];
        }

        count--;
    }

    __device__ T get(unsigned index) const {
        return values[index];
    }

    __device__ void set(unsigned index, const T& value) {
        values[index] = value;
        if (index >= count) {
            count = index + 1;
        }
    }

    __device__ unsigned size() const {
        return count;
    }
};

template <typename T>
__forceinline__ __device__ bool is_null(T const val, T const epsilon = 1e-8f)
{
    return (val < epsilon) && (val > -epsilon);
}

template <typename T>
__forceinline__ __device__ bool are_close(T const a, T const b, T const epsilon)
{
    return FP_MATH(fabs)(a - b) <= epsilon * FP_MATH(fmax)(FP_CONST(1.0), FP_MATH(fmax)(FP_MATH(fabs)(a), FP_MATH(fabs)(b)));
}

__device__ FP_T4 operator+(const FP_T4 &lhs, const FP_T4 &rhs)
{
    return make_fp_t4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

__device__ FP_T4 operator-(const FP_T4 &lhs, const FP_T4 &rhs)
{
    return make_fp_t4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

__device__ FP_T4 operator*(const FP_T4 &lhs, const FP_T &rhs)
{
    return make_fp_t4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}

__device__ FP_T4 operator/(const FP_T4 &lhs, const FP_T &rhs)
{
    return make_fp_t4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

__device__ FP_T4 operator*(const FP_T &lhs, const FP_T4 &rhs)
{
    return make_fp_t4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

__device__ FP_T4 operator/(const FP_T &lhs, const FP_T4 &rhs)
{
    return make_fp_t4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

__device__ FP_T operator*(const FP_T4 &lhs, const FP_T4 &rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

// Swap two integers
template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}


__device__ inline FP_T xor_signmask(FP_T x, int y)
{
    return (FP_T)(int(x) ^ y);
}

__device__ inline FP_T4 abs4(FP_T4 a)
{
    FP_T4 c;
    c.x = fabs(a.x);
    c.y = fabs(a.y);
    c.z = fabs(a.z);
    c.w = fabs(a.w);
    return c;
}

__device__ inline FP_T4 min4(FP_T4 a, FP_T4 b)
{
    FP_T4 c;
    c.x = fmin(a.x, b.x);
    c.y = fmin(a.y, b.y);
    c.z = fmin(a.z, b.z);
    c.w = fmin(a.w, b.w);
    return c;
}

__device__ inline FP_T4 max4(FP_T4 a, FP_T4 b)
{
    FP_T4 c;
    c.x = fmax(a.x, b.x);
    c.y = fmax(a.y, b.y);
    c.z = fmax(a.z, b.z);
    c.w = fmax(a.w, b.w);
    return c;
}

__device__ inline FP_T4 cross4(FP_T4 a, FP_T4 b) // cross product between two 3D vectors
{
    FP_T4 c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
    c.w = 0;

    return c;
}

__device__ inline FP_T4 pointwise_product (FP_T4 a, FP_T4 b)
{
    FP_T4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    return c;
}

__device__ inline FP_T dot4(FP_T4 a, FP_T4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline FP_T max_component(FP_T4 a)
{
    return FP_MATH(fmax)(FP_MATH(fmax)(a.x, a.y), a.z);
}

__device__ inline FP_T min_component(FP_T4 a)
{
    return FP_MATH(fmin)(FP_MATH(fmin)(a.x, a.y), a.z);
}

__device__ inline int maxDimIndex(const FP_T4 &D)
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

__forceinline__ __device__ void calculateTriangleBoundingBox(
    FP_T4 const &vertex1, FP_T4 const &vertex2, FP_T4 const &vertex3, FP_T4 &boundingBoxMin, FP_T4 &boundingBoxMax)
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

__device__ inline FP_T4 getBoundingBoxCentroid(FP_T4 bboxMin, FP_T4 bboxMax)
{
    FP_T4 centroid;

    centroid.x = (bboxMin.x + bboxMax.x) / FP_CONST(2.0);
    centroid.y = (bboxMin.y + bboxMax.y) / FP_CONST(2.0);
    centroid.z = (bboxMin.z + bboxMax.z) / FP_CONST(2.0);

    return centroid;
}

// --- Morton codes -------------------------------------------------------------------------------
__forceinline__ __device__ FP_T4 normalize(FP_T4 point, FP_T4 boundingBoxMin,
                                            FP_T4 boundingBoxMax)
{
    FP_T4 normalized;
    normalized.x = (point.x - boundingBoxMin.x) / (boundingBoxMax.x - boundingBoxMin.x);
    normalized.y = (point.y - boundingBoxMin.y) / (boundingBoxMax.y - boundingBoxMin.y);
    normalized.z = (point.z - boundingBoxMin.z) / (boundingBoxMax.z - boundingBoxMin.z);
    return normalized;
}

__forceinline__ __device__ FP_T4 denormalize(FP_T4 normalized, FP_T4 bboxMin,
                                              FP_T4 bboxMax)
{
    FP_T4 point;
    point.x = bboxMin.x + (bboxMax.x - bboxMin.x) * normalized.x;
    point.y = bboxMin.y + (bboxMax.y - bboxMin.y) * normalized.y;
    point.z = bboxMin.z + (bboxMax.z - bboxMin.z) * normalized.z;
    return point;
}

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned int expandBits(unsigned int value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

template <int N>
__forceinline__ __device__ unsigned int expandBitsBy(unsigned int)
{
    static_assert(0 <= N && N < 10,
                  "expandBitsBy can only be used with values 0-9");

    return 0;
}

template <>
__forceinline__ __device__ unsigned int expandBitsBy<0>(unsigned int x)
{
    return x;
}

template <>
__forceinline__ __device__ unsigned int expandBitsBy<1>(unsigned int x)
{
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

template <>
__forceinline__ __device__ unsigned int expandBitsBy<2>(unsigned int x)
{
    x &= 0x000003ffu;
    x = (x ^ (x << 16)) & 0xff0000ffu;
    x = (x ^ (x << 8)) & 0x0300f00fu;
    x = (x ^ (x << 4)) & 0x030c30c3u;
    x = (x ^ (x << 2)) & 0x09249249u;
    return x;
}

template <>
__forceinline__ __device__ unsigned int expandBitsBy<3>(unsigned int x)
{
    x &= 0xffu;
    x = (x | x << 16) & 0xc0003fu;
    x = (x | x << 8) & 0xc03807u;
    x = (x | x << 4) & 0x8430843u;
    x = (x | x << 2) & 0x9090909u;
    x = (x | x << 1) & 0x11111111u;
    return x;
}
__forceinline__ __device__ unsigned int calculateMortonCode(FP_T4 point)
{
    // Discretize the unit cube into a 10 bit integer
    constexpr unsigned N = 1u << 10;

    FP_T p[3] = {point.x, point.y, point.z};

    unsigned r = 0;
    for (int d = 0; d < 3; ++d)
    {
        auto x = min(max(p[d] * N, FP_T(0)), FP_T(N - 1));
        r += (expandBitsBy<2>((unsigned int)x) << (3 - d - 1));
    }
    return r;
}

// Compact bits from the specified 30-bit value, using only one bit at every 3 from the original value and forming a 10-bit value
__forceinline__ __device__ unsigned int compactBits(unsigned int value)
{
    unsigned int compacted = value;
    compacted &= 0x09249249;
    compacted = (compacted ^ (compacted >> 2)) & 0x030c30c3;
    compacted = (compacted ^ (compacted >> 4)) & 0x0300f00f;
    compacted = (compacted ^ (compacted >> 8)) & 0xff0000ff;
    compacted = (compacted ^ (compacted >> 16)) & 0x000003ff;
    return compacted;
}

// Decodes the 'x' coordinate from a 30-bit morton code. The returned value is a float between 0 and 1
__forceinline__ __device__ FP_T decodeMortonCodeX(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 2);

    return expanded / FP_T(1024.0);
}

// Decodes the 'y' coordinate from a 30-bit morton code. The returned value is a float between 0 and 1.
__forceinline__ __device__ FP_T decodeMortonCodeY(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 1);

    return expanded / FP_T(1024.0);
}

// Decodes the 'z' coordinate from a 30-bit morton code. The returned value is a float between 0 and 1.
__forceinline__ __device__ FP_T decodeMortonCodeZ(unsigned int value)
{
    unsigned int expanded = compactBits(value);

    return expanded / FP_T(1024.0);
}

// Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned long long int expandBits64(
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

// Calculates the point morton code using 63 bits.
__forceinline__ __device__ unsigned long long int calculateMortonCode64(FP_T4 point)
{
    // Discretize the unit cube into a 10 bit integer
    unsigned long long int discretized[3];
    discretized[0] = (unsigned long long int)min(max(point.x * FP_T(2097152.0), FP_T(0.0)), FP_T(2097151.0));
    discretized[1] = (unsigned long long int)min(max(point.y * FP_T(2097152.0), FP_T(0.0)), FP_T(2097151.0));
    discretized[2] = (unsigned long long int)min(max(point.z * FP_T(2097152.0), FP_T(0.0)), FP_T(2097151.0));

    discretized[0] = expandBits64(discretized[0]);
    discretized[1] = expandBits64(discretized[1]);
    discretized[2] = expandBits64(discretized[2]);

    return discretized[0] * 4 + discretized[1] * 2 + discretized[2];
}

// Compact bits from the specified 63-bit value, using only one bit at every 3 from the original value and forming a 21-bit value.
__forceinline__ __device__ unsigned long long int compactBits64(
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

// Decodes the 'x' coordinate from a 63-bit morton code. The returned value is a float between 0 and 1.
__forceinline__ __device__ FP_T decodeMortonCode64X(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 2);

    return expanded / FP_T(2097152.0);
}

// Decodes the 'y' coordinate from a 63-bit morton code. The returned value is a float between 0 and 1.
__forceinline__ __device__ FP_T decodeMortonCode64Y(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 1);

    return expanded / FP_T(2097152.0);
}

// Decodes the 'z' coordinate from a 63-bit morton code. The returned value is a float between 0 and 1.
__forceinline__ __device__ FP_T decodeMortonCode64Z(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value);

    return expanded / FP_T(2097152.0);
}

// Expands the group bounding box using the specified new bounding box coordinates.
__forceinline__ __device__ void expandBoundingBox(FP_T4 &groupBbMin, FP_T4 &groupBbMax,
                                                  const FP_T4 &newBbMin, const FP_T4 &newBbMax)
{
    groupBbMin.x = min(newBbMin.x, groupBbMin.x);
    groupBbMin.y = min(newBbMin.y, groupBbMin.y);
    groupBbMin.z = min(newBbMin.z, groupBbMin.z);

    groupBbMax.x = max(newBbMax.x, groupBbMax.x);
    groupBbMax.y = max(newBbMax.y, groupBbMax.y);
    groupBbMax.z = max(newBbMax.z, groupBbMax.z);
}

// Device implementations
__device__ inline FP_T device_int_as_float(int i)
{
    return __int_as_float(i);
}

__device__ inline int device_float_as_int(FP_T f)
{
    return __float_as_int(f);
}

__device__ inline FP_T device_xorf(FP_T x, int y)
{
    return __int_as_float(__float_as_int(x) ^ y);
}

__device__ inline int device_sign_mask(FP_T x)
{
    return __float_as_int(x) & 0x80000000;
}

// Host implementations
inline FP_T host_int_as_float(int i)
{
    return *(FP_T *)(&i);
}

inline int host_float_as_int(FP_T f)
{
    return *(int *)(&f);
}

inline FP_T host_xorf(FP_T x, int y)
{
    return host_int_as_float(host_float_as_int(x) ^ y);
}

inline int host_sign_mask(FP_T x)
{
    return host_float_as_int(x) & 0x80000000;
}

// Unified interface for both host and device
__device__ inline FP_T __iaf(int i)
{
#ifdef __CUDA_ARCH__
    return device_int_as_float(i);
#else
    return host_int_as_float(i);
#endif
}

__device__ inline int __fai(FP_T f)
{
#ifdef __CUDA_ARCH__
    return device_float_as_int(f);
#else
    return host_float_as_int(f);
#endif
}

__device__ inline FP_T xorf(FP_T x, int y)
{
#ifdef __CUDA_ARCH__
    return device_xorf(x, y);
#else
    return host_xorf(x, y);
#endif
}

__device__ inline int sign_mask(FP_T x)
{
#ifdef __CUDA_ARCH__
    return device_sign_mask(x);
#else
    return host_sign_mask(x);
#endif
}