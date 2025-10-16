#pragma once
#include <curand_kernel.h>
#include <cuda/std/limits>
#include <thrust/unique.h>


#ifdef __FP_T_D__
    using FP_T = double;
#else
    using FP_T = float;
#endif

#ifdef DEBUG
constexpr int debug_col = 1408;
constexpr int debug_row = 1259;
#endif

#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

#define SENTINEL -1
#define INVALID -1

constexpr unsigned MAX_COLLISIONS = 512;

#define MAX_DEPTH 5
#define CACHE_DIM 33 // 1 << 5 + 1
#define CACHE_SIZE (CACHE_DIM * CACHE_DIM)
#define MAX_QUADS_PER_PIXEL 128

typedef unsigned long long morton_t;
typedef int long long delta_t;

// A helper struct to manage quads for subdivision
struct Quad {
    float u, v;     // Top-left corner of the quad within the pixel (0.0 to 1.0)
    float size;     // Size of the quad (e.g., 1.0, 0.5, 0.25...)
    int depth;      // Current subdivision depth
};

// A struct to store the final, converged quads
struct FinalQuad {
    float value;
    float area;
};

// --- 2. Define Vector Types from Base Precision ---
template<typename T> struct Types;

template<>
struct Types<float> {
    using Scalar = float;
    using Vec2 = float2;
    using Vec3 = float3;
    using Vec4 = float4;
};

template<>
struct Types<double> {
    using Scalar = double;
    using Vec2 = double2;
    using Vec3 = double3;
    using Vec4 = double4_32a;
};

// Use the Types struct to define the final type aliases
using FP_T2 = typename Types<FP_T>::Vec2;
using FP_T3 = typename Types<FP_T>::Vec3;
using FP_T4 = typename Types<FP_T>::Vec4;


// --- 3. Define `make_*` Helpers (Macro Required for C-style Structs) ---
#ifdef __FP_T_D__
    // Double precision make_* functions
    #define MAKE_FP_T2(x, y) make_double2(x, y)
    #define MAKE_FP_T3(x, y, z) make_double3(x, y, z)
    #define MAKE_FP_T4(x, y, z, w) make_double4_32a(x, y, z, w)
#else
    // Float precision make_* functions
    #define MAKE_FP_T2(x, y) make_float2(x, y)
    #define MAKE_FP_T3(x, y, z) make_float3(x, y, z)
    #define MAKE_FP_T4(x, y, z, w) make_float4(x, y, z, w)
#endif


// --- 4. Templated Operators (C++17 Compatible) ---
// This single block of code works for all float and double vector types.

// Helper to get the scalar type (float/double) from a vector type
template<typename VecT> struct ScalarType;
template<> struct ScalarType<float2> { using type = float; };
template<> struct ScalarType<float3> { using type = float; };
template<> struct ScalarType<float4> { using type = float; };
template<> struct ScalarType<double2> { using type = double; };
template<> struct ScalarType<double3> { using type = double; };
template<> struct ScalarType<double4_32a> { using type = double; };

// SFINAE helpers to check for members .z and .w (C++17 compatible)
template<typename T, typename = void> struct has_z : std::false_type {};
template<typename T> struct has_z<T, std::void_t<decltype(T::z)>> : std::true_type {};

template<typename T, typename = void> struct has_w : std::false_type {};
template<typename T> struct has_w<T, std::void_t<decltype(T::w)>> : std::true_type {};

// Vector-Vector addition
template <typename VecT>
__device__ __forceinline__ VecT operator+(const VecT& a, const VecT& b) {
    VecT result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    if constexpr (has_z<VecT>::value) { result.z = a.z + b.z; }
    return result;
}

// Vector-Vector subtraction
template <typename VecT>
__device__ __forceinline__ VecT operator-(const VecT& a, const VecT& b) {
    VecT result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    if constexpr (has_z<VecT>::value) { result.z = a.z - b.z; }
    return result;
}

// Vector-Scalar multiplication
template <typename VecT>
__device__ __forceinline__ VecT operator*(const VecT& v, typename ScalarType<VecT>::type s) {
    VecT result;
    result.x = v.x * s;
    result.y = v.y * s;
    if constexpr (has_z<VecT>::value) { result.z = v.z * s; }
    return result;
}

// Scalar-Vector multiplication
template <typename VecT>
__device__ __forceinline__ VecT operator*(typename ScalarType<VecT>::type s, const VecT& v) {
    return v * s; // Reuse the above operator
}

// Vector-Scalar division
template <typename VecT>
__device__ __forceinline__ VecT operator/(const VecT& v, typename ScalarType<VecT>::type s) {
    VecT result;
    result.x = v.x / s;
    result.y = v.y / s;
    if constexpr (has_z<VecT>::value) { result.z = v.z / s; }
    return result;
}

// Vector-Vector multiplication (component-wise)
template <typename VecT>
__device__ __forceinline__ VecT operator*(const VecT& a, const VecT& b) {
    VecT result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    if constexpr (has_z<VecT>::value) { result.z = a.z * b.z; }
    return result;
}

template <typename VecT>
__device__ __forceinline__ VecT abs(const VecT& v) {
    VecT result;
    result.x = fabs(v.x);
    result.y = fabs(v.y);
    if constexpr (has_z<VecT>::value) { result.z = fabs(v.z); }
    return result;
}

// Minimum of two vectors (component-wise)
template <typename VecT>
__device__ __forceinline__ VecT min(const VecT& a, const VecT& b) {
    VecT result;
    result.x = fmin(a.x, b.x);
    result.y = fmin(a.y, b.y);
    if constexpr (has_z<VecT>::value) { result.z = fmin(a.z, b.z); }
    return result;
}

// Maximum of two vectors (component-wise)
template <typename VecT>
__device__ __forceinline__ VecT max(const VecT& a, const VecT&  b) {
    VecT result;
    result.x = fmax(a.x, b.x);
    result.y = fmax(a.y, b.y);
    if constexpr (has_z<VecT>::value) { result.z = fmax(a.z, b.z); }
    return result;
}

// Dot Product (explicit function)
template <typename VecT>
__device__ __forceinline__ typename ScalarType<VecT>::type dot(const VecT& a, const VecT& b) {
    auto result = a.x * b.x + a.y * b.y;
    if constexpr (has_z<VecT>::value) { result += a.z * b.z; }
    // Note: Dot product typically ignores the .w component
    return result;
}

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


struct AreFPValuesClose {
    // A robust absolute tolerance calculated from the overall mesh scale.
    const FP_T scale_dependent_tolerance;

    // A small factor for comparing large t-values, defaults to machine epsilon.
    const FP_T relative_epsilon;

    /**
     * @brief Constructor for the floating-point comparison functor.
     * @param scale The overall scale of the mesh (e.g., the length of its AABB diagonal).
     * @param rel_ep A small factor for the relative comparison part.
     */
    __device__ AreFPValuesClose(FP_T scale, FP_T rel_ep = ::cuda::std::numeric_limits<FP_T>::epsilon()) :
        // Calculate a small, absolute tolerance relative to the entire mesh's size
        scale_dependent_tolerance(scale*.01),
        relative_epsilon(rel_ep)
    {}

    /**
     * @brief Compares two floating-point values using a combined absolute and relative tolerance.
     * @param a The first value.
     * @param b The second value.
     * @return True if the values are considered "close enough".
     */
    __device__ bool operator()(FP_T a, FP_T b) const {
        // The standard check for equality to handle identical values and infinities correctly.
        if (a == b) {
            return true;
        }

        const FP_T diff = (fabs)(a - b);

        // Use the larger of a fixed, scale-dependent tolerance or a relative tolerance.
        // This is robust for values both close to zero and for very large values.
        const FP_T tolerance = (fmax)(scale_dependent_tolerance, relative_epsilon * (fmax)((fabs)(a), (fabs)(b)));

        return diff < tolerance;
    }
};

// Swap two integers
template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
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

__device__ inline FP_T3 cross3(FP_T3 a, FP_T3 b) // cross product between two 3D vectors
{
    FP_T3 c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
    return c;
}

__device__ inline void _swap(FP_T *array, int i, int j) {
	FP_T tmp;

	tmp = array[i];
	array[i] = array[j];
	array[j] = tmp;
}

__device__ inline void _sift_down(FP_T *heap, int start, int end) {
    int root = start;
    int child;

    while (root*2 + 1 <= end) {
        child = root*2 + 1;
        if (child + 1 <= end && (heap[child] < heap[child + 1] ||
        								isnan(heap[child + 1]))) {
            child++;
        }
        if (child <= end && (heap[root] < heap[child] || isnan(heap[child]))) {
        	_swap(heap, root, child);
            root = child;
        } else {
            return;
        }
    }
}

__device__ inline void _heapify(FP_T *array, int size) {
	int start = (size - 2) / 2;

	while (start >= 0) {
		_sift_down(array, start, size - 1);
		start--;
	}
}

__device__ inline void sort(FP_T *array, int size) {
	_heapify(array, size);
    int end = size - 1;

    while (end > 0) {
    	_swap(array, 0, end);
        _sift_down(array, 0, end - 1);
        end--;
    }
}

__device__ int inline max_dim_index(const FP_T4& v) {
    if (v.x > v.y) {
        return (v.x > v.z) ? 0 : 2;
    } else {
        return (v.y > v.z) ? 1 : 2;
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

    centroid.x = (bboxMin.x + bboxMax.x) / (2.0);
    centroid.y = (bboxMin.y + bboxMax.y) / (2.0);
    centroid.z = (bboxMin.z + bboxMax.z) / (2.0);

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