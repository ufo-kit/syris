#pragma once

#include "Commons.cuh"

extern "C" __global__ void calculateBbBoxKernel(FP_T4 *vertices, FP_T4 *bbMin, FP_T4 *bbMax, unsigned int nb_keys);
extern "C" __global__ void projectTriangleCentroid(
    unsigned int const nb_keys, FP_T4 const *vertices, unsigned int *keys,
    FP_T4 *bbMin, FP_T4 *bbMax, FP_T4 const scene_bbMin, FP_T4 const scene_bbMax);
extern "C" __global__ void growTreeKernel(
    unsigned int nb_keys, unsigned int *keys, unsigned int *permutation,
    int *rope, int *left, int *entered,
    FP_T4 *bboxMin, FP_T4 *bboxMax);
extern "C" __global__ void project_parallel_kernel(
    unsigned nb_keys, FP_T *image, uint2 N,
    FP_T4 U, FP_T4 V, FP_T4 W, // projection basis and origin
    FP_T4 upperleft_origin, FP_T2 ps,
    int *rope,
    int *left,
    unsigned *permutation, // BVH tree
    FP_T4 *bboxMin,
    FP_T4 *bboxMax,
    FP_T4 const scene_bbMin, FP_T4 const scene_bbMax,
    FP_T4 *__restrict__ vertices,
    unsigned *globalCounter,
    FP_T epsilon
);