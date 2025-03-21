#pragma once
#include "Commons.cuh"

class Ray {
public:
    __device__ Ray() {};

    __device__ Ray(const Ray &ray);
    __device__ Ray(FP_T4 tail, FP_T4 direction);

    __device__ FP_T invmagnitude(FP_T4 &v);
    __device__ void updateDirection();
    __device__ void updateInvDirection();
    __device__ void updateSign();
    __device__ FP_T4 computeParametric(FP_T t);
    __device__ bool intersects (FP_T4 const &minBbox, FP_T4 const &maxBbox, FP_T &tmin, FP_T &tmax) const; // AABB
    __device__ bool intersects (FP_T4 const &minBbox, FP_T4 const &maxBbox) const; // AABB
    __device__ bool intersects (FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &tmin, FP_T &tmax) const; // Triangle
    __device__ bool intersects (FP_T4 const &V1, FP_T4 const &V2, FP_T4 const &V3, FP_T &t) const; // Triangle

    // setters
    __device__ void setTail(FP_T4 &T);
    __device__ void setHead(FP_T4 &H);
    __device__ void setDirection(FP_T4 &D);
    __device__ void setInvDirection(FP_T4 &ID);

    // getters
    __device__ FP_T4 getTail() const { return this->tail; };
    __device__ FP_T4 getHead() const { return this->head; };
    __device__ FP_T4 getDirection() const { return this->direction; };
    __device__ FP_T4 getInvDirection() const { return this->invDirection; };
    __device__ FP_T4 getOppositeDirection() const { return make_fp_t4(-this->direction.x, -this->direction.y, -this->direction.z, 0.0f); };
    __device__ int getSign(int i) const { return this->sign[i]; };


    __device__ void print() const;

private:
    FP_T4 tail, head;
    FP_T4 direction, invDirection;
    FP_T Sx, Sy, Sz;
    int Kx, Ky, Kz;

    int sign[3];
};