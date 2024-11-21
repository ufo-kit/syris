#pragma once
#include "Commons.cuh"

class Ray {
public:
    __device__ Ray() {};

    __device__ Ray(const Ray &ray);
    __device__ Ray(float4 tail, float4 direction);

    __device__ float normalize(float4 &v);
    __device__ void updateDirection();
    __device__ void updateInvDirection();
    __device__ void updateSign();
    __device__ float4 computeParametric(float t);
    __device__ bool intersects (float4 const &minBbox, float4 const &maxBbox, float &tmin, float &tmax); // AABB
    __device__ bool intersects (float4 const &minBbox, float4 const &maxBbox); // AABB
    __device__ bool intersects (float4 const &V1, float4 const &V2, float4 const &V3, float &tmin, float &tmax); // Triangle
    __device__ bool intersects (float4 const &V1, float4 const &V2, float4 const &V3, float &t); // Triangle

    // setters
    __device__ void setTail(float4 &T);
    __device__ void setHead(float4 &H);
    __device__ void setDirection(float4 &D);
    __device__ void setInvDirection(float4 &ID);

    // getters
    __device__ float4 getTail() { return this->tail; };
    __device__ float4 getHead() { return this->head; };
    __device__ float4 getDirection() { return this->direction; };
    __device__ float4 getInvDirection() { return this->invDirection; };
    __device__ float4 getOppositeDirection() { return make_float4(-this->direction.x, -this->direction.y, -this->direction.z, 0.0f); };
    __device__ int getSign(int i) { return this->sign[i]; };


    __device__ void print();

private:
    float4 tail, head;
    float4 direction, invDirection;
    float Sx, Sy, Sz;
    int Kx, Ky, Kz;

    int sign[3];
};