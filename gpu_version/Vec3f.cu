#include "Vec3f.h"
#include <cmath>
#include <iostream>

__host__ __device__ Vec3f::Vec3f(float x_, float y_, float z_) {
    x = x_;
    y = y_;
    z = z_;
}

__host__ __device__ Vec3f::Vec3f(const Vec3f &a) {
    x = a.x;
    y = a.y;
    z = a.z;
}

__host__ __device__ Vec3f::Vec3f() {
    x = 0; y = 0; z = 0;
}

__host__ __device__ Vec3f Vec3f::operator+(const Vec3f &a) const {
    return Vec3f(x + a.x, y + a.y, z + a.z);
}

__host__ __device__ Vec3f Vec3f::operator-(const Vec3f &a) const {
    return Vec3f(x - a.x, y - a.y, z - a.z);
}

__host__ __device__ Vec3f Vec3f::operator/(const float &a) const {
    return scale(1 / a);
}

__host__ __device__ Vec3f operator*(const float &a, const Vec3f &b) {
    return Vec3f(a * b.x_(), a * b.y_(), a * b.z_());
}

__host__ __device__ Vec3f Vec3f::operator^(const Vec3f &a) const {
    return Vec3f(y*a.z - z*a.y, z*a.x - x*a.z, x*a.y - y*a.x);
}

__host__ __device__ Vec3f Vec3f::operator&(const Vec3f &a) const {
    return Vec3f(x * a.x, y * a.y, z * a.z);
}

__host__ __device__ Vec3f Vec3f::scale(float factor) const {
    return Vec3f(factor * x, factor * y, factor * z);
}

__host__ __device__ Vec3f Vec3f::normalize() const {
    return scale(1 / length());
}

__host__ __device__ Vec3f Vec3f::cap(float max) const {
    float nx = x;
    float ny = y;
    float nz = z;
    if(x > max) nx = max;
    if(y > max) ny = max;
    if(z > max) nz = max;
    return Vec3f(nx, ny, nz);
}

__host__ __device__ float Vec3f::length() const {
    return sqrt(pow(x, 2) + pow(y,2) + pow(z, 2));
}

__host__ __device__ void Vec3f::display() {
    printf("(%f %f %f)^T\n", x, y, z);
}

__host__ __device__ float Vec3f::x_() const { return x; }
__host__ __device__ float Vec3f::y_() const { return y; }
__host__ __device__ float Vec3f::z_() const { return z; }