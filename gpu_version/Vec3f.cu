#include "Vec3f.cuh"

#include <cmath>

__host__ __device__ Vec3f Vec3f::operator+(const Vec3f& a) const {
    return Vec3f(mx + a.mx, my + a.my, mz + a.mz);
}

__host__ __device__ Vec3f Vec3f::operator-(const Vec3f& a) const {
    return Vec3f(mx - a.mx, my - a.my, mz - a.mz);
}

__host__ __device__ Vec3f Vec3f::operator/(float a) const {
    return scale(1 / a);
}

__host__ __device__ Vec3f operator*(float a, const Vec3f& b) {
    return Vec3f(a * b.x(), a * b.y(), a * b.z());
}

__host__ __device__ Vec3f operator*(const Vec3f& b, float a) {
    return a * b;
}

__host__ __device__ Vec3f Vec3f::operator&(const Vec3f& a) const {
    return Vec3f(mx * a.mx, my * a.my, mz * a.mz);
}

__host__ __device__ Vec3f Vec3f::scale(float factor) const {
    return Vec3f(factor * mx, factor * my, factor * mz);
}

__host__ __device__ Vec3f Vec3f::normalize() const {
    return scale(1 / length());
}

__host__ __device__ Vec3f Vec3f::cap(float max) const {
    float nx = mx;
    float ny = my;
    float nz = mz;
    if(mx > max) nx = max;
    if(my > max) ny = max;
    if(mz > max) nz = max;
    return Vec3f(nx, ny, nz);
}

__host__ __device__ float Vec3f::length() const {
    return std::sqrt(mx*mx + my*my + mz*mz);
}

__host__ __device__ float Vec3f::x() const { 
    return mx; 
}

__host__ __device__ float Vec3f::y() const { 
    return my; 
}

__host__ __device__ float Vec3f::z() const { 
    return mz; 
}

__host__ __device__ Vec3f cross(const Vec3f &a, const Vec3f& b) {
    return Vec3f(a.y() * b.z() - a.z() * b.y(), 
                 a.z() * b.x() - a.x() * b.z(), 
                 a.x() * b.y() - a.y() * b.x());
}

__host__ __device__ std::ostream& operator<<(std::ostream& os, const Vec3f& a) {
    os << "(" << a.x() << " " << a.y() << " " << a.z() << ")";
    return os;
}

__host__ __device__ float dot(const Vec3f& a, const Vec3f& b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
