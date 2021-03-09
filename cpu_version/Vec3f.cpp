#include "Vec3f.h"

#include <cmath>
#include <iostream>

Vec3f Vec3f::operator+(const Vec3f& a) const { return Vec3f(x + a.x, y + a.y, z + a.z); }

Vec3f Vec3f::operator-(const Vec3f& a) const { return Vec3f(x - a.x, y - a.y, z - a.z); }

Vec3f Vec3f::operator/(const float& a) const { return scale(1 / a); }

Vec3f operator*(const float& a, const Vec3f& b) { return Vec3f(a * b.x_(), a * b.y_(), a * b.z_()); }

Vec3f Vec3f::operator^(const Vec3f& a) const { return Vec3f(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x); }

Vec3f Vec3f::operator&(const Vec3f& a) const { return Vec3f(x * a.x, y * a.y, z * a.z); }

Vec3f Vec3f::scale(float factor) const { return Vec3f(factor * x, factor * y, factor * z); }

Vec3f Vec3f::normalize() const { return scale(1 / length()); }

Vec3f Vec3f::cap(float max) const {
    float nx = x;
    float ny = y;
    float nz = z;
    if (x > max) nx = max;
    if (y > max) ny = max;
    if (z > max) nz = max;
    return Vec3f(nx, ny, nz);
}

float Vec3f::length() const { return sqrt(x*x + y*y + z*z); }

void Vec3f::display() { std::cout << "(" << x << " " << y << " " << z << ")^T" << std::endl; }

float Vec3f::x_() const { return x; }
float Vec3f::y_() const { return y; }
float Vec3f::z_() const { return z; }