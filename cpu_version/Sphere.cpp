#include "inc/Sphere.hpp"

Vec3f Sphere::get_normal_at(const Vec3f& at) const {
    return Vec3f(at - this->center).normalize();
}