#ifndef CUDA_RAYTRACER_SPHERE_HPP
#define CUDA_RAYTRACER_SPHERE_HPP

#include "Vec3f.hpp"

struct Sphere {
    float radius;
    Vec3f center;
    Color color;

    Sphere(float r, const Vec3f& c, const Vec3f& col) : radius(r), center(c), color(col) {}
    [[nodiscard]] Vec3f get_normal_at(const Vec3f& at) const;
};

#endif //CUDA_RAYTRACER_SPHERE_HPP
