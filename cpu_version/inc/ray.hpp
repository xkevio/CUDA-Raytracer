#ifndef CUDA_RAYTRACER_RAY_HPP
#define CUDA_RAYTRACER_RAY_HPP

#include <optional>
#include <vector>
#include "vec3f.hpp"
#include "sphere.hpp"

struct Ray {
    Vec3f origin;
    Vec3f dir;

    Ray(const Vec3f& o, const Vec3f& d) : origin(o), dir(d) {}

    [[nodiscard]] Vec3f at(float t) const;
    [[nodiscard]] std::optional<float> get_intersection_t(const Sphere& sphere) const;
    [[nodiscard]] std::optional<std::pair<Sphere, float>> get_closest_intersection_t(const std::vector<Sphere>& spheres) const;
};

#endif //CUDA_RAYTRACER_RAY_HPP
