#ifndef CUDA_RAYTRACER_COLOR_HPP
#define CUDA_RAYTRACER_COLOR_HPP

#include <vector>
#include "vec3f.hpp"
#include "light.hpp"
#include "sphere.hpp"
#include "ray.hpp"

namespace color_util {
    constexpr Color BG_COLOR = { 94, 156, 255 };

    Color convert_to_color(const Vec3f& v);
    Color get_color_at(const Ray& r, float intersection, const Light& light, const Sphere& sphere, const std::vector<Sphere>& spheres);
}

#endif //CUDA_RAYTRACER_COLOR_HPP
