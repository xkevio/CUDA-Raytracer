#include <algorithm>
#include "inc/Ray.hpp"
#include "inc/Sphere.hpp"

Vec3f Ray::at(float t) const {
    return this->origin + (t * this->dir);
}

std::optional<float> Ray::get_intersection_t(const Sphere &sphere) const {
    auto a = dot(dir, dir);
    auto b = dot((2.0f * (dir)), (origin - sphere.center));
    auto c = dot((origin - sphere.center), (origin - sphere.center)) - std::pow(sphere.radius, 2.0f);

    auto d = b * b - 4 * (a * c);
    if (d < 0) return std::nullopt;

    const float t0 = ((-b - std::sqrt(d)) / (2 * a));
    const float t1 = ((-b + std::sqrt(d)) / (2 * a));

    if (d == 0) return std::make_optional<>(t0);
    if (t0 < 0 && t1 < 0) return std::nullopt;
    if (t0 > 0 && t1 < 0) return std::make_optional<>(t0);
    if (t0 < 0 && t1 > 0) return std::make_optional<>(t1);
    return t0 < t1 ? std::make_optional<>(t0) : std::make_optional<>(t1);
}

std::optional<std::pair<Sphere, float>> Ray::get_closest_intersection_t(const std::vector<Sphere> &spheres) const {
    std::vector<std::pair<float, Sphere>> intersections;

    for (const auto &item: spheres) {
        auto possible_intersection = this->get_intersection_t(item);
        if (possible_intersection) {
            intersections.emplace_back(*possible_intersection, item);
        }
    }

    if (intersections.empty()) {
        return std::nullopt;
    }

    auto it = std::min_element(intersections.begin(), intersections.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    const auto &[inter, sphere] = intersections[std::distance(intersections.begin(), it)];
    return std::make_optional(std::make_pair(sphere, inter));
}
