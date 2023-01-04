#include <algorithm>
#include "inc/color.hpp"

namespace color_util {
    Color convert_to_color(const Vec3f& v) {
        return Color{std::trunc(v.x() * 255.999f), std::trunc(v.y() * 255.999f), std::trunc(v.z() * 255.999f)};
    }

    Color get_color_at(const Ray& r, float intersection, const Light& light, const Sphere& sphere, const std::vector<Sphere>& spheres) {
        float shadow = 1;

        const Vec3f normal = sphere.get_normal_at(r.at(intersection));
        const Vec3f to_camera((Vec3f(0, 0, 1) - r.at(intersection)).normalize());
        const Vec3f light_ray((light.get_position() - r.at(intersection)).normalize());

        Vec3f reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
        reflection_ray = reflection_ray.normalize();

        const Ray rr(r.at(intersection) + 0.001 * normal, reflection_ray);
        auto new_inter = rr.get_closest_intersection_t(spheres);

        bool reflect = false;
        float reflect_shadow = 1;

        // one extra reflection hardcoded
        if (new_inter) {
            const auto& [new_sphere, inter] = *new_inter;

            const Ray rs(rr.at(inter) + 0.001 * new_sphere.get_normal_at(rr.at(inter)),
                   light.get_position() - rr.at(inter) + 0.001 * new_sphere.get_normal_at(rr.at(inter)));

            for (const auto& obj : spheres) {
                auto opt_t = rs.get_intersection_t(obj);
                if (opt_t && *opt_t > 0.000001f) {
                    reflect_shadow = 0.35;
                    break;
                }
            }

            reflect = true;
        }

        auto ambient = light.get_ambient() * light.get_color();
        auto diffuse = (light.get_diffuse() * std::max(dot(light_ray, normal), 0.0f)) * light.get_color();
        auto specular = light.get_specular() * std::pow(std::max(dot(reflection_ray, to_camera), 0.0f), 32.0f) * light.get_color();

        const Ray shadow_ray(r.at(intersection) + (0.001 * normal), light.get_position() - (r.at(intersection) + 0.001 * normal));

        for (const auto& obj : spheres) {
            auto opt_t = shadow_ray.get_intersection_t(obj);
            if (opt_t && *opt_t > 0.000001f) {
                shadow = 0.35;
                break;
            }
        }

        const auto& [new_sphere, _] = *new_inter;
        auto shadow_color = reflect_shadow * new_sphere.color;
        auto all_light = reflect ? (ambient + diffuse + specular).cap(1) & (0.55 * (sphere.color - shadow_color) + shadow_color).cap(1)
                                 : (ambient + diffuse + specular).cap(1) & sphere.color;

        return color_util::convert_to_color(shadow * all_light);
    }
}
