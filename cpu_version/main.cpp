#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <string>

#include "Light.h"
#include "Vec3f.h"

#define WIDTH 2048
#define HEIGHT 2048

typedef Vec3f Color;

struct Sphere {
    float radius;
    Vec3f center;
    Vec3f color;

    Sphere(float r, const Vec3f &c, const Vec3f &col) : radius(r), center(c), color(col) {}
    Vec3f get_normal_at(const Vec3f &at) const;
};

struct Ray {
    Vec3f origin;
    Vec3f dir;

    Ray(const Vec3f &o, const Vec3f &d) : origin(o), dir(d) {}

    Vec3f at(float t);
    float has_intersection(const Sphere &sphere);
};

Vec3f Sphere::get_normal_at(const Vec3f &at) const {
    return Vec3f(at - center).normalize();
}

Vec3f Ray::at(float t) {
    return origin + (t * dir); 
}

float dot(const Vec3f &a, const Vec3f &b) {
    return a.x_() * b.x_() + a.y_() * b.y_() + a.z_() * b.z_();
}

float Ray::has_intersection(const Sphere &sphere) {
    //auto n_c = sphere.center.normalize();

    auto a = dot(dir, dir);
    auto b = dot((2.0f * (dir)), (origin - sphere.center));
    auto c = dot((origin - sphere.center), (origin - sphere.center)) - pow(sphere.radius, 2);

    auto d = b*b - 4 * (a * c);
    if(d < 0) return -1.0;

    float t0 = ((-b - sqrt(d)) / (2*a));
    float t1 = ((-b + sqrt(d)) / (2*a));
    if(d == 0) return t0;
    if(t0 < 0 && t1 < 0) return -1;
    if(t0 > 0 && t1 < 0) return t0;
    if(t0 < 0 && t1 > 0) return t1;
    return t0 < t1 ? t0 : t1;
}

int get_closest_intersection(const std::vector<Sphere> &spheres, Ray &r, std::vector<float> &intersections) {
    int hp = -1;

    for(auto &obj : spheres) {
        intersections.push_back(r.has_intersection(obj));
    }

    if(intersections.size() == 0) {
        return -1;
    } else if(intersections.size() == 1) {
        return intersections[0] < 0 ? -1 : 0;
    } else {
        float min_val = 100.0;
		for (int i = 0; i < intersections.size(); i++) {
			if (intersections[i] < 0.0) continue;
			else if (intersections[i] < min_val) {
				min_val = intersections[i];
                hp = i;
			}
		}
    }
    return hp;
}

Color convert_to_color(const Vec3f &v) {
    return Color(static_cast<int> (1 * ((v.x_()) * 255.999)), static_cast<int> (1 * ((v.y_()) * 255.999)), static_cast<int> (1 * ((v.z_()) * 255.999)));
}

Color get_color_at(Ray &r, const float &intersection, const Light &light, Sphere &sphere, const std::vector<Sphere> &spheres) {
    auto t = intersection;
    float shadow = 1;

    Vec3f normal = sphere.get_normal_at(r.at(t));

    Vec3f to_camera(Vec3f(0, 0, 1) - r.at(t));
    to_camera = to_camera.normalize();

    Vec3f light_ray(light.get_position() - r.at(t));
    light_ray = light_ray.normalize();

    Vec3f reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
    reflection_ray = reflection_ray.normalize();

    Ray rr(r.at(t) + 0.001 * normal, reflection_ray);
    std::vector<float> intersections;
    int hp = get_closest_intersection(spheres, rr, intersections);
    bool reflect = false;
    float reflect_shadow = 1;
    if(hp != -1) {
        reflect = true;
        Ray rs(rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])), light.get_position() - rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])));
        for(auto &obj : spheres) {
            if(rs.has_intersection(obj) > 0.000001f) reflect_shadow = 0.35;
        }
    }

    auto ambient = light.get_ambient() * light.get_color(); 
    auto diffuse = (light.get_diffuse() * std::max(dot(light_ray, normal), 0.0f)) * light.get_color();
    auto specular = light.get_specular() * pow(std::max(dot(reflection_ray, to_camera), 0.0f), 32) * light.get_color();

    Ray shadow_ray(r.at(t) + (0.001 * normal), light.get_position() - (r.at(t) + 0.001 * normal));
    for(auto &obj : spheres) {
        if(shadow_ray.has_intersection(obj) > 0.000001) shadow = 0.35;
    }

    auto shadow_color = reflect_shadow * spheres[hp].color;
    auto all_light = reflect ? (ambient + diffuse + specular).cap(1) & (0.55 * (sphere.color - shadow_color) + shadow_color).cap(1) : (ambient + diffuse + specular).cap(1) & sphere.color;
    return convert_to_color(shadow * all_light);
}


int main(int, char**) { 
    std::ofstream pbm_file;
    pbm_file.open("img.ppm");
    pbm_file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";

    Vec3f* frame_buffer = new Vec3f[WIDTH * HEIGHT];
    std::vector<std::string> mem_buffer;

    //sphere stretches when not at 0,0
    Sphere sphere(.25, Vec3f(-0.35, -0.35, -1), Vec3f(0.8, 0, 0));
    Sphere sphere2(.25, Vec3f(0.35, -0.35, -1), Vec3f(0.137, 0.6, 0.63));
    Sphere sphere3(.25, Vec3f(0, 0.35, -1), Vec3f(0, 0.8, 0));
    Sphere sphere4(1000, Vec3f(0, -1002, 0), Vec3f(0.5, 0.5, 0.5));
    Vec3f origin(0, 0, 1);

    std::vector<Sphere> spheres;
    spheres.push_back(sphere);
    spheres.push_back(sphere2);
    spheres.push_back(sphere3);
    spheres.push_back(sphere4);

    Light light(Vec3f(1, 1, 1), Vec3f(1, 1, 1));
    light.set_light(.2, .5, .5);

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for
        for (size_t j = 0; j < HEIGHT; j++)
        {
            for (size_t i = 0; i < WIDTH; i++)
            {
                Vec3f ij(2 * (float(i + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float(j + 0.5) / (HEIGHT - 1)), -1);
                Vec3f *dir = new Vec3f(ij - origin);
                Ray r(origin, *dir);

                std::vector<float> intersections;
                
                int hit_point = get_closest_intersection(spheres, r, intersections);
                if(hit_point == -1) {
                    frame_buffer[(j * WIDTH) + i] = Color(94, 156, 255);
                } else {
                    auto color = get_color_at(r, intersections[hit_point], light, spheres[hit_point], spheres);
                    frame_buffer[(j * WIDTH) + i] = color;
                }                
            }
        }
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << ">> time (not incl. writing to file):  " << end << "ms" << std::endl;

    start = std::chrono::steady_clock::now();
    std::cout << ">> Saving Image..." << std::endl;
    for (size_t i = 0; i < WIDTH*HEIGHT; i++)
    {
        mem_buffer.push_back(std::to_string((int) frame_buffer[i].x_()) + " " + std::to_string((int) frame_buffer[i].y_()) + " " + std::to_string((int) frame_buffer[i].z_()));
    }
    std::ostream_iterator<std::string> output_iterator(pbm_file, "\n");
    std::copy(mem_buffer.begin(), mem_buffer.end(), output_iterator);
    auto new_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << ">> Image: \"img.ppm\" saved!" << std::endl;
    std::cout << ">> time spent writing to file:       " << new_end << "ms" << std::endl;
    std::cout << ">> time (all):                       " << end + new_end << "ms" << std::endl;
    
    
    delete frame_buffer;
    pbm_file.close();
    return EXIT_SUCCESS;
}