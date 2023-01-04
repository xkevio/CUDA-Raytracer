#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "inc/Light.hpp"
#include "inc/Vec3f.hpp"
#include "inc/Color.hpp"

using namespace std::chrono;

constexpr int WIDTH = 2048;
constexpr int HEIGHT = 2048;

int main() {
    std::ofstream pbm_file("img.ppm");
    pbm_file << "P3"
             << "\n"
             << WIDTH << " " << HEIGHT << "\n"
             << "255\n";

    std::vector<Vec3f> frame_buffer(WIDTH * HEIGHT);
    std::vector<std::string> mem_buffer;

    // sphere stretches when not at 0,0
    const Sphere sphere(.25, Vec3f(-0.35, -0.35, -1), Vec3f(0.8, 0, 0));
    const Sphere sphere2(.25, Vec3f(0.35, -0.35, -1), Vec3f(0.137, 0.6, 0.63));
    const Sphere sphere3(.25, Vec3f(0, 0.35, -1), Vec3f(0, 0.8, 0));
    const Sphere sphere4(1000, Vec3f(0, -1002, 0), Vec3f(0.5, 0.5, 0.5));
    const Vec3f origin(0, 0, 1);

    const std::vector<Sphere> spheres {sphere, sphere2, sphere3, sphere4};

    Light light(Vec3f(1, 1, 1), Vec3f(1, 1, 1));
    light.set_light(.2, .5, .5);

    auto start = steady_clock::now();
    #pragma omp parallel for default(none) shared(origin, spheres, light, frame_buffer)
    for (size_t j = 0; j < HEIGHT; j++) {
        for (size_t i = 0; i < WIDTH; i++) {
            const Vec3f ij(2 * (float(i + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float(j + 0.5) / (HEIGHT - 1)), -1);
            const Vec3f dir(ij - origin);
            const Ray r(origin, dir);

            auto hit_point = r.get_closest_intersection_t(spheres);

            if (!hit_point) {
                frame_buffer[(j * WIDTH) + i] = color_util::BG_COLOR;
            } else {
                const auto& [sphere_hit, inter] = *hit_point;
                auto color = color_util::get_color_at(r, inter, light, sphere_hit, spheres);
                frame_buffer[(j * WIDTH) + i] = color;
            }
        }
    }
    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();
    std::cout << ">> time (not incl. writing to file):  " << end << "ms" << std::endl;

    start = steady_clock::now();
    std::cout << ">> Saving " << WIDTH << "x" << HEIGHT << " Image..." << std::endl;
    for (size_t i = 0; i < WIDTH * HEIGHT; i++) {
        mem_buffer.push_back(std::to_string((int)frame_buffer[i].x()) + " " +
                             std::to_string((int)frame_buffer[i].y()) + " " +
                             std::to_string((int)frame_buffer[i].z()));
    }
    const std::ostream_iterator<std::string> output_iterator(pbm_file, "\n");
    std::copy(mem_buffer.begin(), mem_buffer.end(), output_iterator);
    auto new_end = duration_cast<milliseconds>(steady_clock::now() - start).count();

    std::cout << ">> Image: \"img.ppm\" saved!" << std::endl;
    std::cout << ">> time spent writing to file:       " << new_end << "ms" << std::endl;
    std::cout << ">> time (all):                       " << end + new_end << "ms" << std::endl;

    return EXIT_SUCCESS;
}