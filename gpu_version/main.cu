#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "cuda_util.h"
#include "Vec3f.h"
#include "Light.h"

const int WIDTH = 2048;
const int HEIGHT = 2048;
const int OBJ_COUNT = 4;

//const int MAX_THREADS_PER_BLOCK = 1024;

using Color = Vec3f;
using namespace std::chrono;

struct Sphere {
    float radius;
    Vec3f center;
    Vec3f color;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(float r, const Vec3f &c, const Vec3f &col) : radius(r), center(c), color(col) {}
    __host__ __device__ Vec3f get_normal_at(const Vec3f &at) const;
};

struct Ray {
    Vec3f origin;
    Vec3f dir;

    __host__ __device__ Ray(const Vec3f &o, const Vec3f &d) : origin(o), dir(d) {}

    __host__ __device__ Vec3f at(float t) const;
    __host__ __device__ float has_intersection(const Sphere &sphere) const;
};

__host__ __device__ Vec3f Sphere::get_normal_at(const Vec3f &at) const {
    return Vec3f(at - center).normalize();
}

__host__ __device__ Vec3f Ray::at(float t) const {
    return origin + (t * dir); 
}

__host__ __device__ float dot(const Vec3f &a, const Vec3f &b) {
    return a.x_() * b.x_() + a.y_() * b.y_() + a.z_() * b.z_();
}

__host__ __device__ float Ray::has_intersection(const Sphere &sphere) const {
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

__device__ constexpr float f_max(float a, float b) {
    return a > b ? a : b;
}

__device__ Color convert_to_color(const Vec3f &v) {
    return Color(static_cast<int> (1 * ((v.x_()) * 255.999)), static_cast<int> (1 * ((v.y_()) * 255.999)), static_cast<int> (1 * ((v.z_()) * 255.999)));
}

__device__ int get_closest_intersection(Sphere* spheres, const Ray& r, float* intersections) {
    int hp = -1;
    for(int ii = 0; ii < OBJ_COUNT; ii++) {
        intersections[ii] = r.has_intersection(spheres[ii]);
    }

    int asize = OBJ_COUNT;
    if(asize == 1) {
        hp = intersections[0] < 0 ? -1 : 0;
    } else {
        if(asize != 0) {
            float min_val = 100.0;
            for (int ii = 0; ii < asize; ii++) {
                if (intersections[ii] < 0.0) continue;
                else if (intersections[ii] < min_val) {
                    min_val = intersections[ii];
                    hp = ii;
                }
            }
        }
    }
    return hp;
}

__device__ Color get_color_at(const Ray &r, float intersection, Light* light, const Sphere &sphere, Sphere* spheres, Vec3f* origin) {
    float shadow = 1;

    Vec3f normal = sphere.get_normal_at(r.at(intersection));

    Vec3f to_camera(*origin - r.at(intersection));
    to_camera = to_camera.normalize();

    Vec3f light_ray(light->get_position() - r.at(intersection));
    light_ray = light_ray.normalize();

    Vec3f reflection_ray = (-1 * light_ray) - 2 * dot((-1 * light_ray), normal) * normal;
    reflection_ray = reflection_ray.normalize();

    Ray rr(r.at(intersection) + 0.001 * normal, reflection_ray);
    float intersections[OBJ_COUNT];
    int hp = get_closest_intersection(spheres, rr, intersections);
    bool reflect = false;
    float reflect_shadow = 1;
    if(hp != -1) {
        reflect = true;
        Ray rs(rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])), light->get_position() - rr.at(intersections[hp]) + 0.001 * spheres[hp].get_normal_at(rr.at(intersections[hp])));
        for(int i = 0; i < OBJ_COUNT; i++) {
            if(rs.has_intersection(spheres[i]) > 0.000001f) reflect_shadow = 0.35;
        }
    }

    auto ambient = light->get_ambient() * light->get_color(); 
    auto diffuse = (light->get_diffuse() * f_max(dot(light_ray, normal), 0.0f)) * light->get_color();
    auto specular = light->get_specular() * pow(f_max(dot(reflection_ray, to_camera), 0.0f), 32) * light->get_color();

    Ray shadow_ray(r.at(intersection) + (0.001f * normal), light->get_position() - (r.at(intersection) + 0.001f * normal));
    for(int i = 0; i < OBJ_COUNT; i++) {
        if(shadow_ray.has_intersection(spheres[i]) > 0.000001f) shadow = 0.35;
    }

    auto all_light = reflect ? (ambient + diffuse + specular).cap(1) & (0.55 * (sphere.color - (reflect_shadow * spheres[hp].color)) + (reflect_shadow * spheres[hp].color)).cap(1) : (ambient + diffuse + specular).cap(1) & sphere.color;
    return convert_to_color(shadow * all_light);
}

__global__ void cast_ray(Vec3f* fb, Sphere* spheres, Light* light, Vec3f* origin) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int tid = (j*WIDTH) + i;
    if(i >= WIDTH || j >= HEIGHT) return;

    Vec3f ij(2 * (float((i) + 0.5) / (WIDTH - 1)) - 1, 1 - 2 * (float((j) + 0.5) / (HEIGHT - 1)), -1);
    Vec3f dir(ij - *origin);
    Ray r(*origin, dir);

    float intersections[OBJ_COUNT];
    int hp = get_closest_intersection(spheres, r, intersections);

    if(hp == -1) {
        fb[tid] = Color(94, 156, 255);
    } else {
        auto color = get_color_at(r, intersections[hp], light, spheres[hp], spheres, origin);
        fb[tid] = color;
    }
}

void initDevice(int& device_handle) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printDeviceProps(devProp);
  
    cudaSetDevice(device_handle);
}

void run_kernel(const int size, Vec3f* fb, Sphere* spheres, Light* light, Vec3f* origin) {
    Vec3f* fb_device = nullptr;
    Sphere* spheres_dv = nullptr;
    Light* light_dv = nullptr;
    Vec3f* origin_dv = nullptr;

    checkErrorsCuda(cudaMalloc((void**) &fb_device, sizeof(Vec3f) * size));
    checkErrorsCuda(cudaMemcpy((void*) fb_device, fb, sizeof(Vec3f) * size, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &spheres_dv, sizeof(Sphere) * OBJ_COUNT));
    checkErrorsCuda(cudaMemcpy((void*) spheres_dv, spheres, sizeof(Sphere) * OBJ_COUNT, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &light_dv, sizeof(Light) * 1));
    checkErrorsCuda(cudaMemcpy((void*) light_dv, light, sizeof(Light) * 1, cudaMemcpyHostToDevice));

    checkErrorsCuda(cudaMalloc((void**) &origin_dv, sizeof(Vec3f) * 1));
    checkErrorsCuda(cudaMemcpy((void*) origin_dv, origin, sizeof(Vec3f) * 1, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 blocks(WIDTH / 16, HEIGHT / 16);
    cast_ray<<<blocks, dim3(16, 16)>>>(fb_device, spheres_dv, light_dv, origin_dv);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(">> time for kernel: %f ms\n", time);

    checkErrorsCuda(cudaMemcpy(fb, fb_device, sizeof(Vec3f) * size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkErrorsCuda(cudaFree(fb_device));
    checkErrorsCuda(cudaFree(spheres_dv));
    checkErrorsCuda(cudaFree(light_dv));
    checkErrorsCuda(cudaFree(origin_dv));
}

int main(int, char**) {
    std::ofstream file("img.ppm");
    file << "P3" << "\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";

    const int n = WIDTH * HEIGHT;
    int device_handle = 0;

    Vec3f* frame_buffer = new Vec3f[n];
    std::vector<std::string> mem_buffer;

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        std::cerr << "initDevice(): No CUDA Device found." << std::endl;
        return EXIT_FAILURE;
    }
    initDevice(device_handle);

    Sphere sphere(.25, Vec3f(-0.35, -0.35, -1), Color((float) 235 / 255, (float) 64 / 255, (float) 52 / 255));
    Sphere sphere2(.25, Vec3f(0.35, -0.35, -1), Color((float) 52 / 255, (float) 198 / 255, (float) 235 / 255));
    Sphere sphere3(.25, Vec3f(0, 0.35, -1), Color(0, 1, 0));
    Sphere sphere4(1000, Vec3f(0, -1002, 0), Color(0.5, 0.5, 0.5));

    Vec3f *origin = new Vec3f(0, 0, 1);

    Sphere *spheres = new Sphere[OBJ_COUNT] {sphere, sphere2, sphere3, sphere4};

    Light *light = new Light(Vec3f(1, 1, 1), Vec3f(1, 1, 1));
    light->set_light(.2, .5, .5);

    std::cout << "===========================================" << std::endl;
    std::cout << ">> Starting kernel for " << WIDTH << "x" << HEIGHT << " image..." << std::endl;
    run_kernel(n, frame_buffer, spheres, light, origin);
    std::cout << ">> Finished kernel" << std::endl;

    auto start = steady_clock::now();
    std::cout << ">> Saving Image..." << std::endl;

    for (size_t i = 0; i < n; i++) {
        mem_buffer.push_back(std::to_string((int) frame_buffer[i].x_()) + " " + std::to_string((int) frame_buffer[i].y_()) + " " + std::to_string((int) frame_buffer[i].z_()));
    }
    std::ostream_iterator<std::string> output_iterator(file, "\n");
    std::copy(mem_buffer.begin(), mem_buffer.end(), output_iterator);

    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();
    std::cout << ">> Finished writing to file in " << end << " ms" << std::endl;
    std::cout << "===========================================" << std::endl;

    file.close();
    delete[] frame_buffer;
    delete origin;
    delete light;
    delete[] spheres;

    return EXIT_SUCCESS;
}