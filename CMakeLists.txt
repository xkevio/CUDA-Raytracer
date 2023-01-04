cmake_minimum_required(VERSION 3.20)
project(CUDA_Raytracer)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)

add_executable(CUDA_Raytracer
        cpu_version/main.cpp
        cpu_version/Color.cpp
        cpu_version/Ray.cpp
        cpu_version/Sphere.cpp)

target_link_libraries(CUDA_Raytracer PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(CUDA_Raytracer PRIVATE cpu_version/inc)
target_compile_options(CUDA_Raytracer PRIVATE -Wall -Wextra -Wpedantic -Werror)
