#ifndef __VEC3F_H__
#define __VEC3F_H__
class Vec3f {
   public:
    __host__ __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3f operator+(const Vec3f& a) const;
    __host__ __device__ Vec3f operator-(const Vec3f& a) const;
    __host__ __device__ Vec3f operator/(const float& a) const;
    __host__ __device__ Vec3f operator^(const Vec3f& a) const;  // cross product
    __host__ __device__ Vec3f operator&(const Vec3f& a) const;  // element wise multiplication
    __host__ __device__ friend Vec3f operator*(const float& a, const Vec3f& b);

    __host__ __device__ float length() const;
    __host__ __device__ Vec3f normalize() const;
    __host__ __device__ Vec3f scale(float factor) const;
    __host__ __device__ Vec3f cap(float max) const;

    __host__ __device__ void display();
    __host__ __device__ float x_() const;
    __host__ __device__ float y_() const;
    __host__ __device__ float z_() const;

   private:
    float x;
    float y;
    float z;
};
#endif