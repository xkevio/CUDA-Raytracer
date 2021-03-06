#ifndef __VEC3F_H__
#define __VEC3F_H__
class Vec3f {
   public:
    constexpr Vec3f(float x = 0, float y = 0, float z = 0) noexcept : x(x), y(y), z(z) {}

    Vec3f operator+(const Vec3f& a) const;
    Vec3f operator-(const Vec3f& a) const;
    Vec3f operator/(const float& a) const;
    Vec3f operator^(const Vec3f& a) const;  // cross product
    Vec3f operator&(const Vec3f& a) const;  // element wise multiplication
    friend Vec3f operator*(const float& a, const Vec3f& b);

    float length() const;
    Vec3f normalize() const;
    Vec3f scale(float factor) const;
    Vec3f cap(float max) const;

    void display();
    float x_() const;
    float y_() const;
    float z_() const;

   private:
    float x;
    float y;
    float z;
};
#endif