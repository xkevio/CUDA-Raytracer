#ifndef __VEC3F_H__
#define __VEC3F_H__

#include <cmath>
#include <ostream>

class Vec3f {
   public:
    constexpr Vec3f(float x = 0, float y = 0, float z = 0) noexcept : mx(x), my(y), mz(z) {}

    Vec3f operator+(const Vec3f& a) const {
        return Vec3f(mx + a.mx, my + a.my, mz + a.mz);
    }
    Vec3f operator-(const Vec3f& a) const {
        return Vec3f(mx - a.mx, my - a.my, mz - a.mz);
    }
    Vec3f operator/(float a) const {
        return scale(1 / a);
    }
    Vec3f operator&(const Vec3f& a) const {
        return Vec3f(mx * a.mx, my * a.my, mz * a.mz);
    }  // element wise multiplication

    constexpr float length() const {
        return std::sqrt(mx*mx + my*my + mz*mz);
    }

    constexpr Vec3f normalize() const {
        return scale(1 / length());
    }

    constexpr Vec3f scale(float factor) const {
        return Vec3f(factor * mx, factor * my, factor * mz);
    }

    constexpr Vec3f cap(float max) const {
        float nx = mx;
        float ny = my;
        float nz = mz;
        if (mx > max) nx = max;
        if (my > max) ny = max;
        if (mz > max) nz = max;
        return Vec3f(nx, ny, nz);
    }

    [[nodiscard]] constexpr float x() const {
        return mx;
    }

    [[nodiscard]] constexpr float y() const {
        return my;
    }

    [[nodiscard]] constexpr float z() const {
        return mz;
    }

   private:
    float mx;
    float my;
    float mz;
};

constexpr Vec3f operator*(float a, const Vec3f& b) {
    return Vec3f(a * b.x(), a * b.y(), a * b.z());
}

constexpr Vec3f operator*(const Vec3f& b, float a) {
    return a * b;
}

constexpr Vec3f cross(const Vec3f& a, const Vec3f& b) {
    return Vec3f(a.y() * b.z() - a.z() * b.y(), 
                 a.z() * b.x() - a.x() * b.z(), 
                 a.x() * b.y() - a.y() * b.x());
}

std::ostream& operator<<(std::ostream& os, const Vec3f& a) {
    os << "(" << a.x() << " " << a.y() << " " << a.z() << ")";
    return os;
}

constexpr float dot(const Vec3f& a, const Vec3f& b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

#endif