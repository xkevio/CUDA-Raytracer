#ifndef LIGHT_HPP
#define LIGHT_HPP

#include "Vec3f.hpp"

class Light {
   private:
    float diffuse = 0;
    float specular = 0;
    float ambient = 0;

    Vec3f position;
    Vec3f color;

   public:
    constexpr Light(const Vec3f& position, const Vec3f& color) : position(position), color(color) {}
    [[nodiscard]] Vec3f get_position() const {
        return position; 
    }
    [[nodiscard]] Vec3f get_color() const {
        return color; 
    }

    void set_light(float amb, float diff, float spec) {
        ambient = amb;
        diffuse = diff;
        specular = spec;
    }

    [[nodiscard]] float get_diffuse() const {
        return diffuse; 
    }
    [[nodiscard]] float get_specular() const {
        return specular; 
    }
    [[nodiscard]] float get_ambient() const {
        return ambient; 
    }
};

#endif