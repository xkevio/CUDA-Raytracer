#include "Vec3f.h"

class Light {
   private:
    float diffuse = 0;
    float specular = 0;
    float ambient = 0;

    Vec3f position;
    Vec3f color;

   public:
    constexpr Light(const Vec3f& position, const Vec3f& color) : position(position), color(color) {}
    Vec3f get_position() const { 
        return position; 
    }
    Vec3f get_color() const { 
        return color; 
    }

    void set_light(float amb, float diff, float spec) {
        ambient = amb;
        diffuse = diff;
        specular = spec;
    }

    float get_diffuse() const { 
        return diffuse; 
    }
    float get_specular() const { 
        return specular; 
    }
    float get_ambient() const { 
        return ambient; 
    }
};