#include "Vec3f.h"

class Light {
    private:
        float diffuse;
        float specular;
        float ambient;

        Vec3f position;
        Vec3f color;
    public:
        Light(Vec3f position, Vec3f color) : position(position), color(color) {}
        Vec3f get_position() const { return position; }
        Vec3f get_color() const { return color; }

        void set_light(float amb, float diff, float spec) {
            diffuse = diff;
            specular = spec;
            ambient = amb;
        }

        float get_diffuse() const { return diffuse; }
        float get_specular() const { return specular; }
        float get_ambient() const { return ambient; }
};