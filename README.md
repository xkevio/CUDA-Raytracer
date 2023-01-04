# CUDA-Raytracer
A simple ray tracer written with CUDA that saves its output in a .ppm file, CPU version included for reference. Below, you can find the rendered image and a performance analysis.

This was made as a final project for the GPU programming course at Otto-von-Guericke University Magdeburg.

## The Render
The code currently generates the following image, though it does support more spheres and light sources:

![3 spheres](images/img.png)

## Performance Analysis
### CPU version (serial vs parallel):
> Note: In the diagrams I used the german spelling of "serial"

![serial](images/CPU%20\(seriell,%20with%20-O2\)%20@%204%20Spheres.png)

![parallel](images/CPU%20\(OpenMP,%20with%20-O2\)%20@%204%20Spheres.png)

### GPU version:
![gpu4spheres](images/GPU%20@%204%20Spheres.png)

> Direct comparison with the parallel CPU version:

![cpuvsgpu](images/CPU%20vs%20GPU%20\(4%20Spheres\).png)

### CPU vs GPU Direct comparison with more than 4 spheres:
> Note: The time axis is shown on a logarithmic scale

![all](images/GPU%20@%202048x2048%20und%20CPU%20\(OpenMP,%20-O2\)%20@%202048x2048.png)
