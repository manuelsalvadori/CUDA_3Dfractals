#ifndef FRACT_H_
#define FRACT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include <memory>
#include <math.h> 
#include <cmath>
#include <cutil_math.h>
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics.hpp>

#include <Pixel.h>
#include <common.h>

#define WIDTH 512.0
#define HEIGHT 512.0
#define MAX_STEPS 128
#define EPSILON 1e-5
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define NUM_STREAMS 1
#define PIXEL_PER_STREAM (WIDTH / 1)

class Fract
{
public:
	Fract(int width, int height);
	virtual ~Fract();
	std::unique_ptr<sf::Image> generateFractal(const float3 &view, pixel *imageDevice, pixel *imageHost, float epsilon, cudaStream_t* streams);
	int getWidth() const;
	int getHeight() const;

private:
	int width;
	int height;
	int lastFrameStartTime{0};
	float rotation{ 0 };
	bool raymarch(sf::Vector3f rayOrigin, sf::Vector3f rayDirection);
	

};

////Global memory
//__constant__ sf::Vector3f* upDevice;
//__constant__ sf::Vector3f* rightDevice;

// Kernel functions
__global__ void distanceField(const float3 &view, pixel* img, float t, float epsilon, int2 streamID);
__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t, float epsilon);
__global__ void computeNormals(const float3 &view1, pixel* img, float t, float epsilon, int2 streamID);
__global__ void childKernel();
__device__ float sphereSolid(float3, float);

__device__ float cubeHollow(float3 ro, float3 b);

#endif /* FRACT_H_ */
