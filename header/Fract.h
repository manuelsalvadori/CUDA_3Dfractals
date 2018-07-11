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

#include <ctime>
#include <sdf_util.hpp>

#define WIDTH 512.0
#define HEIGHT 512.0
#define MAX_STEPS 128
#define EPSILON 0.00005f
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define NUM_STREAMS 16
#define PIXEL_PER_STREAM_X (int)(WIDTH / 4)
#define PIXEL_PER_STREAM_Y (int)(HEIGHT / 4)
#define PIXEL_PER_STREAM (int)((WIDTH / 4)*(HEIGHT / 4))

typedef pixel pixelRegionForStream[PIXEL_PER_STREAM];

class Fract
{
public:
	Fract(int width, int height);
	virtual ~Fract();
	std::unique_ptr<sf::Image> generateFractal(const float3 &view, pixelRegionForStream* imageDevice, pixelRegionForStream * imageHost, cudaStream_t* streams, int peakClk);
	int getWidth() const;
	int getHeight() const;

private:
	int width;
	int height;
	float rotation{ 0 };
	bool raymarch(sf::Vector3f rayOrigin, sf::Vector3f rayDirection);


};

////Global memory
//__constant__ sf::Vector3f* upDevice;
//__constant__ sf::Vector3f* rightDevice;

// Kernel functions
__global__ void distanceField(const float3 &view, pixel* img, float t, int2 streamID);
__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t);
__device__ float DE(const float3 &iteratedPointPosition, float t);
__global__ void computeNormals(const float3 &view1, pixel* img, float t, int2 streamID, int peakClk);
__global__ void childKernel();
__device__ float sphereSolid(float3, float);

__device__ float cubeHollow(float3 ro, float3 b);

#endif /* FRACT_H_ */
