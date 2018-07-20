#ifndef FRACT_H_
#define FRACT_H_

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cutil_math.h>

// Standard
#include <iostream>
#include <memory>
#include <math.h> 
#include <cmath>
#include <ctime>

// SFML
#include <SFML/Graphics.hpp>

// Project
#include <common.h>
#include <sdf_util.hpp>
#include <type_useful.h>
#include <Shapes.h>

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


};


// CUDA functions
// Kernels
__global__ void rayMarching(const float3 &view1, pixel* img, float time, int2 streamID, int peakClk);
__global__ void childKernel();

// Device only
__device__ infoEstimatorResult distanceEstimator(const float3 &iteratedPointPosition, float time);
__device__ float softShadow(float3 origin, float3 direction, float time);
__device__ float hardShadow(float3 origin, float3 direction, float time);

#endif /* FRACT_H_ */
