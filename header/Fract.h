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

	int getWidth() const;
	int getHeight() const;
	std::unique_ptr<sf::Image> generateFractal(const float3 &view, pixelRegionForStream* imageDevice, pixelRegionForStream * imageHost, cudaStream_t* streams, int peakClk);

private:
	int width;
	int height;
	float rotation{ 0 };
	void fillImgWindow(pixelRegionForStream * imageHost, int2 &streamID, std::unique_ptr<sf::Image> &fract_ptr);
	void rayMarchingSequential(pixel* img, int2 coordinates, float time);


};


// CUDA 
// Kernels
__global__ void rayMarching(pixel* img, float time, int2 streamID, int peakClk);
__global__ void childKernel();

// Device only
__host__ __device__ infoEstimatorResult distanceEstimator(const float3 &iteratedPointPosition, float time);
__host__ __device__ void transformationOnPoint(float3 &modifiedIteratedPosition, float time);
__host__ __device__ float softShadow(float3 origin, float3 direction, float time);
__host__ __device__ float hardShadow(float3 origin, float3 direction, float time);
__host__ __device__ void meanOptimization(int globalCounter, int3  blockResults[14][14], int2 &sharedId, bool &hitOk, int &retflag);
__host__ __device__ void computeNormals(const float3 &iteratedPointPosition, float time, float3 &normal, float3 &rayDirection);

#endif /* FRACT_H_ */
