#ifndef SHAPES_H_
#define SHAPES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cutil_math.h>


__device__ float sphere(float3 ro, float r)
{
	return length(ro) - r;
}

__device__ float cube(float3 ro, float3 b)
{
	return length(fmaxf(fabs(ro) - b, float3{ 0.0f ,0.0f,0.0f }));
}

__device__ float torus(float3 p, float2 t)
{
	float2 q = { length(float2{ p.x,p.z }) -t.x, p.y };
	return length(q) - t.y;
}

__device__ float3 rotY(float3 p, float t)
{
	float3 r1 = { cosf(-1 * t), 0.0f, -1 * sinf(-1 * t) };
	float3 r2 = { 0.0f, 1.0f, 0.0f };
	float3 r3 = { sinf(-1 * t), 0.0f, cosf(-1 * t) };
	float3 r = { dot(r1,p), dot(r2,p) , dot(r3,p) };
	return r;
}

__device__ float shapeU(float d1, float d2)
{
	return fmin(d1, d2);
}
#endif /*SHAPES_H_*/