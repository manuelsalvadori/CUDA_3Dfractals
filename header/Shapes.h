#ifndef SHAPES_H_
#define SHAPES_H_

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cutil_math.h>

inline __host__ __device__ float sphereSolid(float3 ro, float r)
{
	return (length(ro) - r);
}

inline __host__ __device__ float cubeHollow(float3 ro, float3 b)
{
	return length(fmaxf(fabs(ro) - b, float3{ 0.0f ,0.0f,0.0f }));
}

inline __host__ __device__ float cubeSolid(float3 ro, float3 b)
{
	float3 d = fabs(ro) - b;
	return min(max(d.x, max(d.y, d.z)), 0.0) + length(fmaxf(d, float3{ 0.0f,0.0f,0.0f }));
}

inline __host__ __device__ float torusSolid(float3 p, float2 t)
{
	float2 q = { length(float2{ p.x,p.z }) - t.x, p.y };
	return length(q) - t.y;
}

inline __host__ __device__ float3 rotY(float3 p, float t)
{
	float3 r1 = { cosf(-1 * t), 0.0f, -1 * sinf(-1 * t) };
	float3 r2 = { 0.0f, 1.0f, 0.0f };
	float3 r3 = { sinf(-1 * t), 0.0f, cosf(-1 * t) };
	float3 r = { dot(r1,p), dot(r2,p) , dot(r3,p) };
	return r;
}

inline __host__ __device__ float shapeUnion(float d1, float d2)
{
	return fminf(d1, d2);
}

inline __host__ __device__ infoEstimatorResult shapeUnion(infoEstimatorResult d1, infoEstimatorResult d2)
{
	return d1.distance < d2.distance ? d1 : d2;
}

// d2 is the shape from wich we are subtracting, d1 is the one beeing subtracted
inline __host__ __device__ float shapeSubtraction(float d1, float d2)
{
	return fmaxf(-1 * d1, d2);
}

inline __host__ __device__ float crossCubeHollow(float3 p, float3 b)
{
	float da = cubeHollow(p, float3{ (b.x + 0.5f), (b.y / 3.0f), (b.z / 3.0f) });
	float db = cubeHollow(float3{ p.y,p.z,p.x }, float3{ (b.x / 3.0f), (b.y + 0.5f), (b.z / 3.0f) });
	float dc = cubeHollow(float3{ p.z,p.x,p.y }, float3{ (b.x / 3.0f), (b.y / 3.0f), (b.z + 0.5f) });
	return min(da, min(db, dc));
}

inline __host__ __device__ float crossCubeSolid(float3 p, float3 b)
{
	float da = cubeSolid(p, float3{ (b.x + 0.5f), (b.y / 3.0f), (b.z / 3.0f) });
	float db = cubeSolid(float3{ p.y,p.z,p.x }, float3{ (b.x / 3.0f), (b.y + 0.5f), (b.z / 3.0f) });
	float dc = cubeSolid(float3{ p.z,p.x,p.y }, float3{ (b.x / 3.0f), (b.y / 3.0f), (b.z + 0.5f) });
	return min(da, min(db, dc));
}

inline __host__ __device__ float sierpinskiPyramidNotOpt(float3 z, int iteration = 3, float scale = 1.0f, float offset = 0.0f)
{
	float3 a1{ 1.0f, 1.0f, 1.0f };
	float3 a2{ -1.0f, -1.0f, 1.0f };
	float3 a3{ 1.0f, -1.0f, -1.0f };
	float3 a4{ -1.0f, 1.0f, -1.0f };
	float3 c;
	int n = 0;
	float dist, d;
	while (n < iteration) {
		c = a1;

		dist = length(z - a1);
		d = length(z - a2);
		if (d < dist)

			c = a2; dist = d;

		d = length(z - a3);
		if (d < dist)

			c = a3; dist = d;

		d = length(z - a4);
		if (d < dist)

			c = a4; dist = d;

		z = scale * z - c * (scale - 1.0);
		n++;
	}

	return length(z) * pow(scale, (float)-n);
}

inline __host__ __device__ float sierpinskiPyramidOpt(float3 z, int iteration = 3, float scale = 1.0f, float offset = 0.0f)
{
	int n = 0;
	while (n < iteration) {
		if (z.x + z.y < 0)
		{
			z.x = -z.y; // fold 1
			z.y = -z.x; // fold 1
		}
		if (z.x + z.z < 0)
		{
			z.x = -z.z; // fold 2
			z.z = -z.x; // fold 2
		}
		if (z.y + z.z < 0)
		{
			z.z = -z.y; // fold 3
			z.y = -z.z; // fold 3
		}
		z = z * scale - offset * (scale - 1.0);
		n++;
	}
	return (length(z)) * pow(scale, (float)(-1 * n));
}

#endif /*SHAPES_H_*/