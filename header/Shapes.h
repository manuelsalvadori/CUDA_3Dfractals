#ifndef SHAPES_H_
#define SHAPES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cutil_math.h>


__device__ float sphereSolid(float3 ro, float r)
{
	return (length(ro) - r);
}

__device__ float cubeHollow(float3 ro, float3 b)
{
	return length(fmaxf(fabs(ro) - b, float3{ 0.0f ,0.0f,0.0f }));
}

__device__ float cubeSolid(float3 ro, float3 b)
{
	float3 d = fabs(ro) - b;
	return min(max(d.x, max(d.y, d.z)), 0.0) + length(fmaxf(d, float3{0.0f,0.0f,0.0f}));
}

__device__ float torusSolid(float3 p, float2 t)
{
	float2 q = { length(float2{ p.x,p.z }) - t.x, p.y };
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

__device__ float shapeUnion(float d1, float d2)
{
	return fminf(d1, d2);
}

// The variable d2 is the one "remaining", d1 is the one beeing "deleted"
__device__ float shapeSubtraction(float d1, float d2)
{
	return fmaxf(-1 * d1, d2);
}

__device__ float crossCubeHollow(float3 p, float3 b)
{
	float da = cubeHollow(p, float3{ (b.x + 0.5f), (b.y / 3.0f), (b.z / 3.0f) });
	float db = cubeHollow(float3{ p.y,p.z,p.x }, float3{ (b.x / 3.0f), (b.y + 0.5f), (b.z / 3.0f) });
	float dc = cubeHollow(float3{ p.z,p.x,p.y }, float3{ (b.x / 3.0f), (b.y / 3.0f), (b.z + 0.5f) });
	return min(da, min(db, dc));
}

__device__ float crossCubeSolid(float3 p, float3 b)
{
	float da = cubeSolid(p, float3{ (b.x + 0.5f), (b.y / 3.0f), (b.z / 3.0f) });
	float db = cubeSolid(float3{ p.y,p.z,p.x }, float3{ (b.x / 3.0f), (b.y + 0.5f), (b.z / 3.0f) });
	float dc = cubeSolid(float3{ p.z,p.x,p.y }, float3{ (b.x / 3.0f), (b.y / 3.0f), (b.z + 0.5f) });
	return min(da, min(db, dc));
}
#endif /*SHAPES_H_*/