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
	return min(max(d.x, max(d.y, d.z)), 0.0) + length(fmaxf(d, float3{ 0.0f,0.0f,0.0f }));
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

__device__ float mengerSponge(float3 p)
{
	float d = cubeSolid(p, float3{ 0.5f,0.5f,0.5f });

	float s = 1.0;
	for (int m = 0; m < 3; m++)
	{
		int3 b = int3{ (int)p.x * s, (int)p.y * s,(int)p.z * s };
		float3 a = float3{ b.x % 2, b.y % 2 ,b.z % 2 } -1.0;
		s *= 3.0;
		float3 r = fabs(float3{ 1.0f,1.0,1.0f } -3.0*fabs(a));

		float da = max(r.x, r.y);
		float db = max(r.y, r.z);
		float dc = max(r.z, r.x);
		float c = (min(da, min(db, dc)) - 1.0) / s;

		d = max(d, c);
	}

	return d;
}


__device__ float sierpinskiPyramidNotOpt(float3 z, int iteration = 3, float scale = 1.0f, float offset = 0.0f)
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

__device__ float sierpinskiPyramidOpt(float3 z, int iteration = 3, float scale = 1.0f, float offset = 0.0f)
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
	return (length(z)) * pow(scale, (float)(-1*n));
}

__device__ float sierpinskiPyramidTest(float3 z, int iteration = 3, float scale = 1.0f, float offset = 0.0f)
{
	/*float3 c = float3{ 0.0f,0.0f ,0.0f };
	int n = 0;
	float dist = 0;
	float d = 0;

	while (n < iteration) {
		for (int i = 0; i < vertices.length(); i++) {
			d = length(z - vertices[i]);
			if (i == 0 || d < dist) { c = vertices[i]; dist = d; }
		}
		z = scale * z - c * (scale - 1.0);
		float 	r = dot(z, z);
		n++;
	}

	return length(z) * pow(scale, (float)-n);*/

}
#endif /*SHAPES_H_*/