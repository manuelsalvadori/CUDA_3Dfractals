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
#define MAX_STEPS 1000
#define EPSILON 0.001

class Fract
{
public:
	Fract(int width, int height);
	virtual ~Fract();
	std::unique_ptr<sf::Image> generateFractal(const float3 &view, pixel *imageDevice, pixel *imageHost);
	int getWidth() const;
	int getHeight() const;

private:
	int width;
	int height;
	bool raymarch(sf::Vector3f rayOrigin, sf::Vector3f rayDirection);
	

};

////Global memory
//__constant__ sf::Vector3f* upDevice;
//__constant__ sf::Vector3f* rightDevice;

// Kernel functions
__global__ void pixelCalculationKernel(const float3 &view, pixel* img);
__global__ void distanceFieldKernel(float3 rayOrigin, float3 rayDirection, int idx, int idy, int x);
__global__ void test(pixel *);

#endif /* FRACT_H_ */
