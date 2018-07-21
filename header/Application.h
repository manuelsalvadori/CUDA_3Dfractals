#ifndef __APPLICATION__
#define __APPLICATION__

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cutil_math.h>

// Standard
#include <iostream>
#include <fstream>

// SFML
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics.hpp>

// Project
#include <Bitmap.h>
#include <BitmapInfoHeader.h>
#include <BitmapFileHeader.h>
#include <common.h>
#include <sdf_util.hpp>
#include <type_useful.h>
#include <Fract.h>


class Application
{
public:
	Application();
	virtual ~Application();

	void startApplication();

private:
	float totalEnlapsedTime = 0.0f;

	void measureEnlapsedTime(const cudaEvent_t &start, const cudaEvent_t &stop);
	void computeFrame(int frameCounter, const cudaEvent_t &start, sf::RenderWindow &window, sf::Color &background, std::shared_ptr<sf::Image> &frame, Fract &fract, float3 &view, pixelRegionForStream * imgDevice[16], pixelRegionForStream * imageHost[16], cudaStream_t  stream[16], int peakClk, sf::Texture &texture, sf::Sprite &sprite, const cudaEvent_t &stop);
	void eventHandling(sf::RenderWindow &window);
	void saveFrame(int width, int height, std::shared_ptr<sf::Image> &frame, int frameCounter);
	void logPerformanceInfo(int frameNumber);
};

#endif /*__APPLICATION__*/_