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
#include <chrono>

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

	void runApplication();

private:
	double totalEnlapsedTime = 0.0f;
	bool isStreamNotBlocking = true;

	void measureEnlapsedTime(const cudaEvent_t & startParallel, const cudaEvent_t & stopParallel, std::chrono::high_resolution_clock::time_point startSequential, std::chrono::high_resolution_clock::time_point stopSequential);
	void computeFrame(int frameCounter, sf::RenderWindow &window, sf::Color &background, std::shared_ptr<sf::Image> &frame, Fract &fract, float3 &view, pixelRegionForStream * imgDevice[NUM_STREAMS], pixelRegionForStream * imageHost[NUM_STREAMS], cudaStream_t  stream[NUM_STREAMS], int peakClk, sf::Texture &texture, sf::Sprite &sprite);
	void eventHandling(sf::RenderWindow &window);
	void saveFrame(int width, int height, std::shared_ptr<sf::Image> &frame, int frameCounter);
	void logPerformanceInfo(int frameNumber);
	void cleanupMemory(pixelRegionForStream * imageHost[NUM_STREAMS], pixelRegionForStream * imgDevice[NUM_STREAMS], cudaStream_t  stream[NUM_STREAMS]);
};

#endif /*__APPLICATION__*/_