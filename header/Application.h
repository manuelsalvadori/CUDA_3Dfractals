#ifndef __APPLICATION__
#define __APPLICATION_

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cutil_math.h>

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

};

#endif /*__APPLICATION__*/_