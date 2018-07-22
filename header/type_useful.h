#ifndef __TYPE_USEFUL__
#define __TYPE_USEFUL__

#include <cstdint>
#include <cutil_math.h>

// Defines for constant values
#define PARALLEL				true
#define WIDTH					512.0
#define HEIGHT					512.0
#define MAX_STEPS				128
#define EPSILON					0.01f
#define BLOCK_DIM_X				8
#define BLOCK_DIM_Y				8
#define NUM_STREAMS				1
#define PIXEL_PER_STREAM_X		(int)(WIDTH /1)
#define PIXEL_PER_STREAM_Y		(int)(HEIGHT/1)
#define PIXEL_PER_STREAM		(int)((WIDTH/1)*(HEIGHT/1))
#define USE_MASK				false
#define MASK_SIZE				7
#define MASK_PERCENTAGE			0.5f
#define FAR_PLANE				100.0f
#define MAX_NUMBER_OF_FRAMES	720

// Color of a pixel
struct pixel
{
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

// Information about distance extimator function
struct infoEstimatorResult {
	float distance;
	float3 color;
};

typedef pixel pixelRegionForStream[PIXEL_PER_STREAM];



#endif /*__TYPE_USEFUL__*/
