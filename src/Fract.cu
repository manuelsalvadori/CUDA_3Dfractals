#include <Fract.h>
#include <Shapes.h>
#include <ctime>
#include <sdf_util.hpp>

Fract::Fract(int width, int height) : width(width), height(height) {}

Fract::~Fract() {}

int Fract::getWidth() const
{
	return width;
}

int Fract::getHeight() const
{
	return height;
}

std::unique_ptr<sf::Image> Fract::generateFractal(const float3 &view, pixel *imageDevice, pixel *imageHost, float epsilon, cudaStream_t* streams)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 dimGrid(width / (dimBlock.x*sqrt(NUM_STREAMS)), height / (dimBlock.y*sqrt(NUM_STREAMS)));

	float currentTime = clock();
	printf("Delta time: %fs\n", ((currentTime - lastFrameStartTime) / 1000));
	lastFrameStartTime = currentTime;
	rotation += 0.174533;


	// Start each kernel on a separate stream
	int2 streamID{ 0,0 };

	int pixelP = PIXEL_PER_STREAM;

	for (int i = 0; i < NUM_STREAMS; i++) {
		streamID.y = i % (width / (dimBlock.x*NUM_STREAMS));
		streamID.x = i / (width / (dimBlock.x*NUM_STREAMS));
		distanceField << <dimGrid, dimBlock >> > (view, imageDevice, rotation, epsilon, streamID);
	}
	CHECK(cudaDeviceSynchronize());

	printf("Tutti gli stream sono arrivati alla fine.\n");

	// Copy final img of the frame
	CHECK(cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost));

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			fract_ptr->setPixel(i, j, sf::Color(imageHost[width * j + i].r, imageHost[width * j + i].g, imageHost[width * j + i].b));
		}
	}

	return fract_ptr;
}

//bool Fract::raymarch(sf::Vector3f rayOrigin, sf::Vector3f rayDirection) {
//
//	float currentDistance = 0.0f;
//
//	for (int i = 0; i < MAX_STEPS; ++i) {
//
//		float extimatedDistanceClosestObject = sceneDistance(rayOrigin + rayDirection * currentDistance);
//
//		if (extimatedDistanceClosestObject < EPSILON) {
//
//			// Do something with p
//			return true;
//
//		}
//		currentDistance += extimatedDistanceClosestObject;
//	}
//	return false;
//}
//
//__global__ void drawTest() {
//	vec3 eye = vec3(0, 0, -1);
//	vec3 up = vec3(0, 1, 0);
//	vec3 right = vec3(1, 0, 0);
//
//	float u = gl_FragCoord.x * 2.0 / g_resolution.x - 1.0;
//	float v = gl_FragCoord.y * 2.0 / g_resolution.y - 1.0;
//	vec3 ro = right * u + up * v;
//	vec3 rd = normalize(cross(right, up));
//
//	vec4 color = vec4(0.0); // Sky color
//
//	float t = 0.0;
//	const int maxSteps = 32;
//	for (int i = 0; i < maxSteps; ++i)
//	{
//		vec3 p = ro + rd * t;
//		float d = length(p) - 0.5; // Distance to sphere of radius 0.5
//		if (d < g_rmEpsilon)
//		{
//			color = vec4(1.0); // Sphere color
//			break;
//		}
//
//		t += d;
//	}
//
//	return color;
//}
//
//
__global__ void distanceField(const float3 &view1, pixel* img, float t, float epsilon, int2 streamID)
{

	int idx = (PIXEL_PER_STREAM * streamID.x) + blockDim.x * blockIdx.x + threadIdx.x;
	int idy = (PIXEL_PER_STREAM * streamID.y) + blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;

	float3 view = { 0, 0, -10 };
	float3 up = { 0, 1, 0 };
	float3 right = { 1, 0, 0 };

	float u = 5 * (idx / WIDTH) - 2.5f;
	float v = 5 * (idy / HEIGHT) - 2.5f;
	float3 rayOrigin = { u, v, view.z };
	float3 rayDirection = { 0,0,1 };

	distanceExtimator(idx, idy, img, x, rayOrigin, rayDirection, t, epsilon);
}

__device__ void distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t, float epsilon)
{
	// Background color
	if (idx < WIDTH && idy < HEIGHT) {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

	float distanceTraveled = 0.0;
	const int maxSteps = MAX_STEPS;
	for (int i = 0; i < maxSteps; ++i)
	{
		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		//float distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
		//float power = abs(cos(t)) * 1;
		//float distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
		float distanceFromClosestObject = mandelbulb(rotY(iteratedPointPosition, t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
		//float distanceFromClosestObject = mengerBox(rotY(iteratedPointPosition, t), 3);
		//float distanceFromClosestObject = sdfSphere(rotY(iteratedPointPosition, t), power);

		// Far plane 
		if (distanceTraveled > 15.0f)
			break;

		if (idx < WIDTH && idy < HEIGHT  && distanceFromClosestObject < epsilon)
		{
			// Sphere color
			img[x].r = 255 - ((i * 255) / MAX_STEPS);
			img[x].g = 255 - ((i * 255) / MAX_STEPS);
			img[x].b = 255 - ((i * 255) / MAX_STEPS);
			break;
		}

		distanceTraveled += distanceFromClosestObject;

		if (isnan(distanceTraveled))
			distanceTraveled = 0.0f;
	}
}

__global__ void childKernel()
{
	printf("Sono un figlio.\n");
}