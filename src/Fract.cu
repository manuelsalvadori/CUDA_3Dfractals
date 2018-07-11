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

std::unique_ptr<sf::Image> Fract::generateFractal(const float3 &view, pixelRegionForStream* imageDevice, pixelRegionForStream * imageHost, cudaStream_t* streams, int peakClk)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 dimGrid(width / (BLOCK_DIM_X*sqrt(NUM_STREAMS)), height / (BLOCK_DIM_Y*sqrt(NUM_STREAMS)));

	// Start each kernel on a separate stream
	int2 streamID{ 0,0 };

	for (int i = 0; i < NUM_STREAMS; i++) {
		streamID.x = i / (width / (BLOCK_DIM_Y*NUM_STREAMS));
		streamID.y = i % (width / (BLOCK_DIM_X*NUM_STREAMS));
		computeNormals << <dimGrid, dimBlock, 0, streams[i] >> > (view, imageDevice[i], rotation, streamID, peakClk);
		cudaMemcpyAsync(&imageHost[i], &imageDevice[i], sizeof(pixelRegionForStream), cudaMemcpyDeviceToHost, streams[i]);
	}

	CHECK(cudaDeviceSynchronize());

	printf("Tutti gli stream sono arrivati alla fine.\n");

	// Copy final img of the frame
	//CHECK(cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost));

	// Fill the window with img
	for (int streamNumber = 0; streamNumber < width/*NUM_STREAMS*/; streamNumber++)
	{
		for (int arrayIndex = 0; arrayIndex < height/*PIXEL_PER_STREAM_X*PIXEL_PER_STREAM_Y*/; arrayIndex++)
		{
			int streamX = arrayIndex / PIXEL_PER_STREAM_Y;
			int streamY = arrayIndex % (int)PIXEL_PER_STREAM_Y;
			//fract_ptr->setPixel(i, j, sf::Color(imageHost[width * j + i].r, imageHost[width * j + i].g, imageHost[width * j + i].b));
			fract_ptr->setPixel(streamNumber, arrayIndex, sf::Color(255, 0, 0));
		}
	}

	return fract_ptr;
}

__global__ void distanceField(const float3 &view1, pixel* img, float t, int2 streamID)
{

	int idx = (PIXEL_PER_STREAM_X * streamID.x) + blockDim.x * blockIdx.x + threadIdx.x;
	int idy = (PIXEL_PER_STREAM_Y * streamID.y) + blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;

	float3 view{ 0, 0, -10 };
	float3 forward{ 0, 0, 1 };
	float3 up{ 0, 1, 0 };
	float3 right{ 1, 0, 0 };

	float u = 5 * (idx / WIDTH) - 2.5f;
	float v = 5 * (idy / HEIGHT) - 2.5f;

	float3 point{ u, v,0 };

	float3 rayOrigin = { 0, 0, view.z };
	float3 rayDirection = normalize(point - rayOrigin);

	distanceExtimator(idx, idy, img, x, rayOrigin, rayDirection, t);
}

__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t)
{
	// Background color
	if (idx < WIDTH && idy < HEIGHT) {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

	float distanceTraveled = 0.0;
	const int maxSteps = MAX_STEPS;
	float distanceFromClosestObject = 0;
	for (int i = 0; i < maxSteps; ++i)
	{
		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		distanceFromClosestObject = DE(iteratedPointPosition, t);


		// Far plane 
		if (distanceTraveled > 15.0f)
			break;

		if (idx < WIDTH && idy < HEIGHT  && distanceFromClosestObject < EPSILON)
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

	return distanceFromClosestObject;
}

__device__ float DE(const float3 &iteratedPointPosition, float t)
{
	//return distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
	//return power = abs(cos(t)) * 40 + 2;
	//return distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
	return mandelbulb(rotY(iteratedPointPosition, t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	//float mBox = mengerBox(rotY(iteratedPointPosition, t), 3);
	//float sphere = sdfSphere(iteratedPointPosition + float3{ 2.0f,2.0f,0.0f }, 0.5f);
	//return mBox;

}

__global__ void computeNormals(const float3 &view1, pixel* img, float t, int2 streamID, int peakClk)
{

	int idx = (PIXEL_PER_STREAM_X * streamID.x) + blockDim.x * blockIdx.x + threadIdx.x;
	int idy = (PIXEL_PER_STREAM_Y * streamID.y) + blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;

	float3 view{ 0, 0, -10 };
	float3 forward{ 0, 0, 1 };
	float3 up{ 0, 1, 0 };
	float3 right{ 1, 0, 0 };

	float u = 5 * (idx / WIDTH) - 2.5f;
	float v = 5 * (idy / HEIGHT) - 2.5f;

	float3 point{ u, v,0 };

	float3 rayOrigin = { 0, 0, view.z };
	float3 rayDirection = normalize(point - rayOrigin);

	float3 lightPosition{ 1.0f,1.0f,view.z };
	float3 lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);
	float3 lightColor = normalize(float3{ 66.0f,1340.f,2440.f });

	float3 halfVector = normalize(-lightDirection - rayDirection);

	// Background color
	if (idx < WIDTH && idy < HEIGHT) {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

	// Clock extimate
	long long int startForTimer = clock64();

	float distanceTraveled = 0.0;
	const int maxSteps = MAX_STEPS;
	float distanceFromClosestObject = 0;
	for (int i = 0; i < maxSteps; ++i)
	{
		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		distanceFromClosestObject = DE(iteratedPointPosition, t);

		lightPosition = float3{ 1.0f,1.0f,view.z };
		lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);

		// Far plane 
		if (distanceTraveled > 100.0f)
			break;

		if (idx < WIDTH && idy < HEIGHT  && distanceFromClosestObject < EPSILON)
		{

			float3 xDir{ 0.5773*EPSILON,0.0f,0.0f };
			float3 yDir{ 0.0f,0.5773*EPSILON,0.0f };
			float3 zDir{ 0.0f,0.0f,0.5773*EPSILON };

			float x1 = DE(iteratedPointPosition + xDir, t);
			float x2 = DE(iteratedPointPosition - xDir, t);
			float y1 = DE(iteratedPointPosition + yDir, t);
			float y2 = DE(iteratedPointPosition - yDir, t);
			float z1 = DE(iteratedPointPosition + zDir, t);
			float z2 = DE(iteratedPointPosition - zDir, t);

			float3 normal = normalize(float3{ x1 - x2 ,y1 - y2,z1 - z2 });

			//Faceforward
			normal = -normal;

			float3 color{ 255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS) };

			float weight = dot(normal, lightDirection);

			// Sphere color

			img[x].r = color.x * lightColor.x;
			img[x].g = color.y * lightColor.y;
			img[x].b = color.z * lightColor.z;



			break;
		}

		distanceTraveled += distanceFromClosestObject;

		if (isnan(distanceTraveled))
			distanceTraveled = 0.0f;
	}

	long long endForTimer = clock64();


	if (idx == 0 && idy == 0)
		printf("Tempo di esecuzione for per il primo thread: %fs\n", ((endForTimer - startForTimer) / ((float)peakClk)));




}

__global__ void childKernel()
{
	printf("Sono un figlio.\n");
}