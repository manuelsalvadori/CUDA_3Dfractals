#include <Fract.h>
#include <Shapes.h>

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

__device__ float normals[6 * NUM_STREAMS];

std::unique_ptr<sf::Image> Fract::generateFractal(const float3 &view, pixelRegionForStream* imageDevice, pixelRegionForStream * imageHost, cudaStream_t* streams, int peakClk)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 dimGrid(width / (BLOCK_DIM_X*sqrt(NUM_STREAMS)), height / (BLOCK_DIM_Y*sqrt(NUM_STREAMS)));

	// Start each kernel on a separate stream
	int2 streamID{ 0,0 };

	for (int streamNumber = 0; streamNumber < NUM_STREAMS; streamNumber++) {
		streamID.x = streamNumber / (width / (PIXEL_PER_STREAM_X));
		streamID.y = streamNumber % (width / (PIXEL_PER_STREAM_Y));
		pixel* streamRegionHost = imageHost[streamNumber];
		pixel* streamRegionDevice = imageDevice[streamNumber];

		computeNormals << <dimGrid, dimBlock, 0, streams[streamNumber] >> > (view, streamRegionDevice, rotation, streamID, streamNumber, peakClk);

		CHECK(cudaMemcpyAsync(streamRegionHost, streamRegionDevice, sizeof(pixelRegionForStream), cudaMemcpyDeviceToHost, streams[streamNumber]));
	}

	CHECK(cudaDeviceSynchronize());

	printf("Tutti gli stream sono arrivati alla fine.\n");

	// Copy final img of the frame
	//CHECK(cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost));

	cudaEvent_t i, e;
	CHECK(cudaEventCreate(&i));
	CHECK(cudaEventCreate(&e));
	cudaEventRecord(i);
	// Fill the window with img
	for (int streamNumber = 0; streamNumber < NUM_STREAMS; streamNumber++)
	{
		pixel* streamRegionHost = imageHost[streamNumber];
		for (int arrayIndex = 0; arrayIndex < PIXEL_PER_STREAM; arrayIndex++)
		{
			int regionX = arrayIndex / PIXEL_PER_STREAM_X;
			int regionY = arrayIndex % PIXEL_PER_STREAM_X;

			streamID.x = streamNumber / (width / (PIXEL_PER_STREAM_X));
			streamID.y = streamNumber % (width / (PIXEL_PER_STREAM_Y));

			int imgX = regionX + (streamID.x*PIXEL_PER_STREAM_X);
			int imgY = regionY + (streamID.y*PIXEL_PER_STREAM_Y);

			//fract_ptr->setPixel(i, j, sf::Color(imageHost[width * j + i].r, imageHost[width * j + i].g, imageHost[width * j + i].b));
			fract_ptr->setPixel(imgX, imgY, sf::Color(streamRegionHost[arrayIndex].r, streamRegionHost[arrayIndex].g, streamRegionHost[arrayIndex].b));
		}
	}
	cudaEventRecord(e);
	cudaEventSynchronize(i);
	cudaEventSynchronize(e);
	float time = 0.0f;
	cudaEventElapsedTime(&time, i, e);
	printf("Time for %f\n", time);
	rotation += 0.174533;

	return fract_ptr;
}
//
//__global__ void distanceField(const float3 &view1, pixel* img, float t, int2 streamID, float* normals)
//{
//
//	int idx = (PIXEL_PER_STREAM_X * streamID.x) + blockDim.x * blockIdx.x + threadIdx.x;
//	int idy = (PIXEL_PER_STREAM_Y * streamID.y) + blockDim.y * blockIdx.y + threadIdx.y;
//	int x = idy * WIDTH + idx;
//
//	float3 view{ 0, 0, -10 };
//	float3 forward{ 0, 0, 1 };
//	float3 up{ 0, 1, 0 };
//	float3 right{ 1, 0, 0 };
//
//	float u = 5 * (idx / WIDTH) - 2.5f;
//	float v = 5 * (idy / HEIGHT) - 2.5f;
//
//	float3 point{ u, v,0 };
//
//	float3 rayOrigin = { 0, 0, view.z };
//	float3 rayDirection = normalize(point - rayOrigin);
//
//	distanceExtimator(idx, idy, img, x, rayOrigin, rayDirection, t, normals);
//}
//
//__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t, float *normals)
//{
//
//	// Background color
//	if (idx < WIDTH && idy < HEIGHT) {
//		img[x].r = 0;
//		img[x].g = 0;
//		img[x].b = 0;
//	}
//
//	float distanceTraveled = 0.0;
//	const int maxSteps = MAX_STEPS;
//	float distanceFromClosestObject = 0;
//	for (int i = 0; i < maxSteps; ++i)
//	{
//		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;
//
//		DE << <1, 1 >> > (iteratedPointPosition, t, 0, 0, normals);
//		cudaDeviceSynchronize();
//
//
//		// Far plane 
//		if (distanceTraveled > 15.0f)
//			break;
//
//		if (idx < WIDTH && idy < HEIGHT  && normals[0] < EPSILON)
//		{
//			// Sphere color
//			img[x].r = 255 - ((i * 255) / MAX_STEPS);
//			img[x].g = 255 - ((i * 255) / MAX_STEPS);
//			img[x].b = 255 - ((i * 255) / MAX_STEPS);
//			break;
//		}
//
//		distanceTraveled += normals[0];
//
//		if (isnan(distanceTraveled))
//			distanceTraveled = 0.0f;
//	}
//
//	return distanceFromClosestObject;
//}

__global__ void DE(const float3 iteratedPointPosition, float t, int normalID, int streamID)
{
	//return distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
	//return power = abs(cos(t)) * 40 + 2;
	//return distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
	normals[normalID + streamID * 6] = mandelbulb(rotY(iteratedPointPosition, t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	//return mengerBox(rotY(iteratedPointPosition, t), 3);
	//return sdfSphere(iteratedPointPosition, 1.0f);
	//return crossCubeSolid(rotY(iteratedPointPosition, t), float3{ 0.5f,0.5f,0.5f });

}

__global__ void computeNormals(const float3 &view1, pixel* img, float t, int2 streamID, int streamNumber, int peakClk)
{
	__shared__ int blockResults[BLOCK_DIM_X + 2 * (MASK_SIZE / 2)][BLOCK_DIM_Y + 2 * (MASK_SIZE / 2)];
	int2 sharedId{ threadIdx.x + (MASK_SIZE / 2),threadIdx.y + (MASK_SIZE / 2) };

	__shared__ int  globalCounter;
	globalCounter = 0;
	__syncthreads();

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * PIXEL_PER_STREAM_X + idx;

	float3 view{ 0, 0, -10 };
	float3 forward{ 0, 0, 1 };
	float3 up{ 0, 1, 0 };
	float3 right{ 1, 0, 0 };

	float u = 5 * (((PIXEL_PER_STREAM_X * streamID.x) + idy) / WIDTH) - 2.5f;
	float v = 5 * (((PIXEL_PER_STREAM_Y * streamID.y) + idx) / HEIGHT) - 2.5f;

	float3 point{ u, v,0 };

	float3 rayOrigin = { 0, 0, view.z };
	float3 rayDirection = normalize(point - rayOrigin);

	float3 lightPosition{ 1.0f,1.0f,view.z };
	float3 lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);
	float3 lightColor = normalize(float3{ 66.0f,134.0f,244.0f });

	float3 halfVector = normalize(-lightDirection - rayDirection);

	// Background color
	if (idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y) {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

	bool hitOk = false;

	// Clock extimate
	long long int startForTimer = clock64();

	float distanceTraveled = 0.0;

	float distanceFromClosestObject = 0;
	for (int i = 0; i < MAX_STEPS; ++i)
	{
		// If 80% of the pixels in the block hit something, block the computation
		// Use the mean of neighbour pixel as color
		if (globalCounter > 0.8f*BLOCK_DIM_X*BLOCK_DIM_Y)
		{
			float meanValue = 0.0f;
			for (int i = -(MASK_SIZE / 2); i <= (MASK_SIZE / 2); i++)
			{
				for (int j = -(MASK_SIZE / 2); j <= (MASK_SIZE / 2); j++)
				{
					meanValue += blockResults[sharedId.x + i][sharedId.y + j];
				}
			}

			meanValue /= (MASK_SIZE*MASK_SIZE);
			blockResults[sharedId.x][sharedId.y] = meanValue;
			hitOk = true;
			break;
		}

		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		DE << <1, 1 >> > (iteratedPointPosition, t, 0, 0);
		cudaDeviceSynchronize();

		lightPosition = float3{ 1.0f,1.0f,view.z };
		lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);

		// Far plane 
		if (distanceTraveled > 100.0f)
			break;

		if (idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y  && normals[0] < EPSILON)
		{

			float3 xDir{ 0.5773*EPSILON,0.0f,0.0f };
			float3 yDir{ 0.0f,0.5773*EPSILON,0.0f };
			float3 zDir{ 0.0f,0.0f,0.5773*EPSILON };

			DE << <1, 1 >> > (iteratedPointPosition + xDir, t, 0, streamNumber);
			DE << <1, 1 >> > (iteratedPointPosition - xDir, t, 1, streamNumber);
			DE << <1, 1 >> > (iteratedPointPosition + yDir, t, 2, streamNumber);
			DE << <1, 1 >> > (iteratedPointPosition - yDir, t, 3, streamNumber);
			DE << <1, 1 >> > (iteratedPointPosition + zDir, t, 4, streamNumber);
			DE << <1, 1 >> > (iteratedPointPosition - zDir, t, 5, streamNumber);
			cudaDeviceSynchronize();

			float3 normal = normalize(float3{ normals[0] - normals[1] ,normals[2] - normals[3],normals[4] - normals[5] });

			//Faceforward
			normal = -normal;

			float3 color{ 255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS) };

			float weight = dot(normal, lightDirection);

			// Save color
			blockResults[sharedId.x][sharedId.y] = weight * color.x;
			hitOk = true;
			atomicAdd(&globalCounter, 1);
			break;
		}

		distanceTraveled += normals[0];

		if (isnan(distanceTraveled))
			distanceTraveled = 0.0f;
	}

	long long endForTimer = clock64();

	__syncthreads();

	if (hitOk == true && idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y)
	{
		//Set final color
		img[x].r = blockResults[sharedId.x][sharedId.y] * lightColor.x  /*+ (endForTimer - startForTimer) / ((float)peakClk)*/;
		img[x].g = blockResults[sharedId.x][sharedId.y] * lightColor.y;
		img[x].b = blockResults[sharedId.x][sharedId.y] * lightColor.z;
	}


	if (idx == 0 && idy == 0)
		printf("Tempo di esecuzione for per il primo thread, in stream %d, %d: %fs\n", streamID.x, streamID.y, ((endForTimer - startForTimer) / ((float)peakClk * 1000)));




}

__global__ void childKernel(float3 numeroACaso)
{
	normals[0] = 4.0f;
	//printf("Sono un figlio %d.\n", numeroACaso);
}