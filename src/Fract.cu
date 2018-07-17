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

//Constant memory 
__constant__ float3 view{ 0, 0, -10.0f };
__constant__ float3 forward{ 0, 0, 1 };
__constant__ float3 up{ 0, 1, 0 };
__constant__ float3 right{ 1, 0, 0 };
__constant__ float3 rayOrigin = { 0, 0, -10.0f };

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
		computeNormals << <dimGrid, dimBlock, 0, streams[streamNumber] >> > (view, streamRegionDevice, rotation, streamID, peakClk);
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

__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float time)
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

		distanceFromClosestObject = DE(iteratedPointPosition, time);


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

__device__ float DE(const float3 &iteratedPointPosition, float time = 0)
{
	//return distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
	//return power = abs(cos(t)) * 40 + 2;
	//return distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
	return mandelbulb(iteratedPointPosition / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	//float n2 = sdfBox(iteratedPointPosition + float3{ 0.0f,-1.5f,0.0f }, float3{4.0f,0.1f,4.0f});
	//return mengerBox(rotY(dodecaFold(iteratedPointPosition), t), 3); //MOLTO FIGO :DDDDD
	//return mandelbulb(rotY(dodecaFold(iteratedPointPosition), t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	//return mengerBox(rotY(iteratedPointPosition, t), 3);
	//return sdfSphere(iteratedPointPosition , 1.0f);
	//return crossCubeSolid(rotY(iteratedPointPosition, t), float3{ 0.5f,0.5f,0.5f });
	//return shapeUnion(n1,n2);

}

__global__ void computeNormals(const float3 &view1, pixel* img, float time, int2 streamID, int peakClk)
{
	// We keep in shared memory a (padded) block of the size of the block.
	// This way we can accees the value computed by neighbour pixels.
	__shared__ int blockResults[BLOCK_DIM_X + 2 * (MASK_SIZE / 2)][BLOCK_DIM_Y + 2 * (MASK_SIZE / 2)];
	int2 sharedId{ threadIdx.x + (MASK_SIZE / 2),threadIdx.y + (MASK_SIZE / 2) };

	// Counter that keep track of how many pixel in the block are already computed.
	__shared__ int  globalCounter;
	globalCounter = 0;
	__syncthreads();

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * PIXEL_PER_STREAM_X + idx;

	float u = 5 * (((PIXEL_PER_STREAM_X * streamID.x) + idy) / WIDTH) - 2.5f;
	float v = 5 * (((PIXEL_PER_STREAM_Y * streamID.y) + idx) / HEIGHT) - 2.5f;

	float3 point{ u, v,0 };
	float3 rayDirection = normalize(point - rayOrigin);

	float3 lightPosition = rotY(float3{ 1.0f,1.0f,2.0f }, time);
	float3 lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);
	float3 lightColor = normalize(float3{ 66.0f,134.0f,244.0f });

	// Background color
	if (idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y) {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

	bool hitOk = false;
	float3 normal{ 0.0f,0.0f,0.0f };
	float3 halfVector{ 0.0f,0.0f,0.0f };
	float weightLight = 0.0f;
	float weightShadow = 0.0f;

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

		distanceFromClosestObject = DE(iteratedPointPosition, time);

		// Far plane 
		if (distanceTraveled > 100.0f)
			break;

		if (idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y  && distanceFromClosestObject < EPSILON)
		{

			float3 xDir{ 0.5773*EPSILON,0.0f,0.0f };
			float3 yDir{ 0.0f,0.5773*EPSILON,0.0f };
			float3 zDir{ 0.0f,0.0f,0.5773*EPSILON };

			float x1 = DE(iteratedPointPosition + xDir, time);
			float x2 = DE(iteratedPointPosition - xDir, time);
			float y1 = DE(iteratedPointPosition + yDir, time);
			float y2 = DE(iteratedPointPosition - yDir, time);
			float z1 = DE(iteratedPointPosition + zDir, time);
			float z2 = DE(iteratedPointPosition - zDir, time);

			normal = normalize(float3{ x1 - x2 ,y1 - y2,z1 - z2 });

			//Faceforward
			if (dot(-rayDirection, normal) < 0)
				normal = -normal;

			// halfVector
			halfVector = (-lightDirection + normal) / length(-lightDirection + normal);

			// The color is the number of iteration
			float3 color{ 255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS),255 - ((i * 255) / MAX_STEPS) };

			// Weight of light in the final color
			weightLight = dot(normal, halfVector);

			// Weight of shadow
			weightShadow = shadow(iteratedPointPosition, -lightDirection);

			// Save color
			blockResults[sharedId.x][sharedId.y] = (1 - weightShadow) * (1 - weightLight) * color.x;
			atomicAdd(&globalCounter, 1);
			hitOk = true;
			break;
		}

		distanceTraveled += distanceFromClosestObject;

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

__device__ float shadow(float3 origin, float3 direction)
{
	float res = 1.0;
	float mint = 0.02f;
	float maxt = 2.5f;
	for (int i = 0; i < 16; i++)
	{
		float h = DE(origin + direction * mint);
		res = min(res, 8.0*h / mint);
		mint += clamp(h, 0.02, 0.10);
		if (res<0.005 || mint>maxt) break;
	}
	return clamp(res, 0.0, 1.0);
}

__global__ void childKernel()
{
	printf("Sono un figlio.\n");
}