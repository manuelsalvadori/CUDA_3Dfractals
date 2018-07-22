#include <Fract.h>

//Constant memory 
__constant__ float3 view{ 0, 0, -10.0f };
__constant__ float3 forwardV{ 0, 0, 1 };
__constant__ float3 upV{ 0, 1, 0 };
__constant__ float3 rightV{ 1, 0, 0 };
__constant__ float3 rayOrigin = { 0, 0, -10.0f };

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

	// Craete img
	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	// Define block and grid dimension
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 dimGrid(width / (BLOCK_DIM_X*sqrt(NUM_STREAMS)), height / (BLOCK_DIM_Y*sqrt(NUM_STREAMS)));

	// Start each kernel on a separate stream
	int2 streamID{ 0,0 };

	//Parallel version
	if (PARALLEL)
	{
		
		for (int streamNumber = 0; streamNumber < NUM_STREAMS; streamNumber++) {
			streamID.x = streamNumber / (width / (PIXEL_PER_STREAM_X));
			streamID.y = streamNumber % (width / (PIXEL_PER_STREAM_Y));
			pixel* streamRegionHost = imageHost[streamNumber];
			pixel* streamRegionDevice = imageDevice[streamNumber];
			rayMarching << <dimGrid, dimBlock, 0, streams[streamNumber] >> > (streamRegionDevice, rotation, streamID, peakClk);
			CHECK(cudaMemcpyAsync(streamRegionHost, streamRegionDevice, sizeof(pixelRegionForStream), cudaMemcpyDeviceToHost, streams[streamNumber]));
		}

		CHECK(cudaDeviceSynchronize());
		printf("Tutti gli stream sono arrivati alla fine.\n");
	}
	// Sequencial version
	else
	{
		int2 coordinates{ 0,0 };
		for (int i = 0; i < WIDTH; i++) {
			for (int j = 0; j < HEIGHT; j++) {
				coordinates.x = i;
				coordinates.y = j;
				rayMarchingSequential(*imageHost, coordinates, rotation);
			}
		}
	}


	// Fill the window with img
	fillImgWindow(imageHost, streamID, fract_ptr);

	// Rotation increment in each frame in radiants.
	rotation += 0.0174533;	// 1° is 0.0174533 radiants

	return fract_ptr;
}

void Fract::fillImgWindow(pixelRegionForStream * imageHost, int2 &streamID, std::unique_ptr<sf::Image> &fract_ptr)
{
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
}

void Fract::rayMarchingSequential(pixel* img, int2 coordinates, float time) {

	int x = coordinates.y*WIDTH + coordinates.x;

	float u = 5 * (coordinates.x / WIDTH) - 2.5f;
	float v = 5 * (coordinates.y / HEIGHT) - 2.5f;

	float3 view{ 0, 0, -10.0f };
	float3 forwardV{ 0, 0, 1 };
	float3 upV{ 0, 1, 0 };
	float3 rightV{ 1, 0, 0 };
	float3 rayOrigin = { 0, 0, -10.0f };

	float3 point = rightV * u + upV * v;;
	float3 rayDirection = normalize(point - rayOrigin);

	float3 lightPosition = rotate(float3{ 1.0f,-3.0f,-1.0f }, upV, time);
	float3 lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);
	float3 lightColor = normalize(float3{ 66.0f,134.0f,244.0f });

	bool hitOk = false;
	float3 normal{ 0.0f,0.0f,0.0f };
	float3 halfVector{ 0.0f,0.0f,0.0f };
	float weightLight = 0.0f;
	float weightShadow = 0.0f;

	float distanceTraveled = 0.0;
	infoEstimatorResult distanceFromClosestObject = { 0, float3{ 0.0f } };
	for (int i = 0; i < MAX_STEPS; ++i)
	{
		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		distanceFromClosestObject = distanceEstimator(iteratedPointPosition, time);

		// Far plane 
		if (distanceTraveled > FAR_PLANE)
			break;

		if (distanceFromClosestObject.distance < EPSILON)
		{
			// Normals
			computeNormals(iteratedPointPosition, time, normal, rayDirection);

			// halfVector
			halfVector = (-lightDirection + normal) / length(-lightDirection + normal);

			// Weight of light in the final color
			weightLight = dot(normal, halfVector);

			// Weight of shadow
			weightShadow = softShadow(iteratedPointPosition, -lightDirection, time);

			hitOk = true;

			break;
		}

		distanceTraveled += distanceFromClosestObject.distance;

		if (isnan(distanceTraveled))
			distanceTraveled = 0.0f;
	}

	//Set final color. If there's no hit, simply make the pixel black.
	if (hitOk == true)
	{
		img[x].r = (weightShadow) * (weightLight)* distanceFromClosestObject.color.x;
		img[x].g = (weightShadow) * (weightLight)* distanceFromClosestObject.color.z;
		img[x].b = (weightShadow) * (weightLight)* distanceFromClosestObject.color.y;
	}
	else {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}
}

__global__ void rayMarching(pixel* img, float time, int2 streamID, int peakClk)
{
	// We keep in shared memory a (padded) block of the size of the block.
	// This way we can accees the value computed by neighbour pixels.
	__shared__ int3 blockResults[BLOCK_DIM_X + 2 * (MASK_SIZE / 2)][BLOCK_DIM_Y + 2 * (MASK_SIZE / 2)];
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

	float3 point = rightV * u + upV * v;;
	float3 rayDirection = normalize(point - rayOrigin);

	float3 lightPosition = rotate(float3{ 1.0f,-3.0f,-1.0f }, upV, 0);
	float3 lightDirection = normalize(float3{ 0.0f,0.0f,0.0f }-lightPosition);
	float3 lightColor = normalize(float3{ 66.0f,134.0f,244.0f });

	bool hitOk = false;
	float3 normal{ 0.0f,0.0f,0.0f };
	float3 halfVector{ 0.0f,0.0f,0.0f };
	float weightLight = 0.0f;
	float weightShadow = 0.0f;

	//long long startForTimer = clock64();

	float distanceTraveled = 0.0;
	infoEstimatorResult distanceFromClosestObject = { 0, float3{0.0f} };
	for (int i = 0; i < MAX_STEPS; ++i)
	{
		//If 80% of the pixels in the block hit something, block the computation
		//Use the mean of neighbour pixel as color
		if (USE_MASK) {
			int returnFlag;
			meanOptimization(globalCounter, blockResults, sharedId, hitOk, returnFlag);
			if (returnFlag == 2) break;
		}

		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;

		distanceFromClosestObject = distanceEstimator(iteratedPointPosition, time);

		// Far plane 
		if (distanceTraveled > FAR_PLANE)
			break;

		if (idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y  && distanceFromClosestObject.distance < EPSILON)
		{
			// Normals
			computeNormals(iteratedPointPosition, time, normal, rayDirection);

			// halfVector
			halfVector = (-lightDirection + normal) / length(-lightDirection + normal);

			// Weight of light in the final color
			weightLight = dot(normal, halfVector);

			// Weight of shadow
			weightShadow = softShadow(iteratedPointPosition, -lightDirection, time);

			// Save color
			blockResults[sharedId.x][sharedId.y].x = (weightShadow) * (weightLight)* distanceFromClosestObject.color.x;
			blockResults[sharedId.x][sharedId.y].z = (weightShadow) * (weightLight)* distanceFromClosestObject.color.z;
			blockResults[sharedId.x][sharedId.y].y = (weightShadow) * (weightLight)* distanceFromClosestObject.color.y;

			// Keep track of how many thread have finished computing
			atomicAdd(&globalCounter, 1);
			hitOk = true;

			break;
		}

		distanceTraveled += distanceFromClosestObject.distance;

		if (isnan(distanceTraveled))
			distanceTraveled = 0.0f;
	}

	//long long endForTimer = clock64();

	if (!(idx < PIXEL_PER_STREAM_X && idy < PIXEL_PER_STREAM_Y))
		return;

	//Set final color. If there's no hit, simply make the pixel black.
	if (hitOk == true)
	{
		img[x].r = blockResults[sharedId.x][sharedId.y].x /*+ (endForTimer - startForTimer) / ((float)peakClk)*/;
		img[x].g = blockResults[sharedId.x][sharedId.y].y;
		img[x].b = blockResults[sharedId.x][sharedId.y].z;
	}
	else {
		img[x].r = 0;
		img[x].g = 0;
		img[x].b = 0;
	}

}

__global__ void childKernel()
{
	printf("Sono un figlio.\n");
}

inline __host__ __device__ void computeNormals(const float3 &iteratedPointPosition, float time, float3 &normal, float3 &rayDirection)
{
	float3 xDir{ 0.5773*EPSILON,0.0f,0.0f };
	float3 yDir{ 0.0f,0.5773*EPSILON,0.0f };
	float3 zDir{ 0.0f,0.0f,0.5773*EPSILON };

	infoEstimatorResult x1 = distanceEstimator(iteratedPointPosition + xDir, time);
	infoEstimatorResult x2 = distanceEstimator(iteratedPointPosition - xDir, time);
	infoEstimatorResult y1 = distanceEstimator(iteratedPointPosition + yDir, time);
	infoEstimatorResult y2 = distanceEstimator(iteratedPointPosition - yDir, time);
	infoEstimatorResult z1 = distanceEstimator(iteratedPointPosition + zDir, time);
	infoEstimatorResult z2 = distanceEstimator(iteratedPointPosition - zDir, time);

	normal = normalize(float3{ x1.distance - x2.distance ,y1.distance - y2.distance,z1.distance - z2.distance });

	//Faceforward
	if (dot(-rayDirection, normal) < 0)
		normal = -normal;
}

inline __host__ __device__ void meanOptimization(int globalCounter, int3  blockResults[BLOCK_DIM_X + 2 * (MASK_SIZE / 2)][BLOCK_DIM_Y + 2 * (MASK_SIZE / 2)], int2 &sharedId, bool &hitOk, int &retflag)
{
	retflag = 1;
	if (globalCounter > MASK_PERCENTAGE * BLOCK_DIM_X * BLOCK_DIM_Y)
	{
		float3 meanValue{ 0.0f,0.0f,0.0f };
		for (int i = -(MASK_SIZE / 2); i <= (MASK_SIZE / 2); i++)
		{
			for (int j = -(MASK_SIZE / 2); j <= (MASK_SIZE / 2); j++)
			{
				meanValue.x += blockResults[sharedId.x + i][sharedId.y + j].x;
				meanValue.y += blockResults[sharedId.x + i][sharedId.y + j].y;
				meanValue.z += blockResults[sharedId.x + i][sharedId.y + j].z;
			}
		}

		meanValue /= (MASK_SIZE*MASK_SIZE);
		blockResults[sharedId.x][sharedId.y].x = meanValue.x;
		blockResults[sharedId.x][sharedId.y].y = meanValue.y;
		blockResults[sharedId.x][sharedId.y].z = meanValue.z;
		hitOk = true;
		{ retflag = 2; return; };
	}
}

inline __host__ __device__ infoEstimatorResult distanceEstimator(const float3 &iteratedPointPosition, float time)
{
	float3 modifiedIteratedPosition = iteratedPointPosition;
	transformationOnPoint(modifiedIteratedPosition, time);

	infoEstimatorResult n1 = { 0.0f, float3{ 33, 255, 176 } };
	infoEstimatorResult n2 = { 0.0f, float3{ 1, 45, 134 } };
	infoEstimatorResult n3 = { 0.0f, float3{ 0.0f, 255.0f, 0.0f } };

	//return distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
	//return power = abs(cos(t)) * 40 + 2;
	//return distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
	n1.distance = mandelbulb(modifiedIteratedPosition / 2.3f, 8, 8.0f, 8.0f/**abs(sin(time / 2)) + 2.0f*/) * 2.3f;
	n2.distance = sdfBox(modifiedIteratedPosition + float3{ 0.0f,-2.0f,0.0f }, float3{ 10.0f,0.1f,10.0f });
	//n3.distance = sdfBox(modifiedIteratedPosition + float3{ 0.0f,0.0f,-2.0f }, float3{ 10.0f,10.0f,0.1f });
	//float n1 =  mengerBox(rotY(dodecaFold(modifiedIteratedPosition), time), 3); //MOLTO FIGO :DDDDD
	//n1.distance = mengerBox(modifiedIteratedPosition, 3);
	//return mandelbulb(rotY(dodecaFold(iteratedPointPosition), t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	//return mengerBox(rotY(iteratedPointPosition, t), 3);
	//return sdfSphere(iteratedPointPosition , 1.0f);
	//n3.distance = crossCubeSolid(modifiedIteratedPosition, float3{ 1.0f,1.0f,1.0f });
	//return shapeUnion(shapeUnion(n1, n2), n3);
	return shapeUnion(n1, n2);
	//return n1;

}

inline __host__ __device__ void transformationOnPoint(float3 &modifiedIteratedPosition, float time)
{
	modifiedIteratedPosition += float3{ 0.0f,0.0f,-10 * abs(sin(time)) };
	modifiedIteratedPosition = rotate(modifiedIteratedPosition, rightV, -0.78539* abs(sin(time))); // Rotate 45°
	modifiedIteratedPosition = rotate(modifiedIteratedPosition, upV, time);
}

inline __host__ __device__ float softShadow(float3 origin, float3 direction, float time)
{
	float res = 1.0;
	float mint = 0.02f;
	float maxt = 2.5f;
	for (int i = 0; i < 16; i++)
	{
		float h = distanceEstimator(origin + direction * mint, time).distance;
		res = min(res, 16.0*h / mint);
		mint += clamp(h, 0.02, 0.10);
		if (res<0.005 || mint>maxt) break;
	}
	return clamp(res, 0.0, 1.0);
}

inline __host__ __device__ float hardShadow(float3 origin, float3 direction, float time)
{
	float res = 1.0;
	float mint = 0.02f;
	float maxt = 2.5f;
	for (float t = mint; t < maxt; )
	{
		float h = distanceEstimator(origin + direction * t, time).distance;
		if (h < EPSILON)
			return 0.0;
		t += h;
	}
	return 1.0;
}
