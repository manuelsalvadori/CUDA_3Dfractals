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
	dim3 dimGrid(width / (BLOCK_DIM_X*sqrt(NUM_STREAMS)), height / (BLOCK_DIM_Y*sqrt(NUM_STREAMS)));

	float currentTime = clock();
	printf("Delta time: %fs\n", ((currentTime - lastFrameStartTime) / 1000));
	lastFrameStartTime = currentTime;
	rotation += 0.174533;


	// Start each kernel on a separate stream
	int2 streamID{ 0,0 };

	for (int i = 0; i < NUM_STREAMS; i++) {
		streamID.y = i % (width / (BLOCK_DIM_X*NUM_STREAMS));
		streamID.x = i / (width / (BLOCK_DIM_Y*NUM_STREAMS));
		computeNormals << <dimGrid, dimBlock >> > (view, imageDevice, rotation, epsilon, streamID);
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

	float3 view{ 0, 0, -10 };
	float3 forward{ 0, 0, 1 };
	float3 up{ 0, 1, 0 };
	float3 right{ 1, 0, 0 };

	float u = 5 * (idx / WIDTH) - 2.5f;
	float v = 5 * (idy / HEIGHT) - 2.5f;

	float3 point{ u, v,0 };

	float3 rayOrigin = { 0, 0, view.z };
	float3 rayDirection = normalize(point - rayOrigin);

	distanceExtimator(idx, idy, img, x, rayOrigin, rayDirection, t, epsilon);
}

__device__ float distanceExtimator(int idx, int idy, pixel * img, int x, const float3 &rayOrigin, const float3 &rayDirection, float t, float epsilon)
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

	return distanceFromClosestObject;
}

__device__ float DE(const float3 &iteratedPointPosition, float t)
{
	//return distanceFromClosestObject = cornellBoxScene(rotY(iteratedPointPosition, t));
	//return power = abs(cos(t)) * 40 + 2;
	//return distanceFromClosestObject = mandelbulbScene(rotY(iteratedPointPosition, t), 1.0f);
	//return distanceFromClosestObject = mandelbulb(rotY(iteratedPointPosition,t) / 2.3f, 8, 4.0f, 1.0f + 9.0f * 1.0f) * 2.3f;
	return mengerBox(rotY(iteratedPointPosition, t), 3);
	//return sdfSphere(iteratedPointPosition, 0.5f);
}

__global__ void computeNormals(const float3 &view1, pixel* img, float t, float epsilon, int2 streamID)
{
	int idx = (PIXEL_PER_STREAM * streamID.x) + blockDim.x * blockIdx.x + threadIdx.x;
	int idy = (PIXEL_PER_STREAM * streamID.y) + blockDim.y * blockIdx.y + threadIdx.y;
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
		if (distanceTraveled > 100.0f)
			break;

		if (idx < WIDTH && idy < HEIGHT  && distanceFromClosestObject < epsilon)
		{

			float3 xDir{ 0.5773*epsilon,0.0f,0.0f };
			float3 yDir{ 0.0f,0.5773*epsilon,0.0f };
			float3 zDir{ 0.0f,0.0f,0.5773*epsilon };

			float x1 = DE(iteratedPointPosition + xDir, t);
			float x2 = DE(iteratedPointPosition - xDir, t);
			float y1 = DE(iteratedPointPosition + yDir, t);
			float y2 = DE(iteratedPointPosition - yDir, t);
			float z1 = DE(iteratedPointPosition + zDir, t);
			float z2 = DE(iteratedPointPosition - zDir, t);

			float3 color = normalize(float3{ x1 - x2 ,y1 - y2,z1 - z2 });

			////Faceforward
			color = -color;

			// Sphere color
			img[x].r = 255 * color.x;
			img[x].g = 255 * color.y;
			img[x].b = 255 * color.z;
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