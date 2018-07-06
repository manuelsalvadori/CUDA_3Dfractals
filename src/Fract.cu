#include <Fract.h>
#include <Shapes.h>
#include <ctime>

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

std::unique_ptr<sf::Image> Fract::generateFractal(const float3 &view, pixel *imageDevice, pixel *imageHost)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimGrid(std::ceil(width / 32), std::ceil(height / 32));
	dim3 dimBlock(32, 32);

	float t = clock();
	printf("urrent time: %f\n", t);
	distanceField << <dimGrid, dimBlock >> > (view, imageDevice, t);

	cudaError_t error3 = cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost);

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
__global__ void distanceField(const float3 &view1, pixel* img, float t)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;

	float3 view = { 0, 0, -10 };
	float3 up = { 0, 1, 0 };
	float3 right = { 1, 0, 0 };

	float u = 2 * (idx / WIDTH) - 1;
	float v = 2 * (idy / HEIGHT) - 1;
	float3 rayOrigin = { u, v, -10 };
	float3 rayDirection = { 0,0,1 };

	float distanceTraveled = 0.0;
	const int maxSteps = MAX_STEPS;
	for (int i = 0; i < maxSteps; ++i)
	{
		float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;
		//float distanceFromClosestObject = length(iteratedPointPosition) - 0.8; // Distance to sphere of radius 0.5
		//float distanceFromClosestObject = length(fmaxf(fabs(iteratedPointPosition) - float3{ 0.2f,0.2f,0.2f }, float3{ 0.0f ,0.0f,0.0f }));
		//float3 r = { 0.2f,0.2f,0.2f };
		//float distanceFromClosestObject = (length(iteratedPointPosition / r) - 1.0) * min(min(r.x, r.y), r.z);

		//float distanceFromClosestObject = sphere(iteratedPointPosition, 0.7f);

		float distanceFromClosestObject = cube(rotY(iteratedPointPosition, t), float3{ 0.2f,0.2f,0.2f });

		if (distanceFromClosestObject < EPSILON && idx < WIDTH && idy < HEIGHT)
		{
			// Sphere color
			img[x].r = (i * 255) / 32;
			img[x].g = (i * 255) / 32;
			img[x].b = (i * 255) / 32;
			break;
		}
		else if (idx < WIDTH && idy < HEIGHT) {
			img[x].r = 255;
			img[x].g = 0;
			img[x].b = 0;
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