#include <Fract.h>

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
	cudaError_t error1 = cudaGetLastError();

	
	parentKernel << <dimGrid, dimBlock >> > (view, imageDevice);

	// ...
	// calcolo il frame corrente su GPU con CUDA
	// ...

	cudaError_t error2 = cudaGetLastError();   // add this line, and check the error code
								  // check error code here

	cudaError_t error3 = cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
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
__global__ void parentKernel(const float3 &view1, pixel* img)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;

	float3 view = { 0, 0, -1 };
	float3 up = { 0, 1, 0 };
	float3 right = { 1, 0, 0 };

	float u = 2 * (idx / WIDTH) - 1;
	float v = 2 * (idx / HEIGHT) - 1;
	float3 rayOrigin = { u, v,0 };
	float3 rayDirection = normalize(cross(right, up));

	//float distanceTraveled = 0.0;
	//const int maxSteps = 1000;
	//for (int i = 0; i < maxSteps; ++i)
	//{
	//	float3 iteratedPointPosition = rayOrigin + rayDirection * distanceTraveled;
	//	float distanceFromClosestObject = length(iteratedPointPosition) - 0.5; // Distance to sphere of radius 0.5
	//	if (distanceFromClosestObject < EPSILON && idx < WIDTH && idy < HEIGHT)
	//	{
	//		// Sphere color
	//		img[x].r = 255;
	//		img[x].g = 0;
	//		img[x].b = 0;
	//		break;
	//	}

	//	distanceTraveled += distanceFromClosestObject;
	//}

	img[x].r = ((u*255)/2.0)+127;
	img[x].g = ((u * 255) / 2.0) + 127;
	img[x].b = ((u * 255) / 2.0) + 127;

	//childKernel << <1, 10 >> > ();
}

__global__ void childKernel()
{
	printf("Sono un figlio.\n");
}