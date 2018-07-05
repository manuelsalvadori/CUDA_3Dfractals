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


std::unique_ptr<sf::Image> Fract::generateFractal(const sf::Vector3f &view, pixel *imageDevice, pixel *imageHost)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimGrid(std::ceil(width / 32), std::ceil(height / 32));
	dim3 dimBlock(32, 32);
	cudaError_t error1 = cudaGetLastError();
	parentKernel << <dimGrid, dimBlock >> > (imageDevice);
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


__global__ void parentKernel(pixel* img)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * WIDTH + idx;
	//printf("Sono il padre prima dell'if, %d, e il primo pixel ha valore r  %d\n", x, img[x].r);

	if (idx < WIDTH && idy < HEIGHT) {
		//printf("Dentro if, %d\n", x);
		img[x].r = 250;
		img[x].g = 250;
		img[x].b = 10;
	}

	//printf("Sono il padre dopo l'if, %d, e il primo pixel ha valore r  %d\n", x, img[x].r);

	//childKernel << <1, 10 >> > ();
}

__global__ void childKernel()
{
	//printf("Sono il figlio! :)\n");
}

