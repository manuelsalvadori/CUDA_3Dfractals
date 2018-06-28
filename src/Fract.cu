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


std::unique_ptr<sf::Image> Fract::generateFractal(const sf::Vector3f &view/*, pixel *imageDevice, pixel *imageHost*/)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	dim3 dimBlock(128, 128);
	dim3 dimGrid(7, 5);
	printf("Sono vivo.\n");
	parentKernel<<<dimGrid, dimBlock>>>(/*imageDevice*/);
	CHECK(cudaDeviceSynchronize());

	// ...
	// calcolo il frame corrente su GPU con CUDA
	// ...

	//CHECK(cudaMemcpy(imageHost, imageDevice, sizeof(pixel)*width*height, cudaMemcpyDeviceToHost));

	//for (int i = 0; i < 800; i++) {
	//	for (int j = 0; j < 600; j++) {
	//		fract_ptr->setPixel(i, j, sf::Color(imageHost[800 * j + i].r, imageHost[800 * j + i].g, imageHost[800 * j + i].b));
	//	}
	//}

	return fract_ptr;
}


__global__ void parentKernel(/*pixel* img*/)
{
	printf("Sono il padre\n");
	/*int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int x = idy * 800 + idx;

	if (idx < 800 && idy < 600) {
		printf("Dentro if\n");
		img[x].r = 250;
		img[x].g = 10;
		img[x].b = 10;
	}*/

	childKernel << <1, 10 >> > ();
}

__global__ void childKernel()
{
	printf("Sono il figlio! :)\n");
}

