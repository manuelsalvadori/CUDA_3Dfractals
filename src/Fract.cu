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


std::unique_ptr<sf::Image> Fract::generateFractal(const sf::Vector3f &view)
{

	std::unique_ptr<sf::Image> fract_ptr(new sf::Image());
	fract_ptr->create(width, height, sf::Color::White);

	printKernels();

	// ...
	// calcolo il frame corrente su GPU con CUDA
	// ...

	return fract_ptr;
}

void printKernels() {
	parentKernel << <1, 10 >> > ();
	cudaDeviceReset();
}


__global__ void parentKernel()
{
	printf("Sono il padre\n");
	childKernel << <1, 10 >> > ();
}

__global__ void childKernel()
{
	printf("Sono il figlio! :)\n");
}

