#include <Fract.h>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

Fract::Fract (int width, int height): width(width), height(height){}

Fract::~Fract(){}

int sf::Image getWidth() const
{
  return width;
}

int sf::Image getHeight() const
{
  return height;
}

unique_ptr<sf::Image> Fract::generateFractal(const sf::Vector3f &view)
{
  unique_ptr<sf::Image> fract_ptr(new sf:Image());
  fract_ptr->create(width, height, sf::Color::White);

  // ...
  // calcolo il frame corrente su GPU con CUDA
  // ...

  return fract_ptr;
}


__global__ void fractGen()
{
  cout << "I am on GPU :D" << endl;
}
