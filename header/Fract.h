#ifndef FRACT_H_
#define FRACT_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics.hpp>


class Fract
{
 public:
   Fract (int width, int height);
   virtual ~Fract();
   std::unique_ptr<sf::Image> generateFractal(const sf::Vector3f &view);
   int getWidth() const;
   int getHeight() const;

 private:
   int width;
   int height;
};


__global__ void parentKernel();
__global__ void childKernel();
void printKernels();

#endif /* FRACT_H_ */
