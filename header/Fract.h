#ifndef FRACT_H_
#define FRACT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <memory>
#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics.hpp>

#include <Pixel.h>
#include <common.h>


class Fract
{
 public:
   Fract (int width, int height);
   virtual ~Fract();
   std::unique_ptr<sf::Image> generateFractal(const sf::Vector3f &view/*, pixel *imageDevice, pixel *imageHost*/);
   int getWidth() const;
   int getHeight() const;

 private:
   int width;
   int height;

};


__global__ void  parentKernel(/*pixel* img*/);
__global__ void childKernel();


#endif /* FRACT_H_ */
