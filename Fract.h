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
}
