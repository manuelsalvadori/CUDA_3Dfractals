#include <iostream>
#include <Fract.h>
#include <Pixel.h>

int main()
{
	int height = 600;
	int width = 800;
	// window creation and setting
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;
	sf::RenderWindow window(sf::VideoMode(width, height), "3D fractal", sf::Style::Default, settings);
	window.setVerticalSyncEnabled(true);
	sf::Color background(0, 0, 0, 255);

	sf::Sprite sprite;
	sf::Texture texture;
	sf::Image fractal;
	sf::Vector3f view(0.f, 0.f, 0.f);
	Fract fract(width, height);

	//pixel* imageHost;
	//imageHost = (pixel*)malloc(sizeof(pixel)*width*height);

	//// Device memory allocation
	//pixel* imgDevice;
	//CHECK(cudaMalloc((pixel**)&imgDevice, sizeof(pixel)*width*height));

	// loop
	while (window.isOpen())
	{
		window.clear(background);

		texture.loadFromImage(*fract.generateFractal(view/*,imgDevice, imageHost*/));
		sprite.setTexture(texture, true);
		window.draw(sprite);


		// event handling
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Down)
			{
				// move camera
			}

			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.display();
	}
	return 0;
}
