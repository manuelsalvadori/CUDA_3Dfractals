#include <Fract.h>

int main()
{
	int width = WIDTH;
	int height = HEIGHT;

	// window creation and setting
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;
	sf::RenderWindow window(sf::VideoMode(width, height), "3D fractal", sf::Style::Default, settings);
	window.setVerticalSyncEnabled(true);
	sf::Color background(0, 0, 0, 255);

	sf::Sprite sprite;
	sf::Texture texture;
	sf::Image fractal;
	float3 view = { 0.f, 0.f, -1.f };
	Fract fract(width, height);

	// Host memory allocation
	pixel* imageHost;
	CHECK(cudaMallocHost((pixel**)&imageHost, sizeof(pixel)*width*height));

	// Device memory allocation
	pixel* imgDevice;
	CHECK(cudaMalloc((pixel**)&imgDevice, sizeof(pixel)*width*height));

	//// Costant memory allocation
	//sf::Vector3f upH(0, 1, 0);
	//sf::Vector3f rightH(1, 0, 0);
	//CHECK(cudaMemcpyToSymbol(upDevice, &upH, sizeof(upH), 0, cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpyToSymbol(rightDevice, &rightH, sizeof(rightH), 0, cudaMemcpyHostToDevice));

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("Timeout attivato: %d.\n", devProp.kernelExecTimeoutEnabled);

	// loop
	while (window.isOpen())
	{
		window.clear(background);

		texture.loadFromImage(*fract.generateFractal(view, imgDevice, imageHost));
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
