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
	float epsilon = 1e-5;

	
	// Host pinned memory allocation
	pixel* imageHost;
	CHECK(cudaMallocHost((pixel**)&imageHost, sizeof(pixel)*width*height));

	// Device memory allocation
	pixel* imgDevice;
	CHECK(cudaMalloc((pixel**)&imgDevice, sizeof(pixel)*width*height));

	// Create necessary streams
	cudaStream_t stream[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		CHECK(cudaStreamCreate(&stream[i]));
	}



	// loop
	while (window.isOpen())
	{
		window.clear(background);

		texture.loadFromImage(*fract.generateFractal(view, imgDevice, imageHost, epsilon, stream));
		sprite.setTexture(texture, true);
		window.draw(sprite);


		// event handling
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Right)
			{
				//Increment epsilon
				epsilon += 0.001f;
				printf("Epsilon attuale: %f\n", epsilon);
			}
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Left)
			{
				//Decrement epsilon
				epsilon -= 0.001f;
				printf("Epsilon attuale: %f\n", epsilon);
			}

			if (event.type == sf::Event::Closed)
				window.close();
		}
		window.display();
	}

	// Cleanup
	cudaFreeHost(imageHost);
	cudaFree(imgDevice);
	for (int i = 0; i < 6; i++) {
		CHECK(cudaStreamDestroy(stream[i]));
	}

	return 0;
}
