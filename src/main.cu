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

	// ClockRate
	int peakClk = 1;
	CHECK(cudaDeviceGetAttribute(&peakClk, cudaDevAttrClockRate, 0));

	// Events used to compute execution time
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));

	// Host pinned memory allocation
	pixelRegionForStream* imageHost[NUM_STREAMS];
	// Device memory allocation
	pixelRegionForStream* imgDevice[NUM_STREAMS];
	// Create necessary streams
	cudaStream_t stream[NUM_STREAMS];


	for (int i = 0; i < NUM_STREAMS; i++) {
		CHECK(cudaMallocHost((pixelRegionForStream**)&imageHost[i], sizeof(pixelRegionForStream)));
		CHECK(cudaMalloc((pixelRegionForStream**)&imgDevice[i], sizeof(pixelRegionForStream)));
		CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
	}

	////Test, setta a bianco
	//for (int i = 0; i < NUM_STREAMS; i++)
	//{
	//	for (int j = 0; j < PIXEL_PER_STREAM; j++)
	//	{
	//		pixel* temp = *imageHost[i];
	//		temp[j].r = 255;
	//		temp[j].g = 0;
	//		temp[j].b = 0;
	//	}
	//}


	int frameCounter = 0;

	// loop
	while (window.isOpen())
	{
		printf("Frame Numero %d\n", frameCounter);
		CHECK(cudaEventRecord(start));
		window.clear(background);
		texture.loadFromImage(*fract.generateFractal(view, imgDevice[0], imageHost[0], stream, peakClk));
		sprite.setTexture(texture, true);
		window.draw(sprite);
		CHECK(cudaEventRecord(stop));

		CHECK(cudaEventSynchronize(start));
		CHECK(cudaEventSynchronize(stop));
		float milliseconds = 0;
		CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		printf("Tempo calcolo frame: %fs\n", (milliseconds / 1000));
		printf("--------------\n", (milliseconds / 1000));
		frameCounter++;


		// event handling
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Right)
			{
				// Camera movement
				// ...
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
