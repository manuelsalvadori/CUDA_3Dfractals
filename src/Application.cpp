#include <Application.h>

Application::Application(){}

Application::~Application() {}

void Application::startApplication() {

	// Window creation and setting
	int width = WIDTH;
	int height = HEIGHT;
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
		CHECK(cudaStreamCreate(&stream[i]));
	}

	int frameCounter = 0;


	// Loop
	while (window.isOpen())
	{
		// Compute frame
		printf("Frame Numero %d\n", frameCounter);
		CHECK(cudaEventRecord(start));
		window.clear(background);
		std::shared_ptr<sf::Image> frame = fract.generateFractal(view, imgDevice[0], imageHost[0], stream, peakClk);
		texture.loadFromImage(*frame);
		sprite.setTexture(texture, true);
		window.draw(sprite);
		CHECK(cudaEventRecord(stop));

		// Measure enlapsed time
		CHECK(cudaEventSynchronize(start));
		CHECK(cudaEventSynchronize(stop));
		float milliseconds = 0;
		CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		printf("Tempo calcolo frame: %fs\n", (milliseconds / 1000));
		printf("--------------\n", (milliseconds / 1000));

		// Save frame 
		caveofprogramming::Bitmap bitmap(width, height);
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				bitmap.setPixel(i, j, (uint8_t)frame->getPixel(i, j).r, (uint8_t)frame->getPixel(i, j).g, (uint8_t)frame->getPixel(i, j).b);

		bitmap.write("img/Frame_" + std::to_string(frameCounter) + ".bmp");

		frameCounter++;

		// Stop the execution after 360 frames
		if (frameCounter >= 360)
			window.close();

		// Event handling
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

}