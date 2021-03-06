#include <Application.h>

Application::Application() {}

Application::~Application() {}

void Application::runApplication() {

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

	// Timers for computing sequential version
	std::chrono::high_resolution_clock::time_point begin;
	std::chrono::high_resolution_clock::time_point end;

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

	// Loop
	while (window.isOpen())
	{
		// Time before frame computation
		printf("Frame Numero %d\n", frameCounter);
		if (PARALLEL)
			CHECK(cudaEventRecord(start))
		else
			begin = std::chrono::high_resolution_clock::now();


		// Compute frame
		std::shared_ptr<sf::Image> frame;
		computeFrame(frameCounter, window, background, frame, fract, view, imgDevice, imageHost, stream, peakClk, texture, sprite);

		// Time after frame computation
		if (PARALLEL)
			CHECK(cudaEventRecord(stop))
		else
			end = std::chrono::high_resolution_clock::now();

		// Measure enlapsed time
		measureEnlapsedTime(start, stop, begin, end);

		// Save frame 
		saveFrame(width, height, frame, frameCounter);

		frameCounter++;

		// Stop the execution after MAX_NUMBER_OF_FRAMES
		if (frameCounter >= MAX_NUMBER_OF_FRAMES) {

			// Log informations
			logPerformanceInfo(frameCounter);

			window.close();
		}

		// Event handling
		eventHandling(window);

		window.display();
	}

	// Cleanup
	cleanupMemory(imageHost, imgDevice, stream);

}

void Application::cleanupMemory(pixelRegionForStream * imageHost[NUM_STREAMS], pixelRegionForStream * imgDevice[NUM_STREAMS], cudaStream_t  stream[NUM_STREAMS])
{
	cudaFreeHost(imageHost);
	cudaFree(imgDevice);
	for (int i = 0; i < NUM_STREAMS; i++) {
		CHECK(cudaStreamDestroy(stream[i]));
	}
}

void Application::logPerformanceInfo(int frameNumber)
{
	ofstream benchmarksLog;
	benchmarksLog.open("benchmarks/benchmarks.txt", ios::app);
	benchmarksLog << ("Versione parallela: " + std::to_string(PARALLEL) + "\n");
	benchmarksLog << ("Frame renderizzati: " + std::to_string(frameNumber) + "\n");
	benchmarksLog << ("Dimensione frame: " + std::to_string((int)WIDTH) + "x" + std::to_string((int)HEIGHT) + "\n");
	benchmarksLog << ("Modello: MandelBulb\n");
	if (PARALLEL) {
		benchmarksLog << ("Dimensione blocco: " + std::to_string((int)BLOCK_DIM_X) + "x" + std::to_string((int)BLOCK_DIM_Y) + "\n");
		benchmarksLog << ("Dimensione griglia: " + std::to_string((int)(WIDTH / (BLOCK_DIM_X*sqrt(NUM_STREAMS)))) + "x" + std::to_string((int)(HEIGHT / (BLOCK_DIM_Y*sqrt(NUM_STREAMS)))) + "\n");
		benchmarksLog << ("Numero stream: " + std::to_string((int)NUM_STREAMS) + "\n");
		benchmarksLog << ("Stream non bloccanti: true\n");
		benchmarksLog << ("Trasferimenti asincroni: true\n");
		benchmarksLog << ("Maschera shared memory: " + std::to_string(USE_MASK) + "\n");
		if (USE_MASK)
			benchmarksLog << ("Dimensioni maschera filtro: " + std::to_string((int)MASK_SIZE) + "\n");
	}
	benchmarksLog << ("Numero iterazioni max DE: " + std::to_string((int)MAX_STEPS) + "\n");
	benchmarksLog << ("Valore epsilon: " + std::to_string(EPSILON) + "\n");
	benchmarksLog << ("Tempo di calcolo totale: " + std::to_string(totalEnlapsedTime) + "s\n");
	benchmarksLog << ("Tempo di calcolo medio per frame: " + std::to_string(totalEnlapsedTime / MAX_NUMBER_OF_FRAMES) + "s\n");
	benchmarksLog << ("--------------\n");
	benchmarksLog.close();
}

void Application::eventHandling(sf::RenderWindow &window)
{
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
}

void Application::saveFrame(int width, int height, std::shared_ptr<sf::Image> &frame, int frameCounter)
{
	caveofprogramming::Bitmap bitmap(width, height);
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			bitmap.setPixel(i, j, (uint8_t)frame->getPixel(i, j).r, (uint8_t)frame->getPixel(i, j).g, (uint8_t)frame->getPixel(i, j).b);

	bitmap.write("img/Frame_" + std::to_string(frameCounter) + ".bmp");
}

void Application::measureEnlapsedTime(const cudaEvent_t & startParallel, const cudaEvent_t & stopParallel, std::chrono::high_resolution_clock::time_point startSequential, std::chrono::high_resolution_clock::time_point stopSequential)
{
	double seconds = 0;
	if (PARALLEL)
	{
		CHECK(cudaEventSynchronize(startParallel));
		CHECK(cudaEventSynchronize(stopParallel));
		float milliseconds = 0;
		CHECK(cudaEventElapsedTime(&milliseconds, startParallel, stopParallel));
		seconds = (milliseconds / 1000);
	}
	else
	{
		seconds = (double)std::chrono::duration_cast<std::chrono::milliseconds>(stopSequential - startSequential).count() / 1000;
	};
	printf("Tempo calcolo frame: %fs\n", seconds);
	printf("--------------\n");
	totalEnlapsedTime += seconds;
}

void Application::computeFrame(int frameCounter, sf::RenderWindow &window, sf::Color &background, std::shared_ptr<sf::Image> &frame, Fract &fract, float3 &view, pixelRegionForStream * imgDevice[NUM_STREAMS], pixelRegionForStream * imageHost[NUM_STREAMS], cudaStream_t  stream[NUM_STREAMS], int peakClk, sf::Texture &texture, sf::Sprite &sprite)
{
	window.clear(background);
	frame = fract.generateFractal(view, imgDevice[0], imageHost[0], stream, peakClk, frameCounter);
	texture.loadFromImage(*frame);
	sprite.setTexture(texture, true);
	window.draw(sprite);
}
