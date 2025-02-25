// a CUDA based pathtracer, no volumes yet
// made by Toiture0x04D2
// TODO : add volume support (and maybe an integrated atmosphere idk yet) and also shader based antialiasing

#include "pathtracer.cuh"

// important includes 

// important constants !!!
#define IMG_SIZE_X 1280
#define IMG_SIZE_Y 720

void fillDisplay(uint8_t* display_buff) {
    for (int i = 0; i < IMG_SIZE_X * IMG_SIZE_Y * 4; i += 4) {
        display_buff[i] = 0;
        display_buff[i + 1] = 0;
        display_buff[i + 2] = 0;
        display_buff[i + 3] = 255;
    }
}
void handleKey(float deltaTime, pathtracer::kernelParams& params, sf::RenderWindow* window) {
    if (!params.isRendering) {
        float cameraSpeed = params.cameraSpeed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z)) {
            //params.rayOrigin.z -= 1. * deltaTime;
            params.frameIndex = 0;
            params.rayOrigin.z += -cosf(params.cameraAngle.x) * deltaTime * cameraSpeed;
            params.rayOrigin.x += sinf(params.cameraAngle.x) * deltaTime * cameraSpeed;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            //params.rayOrigin.z += 1. * deltaTime;
            params.frameIndex = 0;
            params.rayOrigin.z -= -cosf(params.cameraAngle.x) * deltaTime * cameraSpeed;
            params.rayOrigin.x -= sinf(params.cameraAngle.x) * deltaTime * cameraSpeed;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)) {
            //params.rayOrigin.x -= 1. * deltaTime;
            params.frameIndex = 0;
            params.rayOrigin.z -= sinf(params.cameraAngle.x) * deltaTime * cameraSpeed;
            params.rayOrigin.x -= cosf(params.cameraAngle.x) * deltaTime * cameraSpeed;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
            //params.rayOrigin.x += 1. * deltaTime;
            params.frameIndex = 0;
            params.rayOrigin.z += sinf(params.cameraAngle.x) * deltaTime * cameraSpeed;
            params.rayOrigin.x += cosf(params.cameraAngle.x) * deltaTime * cameraSpeed;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
            params.rayOrigin.y -= 1. * deltaTime * cameraSpeed;
            params.frameIndex = 0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            params.rayOrigin.y += 1. * deltaTime * cameraSpeed;
            params.frameIndex = 0;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) {
            std::cout << "Camera Position : " << params.rayOrigin.x << ", " << params.rayOrigin.y << ", " << params.rayOrigin.z << ";\n";
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
            params.sunDirection = pathtracer::normalize(make_float3(sin(params.cameraAngle.x) * cos(params.cameraAngle.y), sin(params.cameraAngle.y), -cos(params.cameraAngle.x) * cos(params.cameraAngle.y)));
        }
        sf::Vector2f mousePosition = window->mapPixelToCoords(sf::Mouse::getPosition(*window));
        float angleX = (mousePosition.x - params.windowSize.x * 0.5) / 180.;
        float angleY = (params.windowSize.y * 0.5 - mousePosition.y) / 180.;
        if(sf::Mouse::isButtonPressed(sf::Mouse::Middle))
            params.cameraAngle = make_float3(angleX, angleY, 0.);
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Multiply)) {
        params.isRendering = true;
        params.frameIndex = 0;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Divide)) {
        params.isRendering = false;
    }
}
void saveToFile(sf::Texture* texture) {
    sf::Image img = texture->copyToImage();
    time_t timestamp;
    time(&timestamp);

    char fileName0[26];
    ctime_s(fileName0, sizeof(fileName0), &timestamp);
    std::string fileName1;
    for (int i = 4; i < strlen(fileName0); i++) {
        char ch = fileName0[i];
        if (ch == ' ' || ch == ':')
            fileName1 += '_';
        else if (ch == '\n')
            break;
        else fileName1 += ch;
    }
    if (!img.saveToFile("saves/screenshots/screen_" + fileName1 + ".png"))std::cout << "CAN'T SAVE IMAGE\n";
    else std::cout << "image saved in saves/screenshots/screen_" + fileName1 + ".png\n";
}

void initScene(pathtracer::kernelParams &params) {
    int numOfTriangles = 0;
    int numOfMaterials = 0;
    int maxModelNameSize = 128;

    std::cout << "\n\n---- LOADING NEW MODEL ----\n";

    // user input for model
    char* path = new char[maxModelNameSize];
    std::cout << "Load model : ";
    std::cin.getline(path, maxModelNameSize);
    const char* inputCst = path;

    // model and bvh relative
    if (!pathtracer::readObjFile(inputCst, numOfTriangles, numOfMaterials)) {
        std::cout << "Failed to load model, please relaunch.\n";
        exit(1);
    }
    std::cout << "Please wait during BVH construction\n";
    sf::Clock bvhDebugClock;
    bvhDebugClock.restart();
    pathtracer::buildBVH(numOfTriangles);
    std::cout << "BVH built in " << bvhDebugClock.getElapsedTime().asSeconds() << " seconds;\n";

    
    // initialisation of cuda
    pathtracer::initCuda(params);
    pathtracer::transferTriangles(pathtracer::TriangleIdx, pathtracer::allTriangles, numOfTriangles);
    pathtracer::tranfertMaterials(pathtracer::materialList, numOfMaterials);
    pathtracer::tranfertBVH(pathtracer::bvhNode, pathtracer::rootNodeIdx, pathtracer::nodesUsed, numOfTriangles);
}

int main()
{
    sf::RenderWindow window(sf::VideoMode({ IMG_SIZE_X, IMG_SIZE_Y }), "CUDA pathtracer, v0.9");

    srand(time(NULL));
    std::cout << "---- Pathtracer made by Toiture, v0.9 ----\n";

    // kernel params
    pathtracer::kernelParams params;
    memset(&params, 0, sizeof(pathtracer::kernelParams));
    params.windowSize = make_int2(IMG_SIZE_X, IMG_SIZE_Y);
    params.rayDirectionZ = make_float3(0., 0., -1.);
    params.cameraAngle = make_float3(0., 0., 0.);
    params.rayOrigin = make_float3(0., 1.5, 5.);
    params.cameraSpeed = 720.f;
    params.focalDistance = 200.f;
    params.DOF_strenght = 0.f;
    params.frameIndex = 0;
    params.pixelBuffer = new uint8_t[IMG_SIZE_X * IMG_SIZE_Y * 4];
    params.isRendering = false;
    fillDisplay(params.pixelBuffer);

    params.sunDirection = pathtracer::normalize(make_float3(1., 0.7, -1.));

    cudaArray_t cubeMatData = 0;
    if (!pathtracer::genTextureFromHDR(&params.cubeMap, &cubeMatData, "assets/cubemaps/sunset_noSun.hdr")) {
        std::cout << "Failed to load hdr !\n";
        exit(1);
    }

    // buttons handler
    pathtracer::Buttons::button_slider cameraSpeed({ 0.f,0.f }, { 200.f, 50.f });
    cameraSpeed.setRef(&params.cameraSpeed, 120.f, 1440.f);
    pathtracer::Buttons::button_slider dofStrenght({ 0.f,50.f }, { 200.f,50.f });
    dofStrenght.setRef(&params.DOF_strenght, 0.f, 20.f);
    pathtracer::Buttons::button_slider focalDist({ 0.f,100.f }, { 200.f,50.f });
    focalDist.setRef(&params.focalDistance, 0.f, 1000.f);

    //shaders handler
    sf::Shader antialias;
    antialias.loadFromFile("assets/shaders/vertex.glsl", "assets/shaders/antialiasing.glsl");
    antialias.setUniform("texture", sf::Shader::CurrentTexture);
    antialias.setUniform("resolution", sf::Vector2f(IMG_SIZE_X, IMG_SIZE_Y));

    // scene creation
    initScene(params);
    
    // main drawing
    sf::Sprite drawer;
    sf::Texture display_texture;
    if (!display_texture.create(IMG_SIZE_X, IMG_SIZE_Y)) std::cout << "ERROR CREATING DISPLAY TEXTURE IN main.cpp \n";
    display_texture.update(params.pixelBuffer);
    display_texture.setSmooth(true);

    // mostly random things (clock and font)
    float totalTime = 0.;
    float deltaTime = 1.;
    sf::Clock mainClock;
    mainClock.restart();

    sf::Font font;
    sf::Text debugText;
    if (!font.loadFromFile("assets/fonts/minecraft-regular.ttf")) std::cout << "ERROR LOADING FONT IN main.cpp\n";
    debugText.setFont(font);
    std::string displayString("FPS : ");
    debugText.setString(displayString + to_string(1. / deltaTime));
    debugText.setCharacterSize(20);
    debugText.setFillColor(sf::Color::Red);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        handleKey(deltaTime, params, &window);
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::M)) {
            saveToFile(&display_texture);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl) && sf::Keyboard::isKeyPressed(sf::Keyboard::G)) {
            pathtracer::rootNodeIdx = 0, pathtracer::nodesUsed = 1;

            initScene(params);
            params.frameIndex = 0;
            params.isRendering = false;
        }

        // rendering
        pathtracer::render(params);
        display_texture.update(params.pixelBuffer);

        // drawing
        drawer.setTexture(display_texture);
        window.draw(drawer);
        
        cameraSpeed.update(&window);
        dofStrenght.update(&window);
        focalDist.update(&window);

        // delta time calculations
        deltaTime = mainClock.restart().asSeconds();
        totalTime += deltaTime;
        float fps = 1. / deltaTime;
        debugText.setString(displayString + to_string(fps) + "\nTime spent : " + to_string(totalTime));
        // debug drawing
        window.draw(debugText);
        //std::cout << "fps : " << fps << "Total time : " << totalTime << "\n";

        window.display();

        params.frameIndex ++;
        
    }
    pathtracer::endCuda();
    
    delete[] pathtracer::allTriangles;
    delete[] pathtracer::TriangleIdx;
    delete[] pathtracer::materialList;
    delete[] pathtracer::bvhNode;
} 