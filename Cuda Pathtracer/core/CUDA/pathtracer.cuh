#pragma once

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>

#include "includes.cuh"

using namespace std;

#include "utility/float3_header.cuh"
#include "utility/float2_header.cuh"

#include "utility/constants.cuh"
#include "utility/structs.cuh"
#include "utility/cuda_utility.cuh"
#include "utility/tonemapping.cuh"
#include "disney.cuh"

#include "utility/intersectors.cuh"

#include "../user_interface/button.h"
#include "../image_loader.h"
#include "../envmap.h"
#include "../file_reader.h"
#include "../BVH/BVH_builder.cpp"

#define SCENE 0

namespace pathtracer {

	
	void initCuda(kernelParams &params);
	void render(kernelParams params);
	void endCuda();

	void transferTriangles(int* trianglesI, Triangle* allTri, int nbTri);
	void transfertMaterials(Material* hostMatList, int nbMat);
	void transfertBVH(BVH_Node* nodes_host, int rootNIdx_host, int nodesUsed_host, int nbTri);
}