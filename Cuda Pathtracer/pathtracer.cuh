#pragma once

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <utility>
#include <string>
#include <ctime>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <curand_kernel.h>

using namespace std;

#include "utility/float3_header.cuh"
#include "utility/float2_header.cuh"

#include "utility/constants.cuh"
#include "utility/structs.cuh"
#include "utility/cuda_utility.cuh"
#include "disney.cuh"

#define __CUDA_INTERNAL_COMPILATION__
#include "utility/intersectors.cuh"
#undef __CUDA_INTERNAL_COMPILATION__

#include "file_reader.h"
#include "BVH/BVH_builder.cpp"

#define SCENE 0

namespace pathtracer {

	struct kernelParams {
		int2 windowSize;

		// camera stuff
		float3 rayOrigin;
		float3 rayDirectionZ;
		float3 cameraAngle;

		float3 sunDirection;

		float focalDistance;
		float DOF_strenght;

		int frameIndex;

		uint8_t* pixelBuffer;

		bool isRendering;
	};
	void initCuda(kernelParams &params);
	void render(kernelParams params);
	void endCuda();

	void transferTriangles(int* trianglesI, Triangle* allTri, int nbTri);
	void tranfertMaterials(Material* hostMatList, int nbMat);
	void tranfertBVH(BVH_Node* nodes_host, int rootNIdx_host, int nodesUsed_host, int nbTri);
}