/*Copyright © 2025 Toiture1234
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the “Software”), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

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