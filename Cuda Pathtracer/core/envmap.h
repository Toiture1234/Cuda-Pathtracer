#pragma once

/* thanks to https ://github.com/knightcrawler25/GLSL-PathTracer/blob/master/src/core/EnvironmentMap.h 
 * because it really helped me out on how to importance sample hdris.
 */

 /*
  * MIT License
  *
  * Copyright(c) 2019 Asif Ali
  *
  * Permission is hereby granted, free of charge, to any person obtaining a copy
  * of this software and associated documentation files (the "Software"), to deal
  * in the Software without restriction, including without limitation the rights
  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  * copies of the Software, and to permit persons to whom the Software is
  * furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in all
  * copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  * SOFTWARE.
  */

#include "hdr_loader.h"
#include "CUDA/includes.cuh"

namespace pathtracer {
	class envMap {
	public :
		envMap() {
			width = height = 0;
			data = cdf = nullptr;
			sum = 0.f;
		}

		bool loadMap(const std::string path);
		void buildCDF();

		bool generateCUDAtexture(cudaTextureObject_t* texObj, cudaArray_t* texData);
		bool generateCUDAtexture_cdf(cudaTextureObject_t* cdfObj, cudaArray_t* cdfData);
		void generateCUDAenvmap(cudaTextureObject_t* texObj, cudaArray_t* texData, cudaTextureObject_t* cdfObj, cudaArray_t* cdfData);

		void transfertToParams(int2* size, float* tot_sum);

		unsigned int width;
		unsigned int height;
		float sum;
		float* data;
		float* cdf;
	};
}