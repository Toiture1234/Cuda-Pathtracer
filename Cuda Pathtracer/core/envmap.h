#pragma once

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