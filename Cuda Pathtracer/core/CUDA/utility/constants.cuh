#pragma once

namespace pathtracer {
	__device__ __constant__ const float PI = 3.1415926535f;
	__device__ __constant__ const float INV_PI = 0.318309f;
	__device__ __constant__ const float TWO_PI = 6.283184f;
	__device__ __constant__ const float INV_TWO_PI = 0.159154976203148f;
	__device__ __constant__ const float INV_4_PI = 0.78539816339744830961566084581988f;

#define F3_1 make_float3(1.f,1.f,1.f)
#define F3_0 make_float3(0.f,0.f,0.f)
}