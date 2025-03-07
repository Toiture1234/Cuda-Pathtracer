#include "../pathtracer.cuh"

namespace pathtracer {
	__device__ inline float3 skyBoxSample(float3 direction, kernelParams params) {
		//return make_float3(0.f, 0.f, 0.f);
		const float4 texVal = tex2D<float4>(params.cubeMap,
			atan2f(direction.z, direction.x) * (float)(0.5f / PI) + 0.5f, 1.f - (direction.y * 0.5f + 0.5f));
		return make_float3(texVal.x, texVal.y, texVal.z);
	}

	// evaluation
	__device__ inline float4 evalEnvmap(float3 dir, kernelParams params) {
		float theta = acosf(clamp(dir.y, -1.0f, 1.0f));
		float2 uv = make_float2((PI + atan2f(dir.z, dir.x)) * INV_TWO_PI, theta * INV_PI);

		float4 read = tex2D<float4>(params.cubeMap, uv.x,uv.y);
		float3 color = make_float3(read.x, read.y, read.z);

		float pdf = Disney::Luminance(color) / params.envmap_sum;

		return make_float4(color.x, color.y, color.z, (pdf * params.envMap_size.x * params.envMap_size.y) / (TWO_PI * PI * sin(theta)));
	}

	// sampling
	__device__ inline float f(float x) {
		return x;
	}
	__device__ inline float2 binarySearch(float xi, kernelParams params) {
		int lower = 0;
		int upper = params.envMap_size.y - 1;
		while (lower < upper) {
			int mid = (lower + upper) >> 1;
			if (xi < tex2D<float>(params.envMap_cdf, 1., (float)mid / (float)params.envMap_size.y)) {
				upper = mid;
			}
			else {
				lower = mid + 1;
			}
		}
		int y = (int)clamp(lower, 0, params.envMap_size.y - 1);

		lower = 0;
		upper = params.envMap_size.x - 1;
		while (lower < upper) {
			int mid = (lower + upper) >> 1;
			if (xi < tex2D<float>(params.envMap_cdf, (float)mid / params.envMap_size.x, (float)y / (float)params.envMap_size.y)) {
				upper = mid;
			}
			else {
				lower = mid + 1;
			}
		}
		int x = (int)clamp(lower, 0, params.envMap_size.x - 1);
		return make_float2((float)x / (float)params.envMap_size.x, (float)y / (float)params.envMap_size.y);

		
	}

	__device__ inline float4 sampleEnvMap(float3& color, kernelParams params, Rand_state& state) {
		float2 uv = binarySearch(randC(&state) * params.envmap_sum, params);
		//float2 uv = make_float2(randC(&state), randC(&state));

		float4 read = tex2D<float4>(params.cubeMap, uv.x, uv.y);
		color = make_float3(read.x, read.y, read.z);
		float pdf = Disney::Luminance(color) / params.envmap_sum;

		float phi = uv.x * TWO_PI;
		float theta = uv.y * PI;

		float sin_theta = sinf(theta);

		if (sin_theta == 0.f) pdf = 0.f;
		return make_float4(-sin_theta * cos(phi), cosf(theta), -sin_theta * sin(phi), (pdf * params.envMap_size.x * params.envMap_size.y) / (TWO_PI * PI * sin_theta));
	}
}