#pragma once

namespace pathtracer {

	// ACES
	__device__  inline float3 aces(float3 color) {
		const float a = 2.51f;
		const float b = 0.03f;
		const float c = 2.43f;
		const float d = 0.59f;
		const float e = 0.14f;
		color = clamp((color * (color * a + b)) / (color * (color * c + d) + e), make_float3(0.f, 0.f, 0.f), make_float3(1.f, 1.f, 1.f));
		return color;
	}

	// AgX
	__device__ inline float3 AgX_contrastApprox(float3 x) {
		float3 x2 = x * x;
		float3 x4 = x2 * x2;
		return 15.5f * x4 * x2
			- 40.14f * x4 * x
			+ 31.96f * x4
			- 6.868f * x2 * x
			+ 0.4298f * x2
			+ 0.1191f * x
			- 0.00232f;
	}
	__device__ inline float3 AgX(float3 x) {
		const float min_ev = -12.47393f;
		const float max_ev = 4.026069f;

		x = make_float3(0.842479062253094f * x.x + 0.0423282422610123f * x.y + 0.0423756549057051f * x.z,
			0.0784335999999992f * x.x + 0.878468636469772f * x.y + 0.0784336f * x.z,
			0.0792237451477643f * x.x + 0.0791661274605434f * x.y + 0.879142973793104f * x.z);

		x = clamp(log2_3(x), make_float3(min_ev, min_ev, min_ev), make_float3(max_ev, max_ev, max_ev));
		x = (x - min_ev) / (max_ev - min_ev);
		return AgX_contrastApprox(x);
	}

	__device__ inline float3 AgX_Eotf(float3 x) {
		return make_float3(1.19687900512017f * x.x - 0.0528968517574562f * x.y - 0.0529716355144438f * x.z,
			-0.0980208811401368f * x.x + 1.15190312990417f * x.y - 0.0980434501171241f * x.z,
			-0.0990297440797205f * x.x - 0.0989611768448433f * x.y + 1.15107367264116f * x.z);
	}
	__device__ inline float3 AgX_look(float3 x) {
		const float3 lw = make_float3(0.2126f, 0.7152f, 0.0722f);
		float luma = dot(lw, x);

		float3 offset = make_float3(0.f, 0.f, 0.f);
		float3 slope = make_float3(1.f, 1.f, 1.f);
		float power = 1.35f;
		float sat = 1.4f;

		x = pow3(x * slope + offset, power);
		return luma + sat * (x - luma);
	}
	__device__ inline float3 AgX_tonemap(float3 x) {
		x = AgX(x);
		x = AgX_look(x);
		x = AgX_Eotf(x);
		return x;
	}
}