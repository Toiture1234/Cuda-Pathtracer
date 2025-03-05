#pragma once

typedef curandStatePhilox4_32_10_t Rand_state;
#define randC(state) curand_uniform(state)

namespace pathtracer {
	__device__ __host__ inline float clamp(float a, float l, float h) {
		return fminf(h, fmaxf(l, a));
	}
	__device__ __host__ inline float3 clamp(const float3& a, const float3& l, const float3& h) {
		return make_float3(clamp(a.x, l.x, h.x), clamp(a.y, l.y, h.y), clamp(a.z, l.z, h.z));
	}
	__device__ __host__ inline float sign(float x) {
		if (x == 0) return 0;
		return x > 0. ? 1. : -1.;
	}
	__device__ __host__ inline float3 sign3(float3 x) {
		return make_float3(sign(x.x), sign(x.y), sign(x.z));
	}
	__device__ __host__ inline float step(float s, float x) {
		return x > s ? 1. : 0.;
	}
	__device__ __host__ inline float smoothstep(float edge0, float edge1, float x) {
		float t = clamp((x - edge0) / (edge1 - edge0), 0., 1.);
		return t * t * (3.0 - 2.0 * t);
	}
	__device__ __host__ inline float mix(float a, float b, float m) {
		return a * (1. - m) + b * m;
	}
	__device__ __host__ inline float3 mix(float3 a, float3 b, float m) {
		return make_float3(a.x * (1. - m) + b.x * m, a.y * (1. - m) + b.y * m, a.z * (1. - m) + b.z * m);
	}
	__device__  inline float3 generateUniformSample(Rand_state& rand_state) {
		float z = randC(&rand_state) * 2.0f - 1.0f;
		float a = randC(&rand_state) * PI * 2.;
		float r = sqrtf(1.0f - z * z);
		float x = r * cosf(a);
		float y = r * sinf(a);
		return make_float3(x, y, z);
	}

	__device__  __host__ inline float getN(float3 value, int n) {
		switch (n)
		{
		case 0:
			return value.x;
		case 1:
			return value.y;
		case 2:
			return value.z;
		}
	}
	__device__ inline void Onb(float3 N, float3& T, float3& B)
	{
		float3 up = abs(N.z) < 0.999 ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
		T = normalize(cross(up, N));
		B = cross(N, T);
	}
	__device__ inline float3 ToWorld(float3 X, float3 Y, float3 Z, float3 V)
	{
		return V.x * X + V.y * Y + V.z * Z;
	}

	__device__ inline float3 ToLocal(float3 X, float3 Y, float3 Z, float3 V)
	{
		return make_float3(dot(V, X), dot(V, Y), dot(V, Z));
	}

	// omg Inigo Quilez you saved my life
	__device__ inline float3 refIfNeg(float3 v, float3 r) {
		float k = dot(v, r);
		return k > 0.0f ? v : v - 2.0f * r * k;
	}

	__device__ inline float powerHeuristic(float a, float b) {
		float t = a * a;
		return t / (b * b + t);
	}
}