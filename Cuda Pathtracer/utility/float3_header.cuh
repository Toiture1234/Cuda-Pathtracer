#pragma once

namespace pathtracer {
	// vec3
	__device__ __host__ inline float3 operator+(const float3 a, const float3 b) {
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	__device__ __host__ inline float3 operator-(const float3 a, const float3 b) {
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	__device__ __host__ inline float3 operator*(const float3 a, const float3 b) {
		return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
	}
	__device__ __host__ inline float3 operator/(const float3 a, const float3 b) {
		return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
	}
	__device__ __host__ inline float3 operator+=(float3& a, float3 b) {
		a.x += b.x; a.y += b.y; a.z += b.z;
	}
	__device__ __host__ inline float3 operator-=(float3& a, float3 b) {
		a.x -= b.x; a.y -= b.y; a.z -= b.z;
	}
	__device__ __host__ inline float3 operator*=(float3& a, float3 b) {
		a.x *= b.x; a.y *= b.y; a.z *= b.z;
	}
	__device__ __host__ inline float3 operator/=(float3& a, float3 b) {
		a.x /= b.x; a.y /= b.y; a.z /= b.z;
	}

	__device__ __host__ inline float3 operator+(const float3& a, const float b) {
		return make_float3(a.x + b, a.y + b, a.z + b);
	}
	__device__ __host__ inline float3 operator-(const float3& a, const float b) {
		return make_float3(a.x - b, a.y - b, a.z - b);
	}
	__device__ __host__ inline float3 operator*(const float3& a, const float b) {
		return make_float3(a.x * b, a.y * b, a.z * b);
	}
	__device__ __host__ inline float3 operator/(const float3& a, const float b) {
		return make_float3(a.x / b, a.y / b, a.z / b);
	}

	__device__ __host__ inline float3 operator+(const float a, const float3 b) {
		return make_float3(a + b.x, a + b.y, a + b.z);
	}
	__device__ __host__ inline float3 operator-(const float a, const float3 b) {
		return make_float3(a - b.x, a - b.y, a - b.z);
	}
	__device__ __host__ inline float3 operator*(const float a, const float3 b) {
		return make_float3(a * b.x, a * b.y, a * b.z);
	}
	__device__ __host__ inline float3 operator/(const float a, const float3 b) {
		return make_float3(a / b.x, a / b.y, a / b.z);
	}

	__device__ __host__ inline float3 operator+=(float3& a, float b) {
		a.x += b; a.y += b; a.z += b;
	}
	__device__ __host__ inline float3 operator-=(float3& a, float b) {
		a.x -= b; a.y -= b; a.z -= b;
	}
	__device__ __host__ inline float3 operator*=(float3& a, float b) {
		a.x *= b; a.y *= b; a.z *= b;
	}
	__device__ __host__ inline float3 operator/=(float3& a, float b) {
		a.x /= b; a.y /= b; a.z /= b;
	}

	__device__ __host__ inline float length(const float3& a) {
		return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	}
	__device__ __host__ inline float3 normalize(const float3& a) {
		float invL = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
		//float invL = 1. / length(a);
		return make_float3(a.x * invL, a.y * invL, a.z * invL);
	}
	__device__ __host__ inline float dot(const float3& u, const float3& v) {
		return u.x * v.x + u.y * v.y + u.z * v.z;
	}
	__device__ __host__ inline float3 cross(const float3 x, const float3 y) {
		return make_float3(x.y * y.z - y.y * x.z, x.z * y.x - y.z * x.x, x.x * y.y - y.x * x.y);
	}

	__device__ __host__ inline float3 reflect(const float3 I, const float3 N) {
		return I - N * 2.0f * dot(N, I);
	}
	__device__ __host__ inline float3 refract(float3 I, float3 N, float IOR) {
		float k = 1.0 - IOR * IOR * (1. - dot(N, I) * dot(N, I));
		if (k < 0.) return make_float3(0., 0., 0.);
		else {
			return I * IOR - N * (IOR * dot(N, I) + sqrtf(k));
		}
	}
	__device__ __host__ inline float sum(float3 a) {
		return a.x + a.y + a.z;
	}

	__device__ __host__ inline float3 pow3(float3 a, float p) {
		return make_float3(pow(a.x, p), pow(a.y, p), pow(a.z, p));
	}
	__device__ __host__ inline float3 abs3(float3 a) {
		return make_float3(abs(a.x), abs(a.y), abs(a.z));
	}

	__device__ __host__ inline float3 min3(float3 a, float3 b) {
		//return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
		return make_float3(a.x < b.x ? a.x : b.x,
			a.y < b.y ? a.y : b.y,
			a.z < b.z ? a.z : b.z);
	}
	__device__ __host__ inline float3 max3(float3 a, float3 b) {
		//return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
		return make_float3(a.x > b.x ? a.x : b.x,
			a.y > b.y ? a.y : b.y,
			a.z > b.z ? a.z : b.z);
	}
	__device__ __host__ inline float3 log2_3(float3 x) {
		return make_float3(log2f(x.x), log2f(x.y), log2f(x.z));
	}
}