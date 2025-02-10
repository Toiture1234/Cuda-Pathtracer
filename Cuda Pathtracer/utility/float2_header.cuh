#pragma once

namespace pathtracer {
	// vec2
	__device__ inline float2 operator+(const float2& a, const float2 b) {
		return make_float2(a.x + b.x, a.y + b.y);
	}
	__device__ inline float2 operator-(const float2& a, const float2 b) {
		return make_float2(a.x - b.x, a.y - b.y);
	}
	__device__ inline float2 operator*(const float2& a, const float2 b) {
		return make_float2(a.x * b.x, a.y * b.y);
	}
	__device__ inline float2 operator/(const float2& a, const float2 b) {
		return make_float2(a.x / b.x, a.y / b.y);
	}
	__device__ inline float2 operator+=(float2& a, float2 b) {
		a.x += b.x; a.y += b.y;
	}
	__device__ inline float2 operator-=(float2& a, float2 b) {
		a.x -= b.x; a.y -= b.y;
	}
	__device__ inline float2 operator*=(float2& a, float2 b) {
		a.x *= b.x; a.y *= b.y;
	}
	__device__ inline float2 operator/=(float2& a, float2 b) {
		a.x /= b.x; a.y /= b.y;
	}

	__device__ inline float2 operator+(const float2& a, const float b) {
		return make_float2(a.x + b, a.y + b);
	}
	__device__ inline float2 operator-(const float2& a, const float b) {
		return make_float2(a.x - b, a.y - b);
	}
	__device__ inline float2 operator*(const float2& a, const float b) {
		return make_float2(a.x * b, a.y * b);
	}
	__device__ inline float2 operator/(const float2& a, const float b) {
		return make_float2(a.x / b, a.y / b);
	}
	__device__ inline float2 operator+=(float2& a, float b) {
		a.x += b; a.y += b;
	}
	__device__ inline float2 operator-=(float2& a, float b) {
		a.x -= b; a.y -= b;
	}
	__device__ inline float2 operator*=(float2& a, float b) {
		a.x *= b; a.y *= b;
	}
	__device__ inline float2 operator/=(float2& a, float b) {
		a.x /= b; a.y /= b;
	}

	__device__ inline float length(const float2& a) {
		return sqrtf(a.x * a.x + a.y * a.y);
	}
	__device__ inline float2 normalize(const float2& a) {
		float invL = 1.0 / length(a);
		return make_float2(a.x * invL, a.y * invL);
	}
	__device__ inline float dot(const float2& u, const float2& v) {
		return u.x * v.x + u.y * v.y;
	}
}