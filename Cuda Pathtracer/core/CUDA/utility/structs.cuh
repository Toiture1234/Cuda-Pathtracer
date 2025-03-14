/*Copyright � 2025 Toiture1234
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the �Software�), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

namespace pathtracer {
	struct Ray
	{
		float3 o, d;
		// constructors
		__device__ inline Ray() {
			o = F3_0;
			d = F3_0;
		}
		__device__ inline Ray(float3 O, float3 D) {
			o = O; d = D;
		}
		__device__ ~Ray() {}
	};

	struct Medium {
		float G;
		float3 sigmaA;
		float3 sigmaS; // actually useless
		__device__ __host__ inline Medium() {
			G = 0.f;
			sigmaA = sigmaS = F3_0;
		}
	};
	struct Material {
		float3 baseColor;
		float3 emissive;
		float roughness;
		float anisotropic;
		float metallic;
		float subsurface;
		float sheen;
		float clearcoat;
		float clearcoatRoughness;
		float specTrans;
		float alpha; // almost same as specTrans but no refraction and no change of index of refraction
		float ior;
		Medium medium;

		cudaTextureObject_t diffuseTexture = 0; // actually texture of float4
		bool useTexture;

		cudaTextureObject_t roughnessTexture = 0;
		bool use_mapPr;

		cudaTextureObject_t metallicTexture = 0;
		bool use_mapPm;

		cudaTextureObject_t emissiveTexture = 0;
		bool use_mapKe;

		cudaTextureObject_t normalTexture = 0;
		bool use_mapNor;

		__device__ __host__ inline Material() {
			baseColor = F3_0;
			emissive = F3_0;
			medium = Medium();
			roughness = 0.01f;
			anisotropic = 0.f;
			metallic = 0.f;
			subsurface = 0.f;
			sheen = 0.f;
			clearcoat = 0.f;
			clearcoatRoughness = 0.f;
			specTrans = 0.f;
			ior = 1.5f;
			useTexture = 0;
			use_mapPr = 0;
			use_mapPm = 0;
			use_mapNor = 0;
			alpha = 1.;
			use_mapKe = 0;
		};
	};
	struct Hit {
		float t = 1e10;
		float3 normal = make_float3(0., 0., 0.);
		Material mat;
		bool hit = false;
		bool isInside = false;
		// consructors
		__device__ inline Hit() {}
		__device__ inline Hit(float d, float3 n, Material m) {
			t = d; normal = n; mat = m;
		}

		__device__ ~Hit() {}
	};

	struct Triangle {
		float3 a, b, c, origin;
		float3 nA, nB, nC;
		float2 tA, tB, tC;
		int matIndex;
		__device__ __host__ inline Triangle() {
			a = b = c = nA = nB = nC = origin = make_float3(0.f, 0.f, 0.f);
			tA = tB = tC = make_float2(0.f, 0.f);
			matIndex = 0;
		}
		__device__ __host__ inline Triangle(float3 A, float3 B, float3 C) {
			a = A; b = B; c = C;
			nA = nB = nC = origin = make_float3(0.f, 0.f, 0.f);
			tA = tB = tC = make_float2(0.f, 0.f);
			matIndex = 0;
		}
		__device__ __host__ ~Triangle() {};
	};

	struct BVH_Node {
		float3 aabbMin, aabbMax;
		int leftFirst, triCount;
		__device__ __host__ BVH_Node() {
			aabbMin = aabbMax = make_float3(0.f, 0.f, 0.f);
			leftFirst = triCount = 0;
		}
	};
	struct aabb {
		float3 bMin = make_float3(1e30, 1e30, 1e30), bMax = make_float3(-1e30, -1e30, -1e30);
		void grow(float3 p) {
			bMin = min3(bMin, p), bMax = max3(bMax, p);
		}
		float area() {
			float3 e = bMax - bMin;
			return e.x * e.y + e.y * e.z + e.z * e.x;
		}
	};

	struct kernelParams {
		int2 windowSize;

		// camera stuff
		float3 rayOrigin;
		float3 rayDirectionZ;
		float3 cameraAngle;
		float cameraSpeed;

		float3 sunDirection;

		float focalDistance;
		float DOF_strenght;
		float fov;

		int frameIndex;

		uint8_t* pixelBuffer;

		bool isRendering;

		float envmap_sum;
		int2 envMap_size;
		cudaTextureObject_t cubeMap;
		cudaTextureObject_t envMap_cdf;

		// post process
		float3 mult;
		float gamma;
		float contrast;
		float saturation;
		float exposure;
	};
}