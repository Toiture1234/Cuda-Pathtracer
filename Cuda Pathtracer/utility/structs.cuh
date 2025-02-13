#pragma once

namespace pathtracer {
	struct Ray
	{
		float3 o, d;
		// constructors
		__device__ inline Ray() {}
		__device__ inline Ray(float3 O, float3 D) {
			o = O; d = D;
		}
		__device__ ~Ray() {}
	};
	/*struct Material {
		// old one
		float3 albedo = make_float3(0., 0., 0.), specularAlbedo = make_float3(0., 0., 0.), absorbtion = make_float3(0., 0., 0.);
		float refractionProba = 0., smoothness = 1., f0 = 0., f90 = 0.;
		float3 emissive = make_float3(0., 0., 0.);
		bool specularPossible = false;
		float n = 1., IOR = 1.;
		// constructors
		__device__ Material() {}
		__device__ Material(float3 a, float3 sA, float3 ab, bool specPossible, float rP) {
			albedo = a; specularAlbedo = sA; absorbtion = ab; refractionProba = rP; specularPossible = specPossible;
		}
		__device__ Material(float3 a, float3 sA, float3 ab, bool specPossible, float rP, float sm, float f_0, float f_90, float N) {
			albedo = a; specularAlbedo = sA; absorbtion = ab; specularPossible = specPossible; refractionProba = rP; smoothness = sm; f0 = f_0; f90 = f_90; n = N; IOR = N;
		}
		__device__ void setEmissive(float3 e) {
			emissive = e;
		}

		// diffuse part
		float3 albedo;// = make_float3(0., 0., 0.);

		// specular part
		float3 specularAlbedo;// = make_float3(0., 0., 0.);
		float f0;// = 0.;
		float f90;// = 0.;
		float specularSmoothness;// = 0.;

		// refraction part
		float3 absorption;// = make_float3(0., 0., 0.);
		float refractionProba;// = 0.;
		float refrationSmoothness;// = 0.;

		float3 emissive;// = make_float3(0., 0., 0.);
		float3 IOR;// = 1.;
		//float3 n;// = 1.;
		
		__device__ __host__ inline Material() {
			albedo = make_float3(0.f, 0.f, 0.f);
			specularAlbedo = make_float3(0.f, 0.f, 0.f);
			f0 = 0.f, f90 = 0.f;
			specularSmoothness = 0.f;
			absorption = make_float3(0.f, 0.f, 0.f);
			refractionProba = 0.f, refrationSmoothness = 0.f;
			emissive = make_float3(0.f, 0.f, 0.f);
			IOR = make_float3(0.f, 0.f, 0.f);
		}
		__device__ __host__ inline Material(float3 a, float3 sA, float f_0, float f_90, float sR, float3 ab, float rP, float rR, float3 e, float3 ior) {
			albedo = a;
			specularAlbedo = sA;
			f0 = f_0;
			f90 = f_90;
			specularSmoothness = sR;
			absorption = ab;
			refractionProba = rP;
			refrationSmoothness = rR;
			emissive = e;
			IOR = ior;
			//n = ior;
		}
		__device__ __host__ ~Material() {}
	};*/

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
		float ior;
		float3 extinction;

		cudaTextureObject_t diffuseTexture;

		__device__ __host__ inline Material() {
			baseColor = make_float3(0.f, 0.f, 0.f);
			emissive = make_float3(0.f, 0.f, 0.f);
			extinction = make_float3(0.f, 0.f, 0.f);
			roughness = 0.01f;
			anisotropic = 0.f;
			metallic = 0.f;
			subsurface = 0.f;
			sheen = 0.f;
			clearcoat = 0.f;
			clearcoatRoughness = 0.f;
			specTrans = 0.f;
			ior = 1.5f;
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
}