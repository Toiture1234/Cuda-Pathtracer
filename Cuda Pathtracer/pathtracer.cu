#include "pathtracer.cuh" // very important first include for cuda to work

namespace pathtracer {
	// constants
	__shared__ float3* accum_buffer_dev;
	__shared__ uint8_t* display_buffer_dev;
	__shared__ uint8_t* pathtracer_buffer_dev;

	Triangle* cudaTriangleList;
	int* cudaTrianglesIndex;
	BVH_Node* cudaBVHNodes;
	Material* cudaMaterialList;

	__device__ int RootandUsedNodes_dev[2];

	// host functions
	void initCuda(kernelParams& params) {
		printf("Init cuda...\n");

		// perf metrics
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		// device
		cudaError_t error = cudaSetDevice(0);
		if (error != cudaSuccess) {
			printf("ERROR : no device compatible !!");
		}
		// memory alloc
		cudaMalloc((void**)&accum_buffer_dev, params.windowSize.x * params.windowSize.y * sizeof(float3));
		cudaMalloc((void**)&display_buffer_dev, params.windowSize.x * params.windowSize.y * 4 * sizeof(uint8_t));
		cudaMalloc((void**)&pathtracer_buffer_dev, params.windowSize.x * params.windowSize.y * 4 * sizeof(uint8_t));

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float miliS = 0;
		cudaEventElapsedTime(&miliS, start, stop);

		printf("Initialisation done in %f ms !\nInformations : \n", miliS);
		printf("    Ray size : %i bytes,\n    Material size : %i bytes,\n    Hit size : %i bytes.\n", sizeof(Ray), sizeof(Material), sizeof(Hit));
	}
	void transferTriangles(int* trianglesI, Triangle* allTri, int nbTri) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		cudaMalloc((void**)&cudaTriangleList, nbTri * sizeof(Triangle));
		cudaMemcpy(cudaTriangleList, allTri, nbTri * sizeof(Triangle), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&cudaTrianglesIndex, nbTri * sizeof(int));
		cudaMemcpy(cudaTrianglesIndex, trianglesI, nbTri * sizeof(int), cudaMemcpyHostToDevice);
		
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		printf("Triangles tranfered in %f ms \n", ms);
	}
	void tranfertMaterials(Material* hostMatList, int nbMat) {
		cudaMalloc((void**)&cudaMaterialList, nbMat * sizeof(Material));
		cudaMemcpy(cudaMaterialList, hostMatList, nbMat * sizeof(Material), cudaMemcpyHostToDevice);
		printf("Materials transfered !\n");
	}
	void tranfertBVH(BVH_Node* nodes_host, int rootNIdx_host, int nodesUsed_host, int nbTri) {
		//cudaMemcpyToSymbol(Nodes_dev, nodes_host, (N * 2 - 1) * sizeof(BVH_Node));
		int p[2] = { rootNIdx_host, nodesUsed_host };
		cudaMemcpyToSymbol(RootandUsedNodes_dev, p, 2 * sizeof(int));

		cudaMalloc((void**)&cudaBVHNodes, (nbTri * 2 - 1) * sizeof(BVH_Node));
		cudaMemcpy(cudaBVHNodes, nodes_host, (nbTri * 2 - 1) * sizeof(BVH_Node), cudaMemcpyHostToDevice);
		printf("BVH transfered !!\n");
	}
	
	void endCuda() {

		cudaError_t error = cudaFree(accum_buffer_dev);
		if (error != cudaSuccess) printf("CAN'T FREE ACCUM_BUFFER_DEV IN pathtracer.cu : %s", cudaGetErrorString(error));
		error = cudaFree(display_buffer_dev);
		if (error != cudaSuccess) printf("CAN'T FREE DISPLAY_BUFFER_DEV IN pathtracer.cu : %s", cudaGetErrorString(error));
		error = cudaFree(pathtracer_buffer_dev);
		if (error != cudaSuccess) printf("CAN'T FREE PATHTRACER_BUFFER_DEV IN pathtracer.cu : %s", cudaGetErrorString(error));

		cudaFree(cudaTriangleList);
		cudaFree(cudaBVHNodes);
		cudaFree(cudaTrianglesIndex);
		cudaFree(cudaMaterialList);

		printf("Everything has been free !\n");
	}
	// device code

	// device constants (like triangles or something)
	

	//pathtracer

	__device__ inline bool triangleIntersect(Hit& hit, Ray ray, Triangle tri, Material mat) {
		float3 edge1 = tri.b - tri.a;
		float3 edge2 = tri.c - tri.a;
		float3 h = cross(ray.d, edge2);
		float a = dot(edge1, h);
		if (a > -0.0001f && a < 0.0001f) return false;
		float f = 1. / a;
		float3 s = ray.o - tri.a;
		float u = f * dot(s, h);
		if (u < 0. || u > 1) return false;
		float3 q = cross(s, edge1);
		float v = f * dot(ray.d, q);
		if (v < 0 || u + v > 1) return false;
		float t = f * dot(edge2, q);

		if (t < 0. || t > hit.t) return false;
		float3 oldAbsorption = hit.mat.extinction;

		hit.t = t;
		hit.mat = mat;
		hit.hit = true;

		float w = 1.f - u - v;

		//some shit to refine here
		float3 normalCalc = normalize(cross(edge1, edge2));
		float mult = -sign(dot(ray.d, normalCalc));
		normalCalc *= mult;
		float3 normalFile = normalize(tri.nA * w + tri.nB * u + tri.nC * v) * mult;
		hit.normal = normalFile;

		// it would be smarter to put this inside intersectBVH but i'm lazy now
		float2 samplePos = tri.tA * w + tri.tB * u + tri.tC * v;
		if (mat.useTexture) {
			float4 read = tex2D<float4>(mat.diffuseTexture, samplePos.x, samplePos.y);
			hit.mat.baseColor = make_float3(read.x, read.y, read.z) * read.w;
			hit.mat.alpha = read.w;
		}
		if (mat.use_mapPr) {
			float read = tex2D<float>(mat.roughnessTexture, samplePos.x, samplePos.y);
			hit.mat.roughness = read;
		}
		if (mat.use_mapPm) {
			float read = tex2D<float>(mat.metallicTexture, samplePos.x, samplePos.y);
			hit.mat.metallic = read;
		}
		if (mat.use_mapKe) {
			float read = tex2D<float>(mat.emissiveTexture, samplePos.x, samplePos.y);
			hit.mat.emissive = make_float3(read, read, read);
		}
		if (mat.use_mapNor) {
			float3 T, B;
			Onb(hit.normal, T, B);
			float4 read = tex2D<float4>(mat.normalTexture, samplePos.x, samplePos.y);
			hit.normal = ToWorld(T, B, hit.normal, make_float3(read.x * 2.f - 1.f, read.y * 2.f - 1.f, read.z * 2.f - 1.f));
		}
		//hit.normal = normalize(mix(normalCalc, hit.normal, smoothstep(0.f,0.3f, -dot(ray.d, hit.normal))));
		hit.normal = refIfNeg(hit.normal, -1.0f * ray.d);
		return true;
	}

	__device__ inline bool BVHIntersect(Hit& hit, Ray ray, int3& debug, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats, bool sunRay) {
		//BVH_Node stack[10];
		int stack[10];
		int stackIdx = 0;
		stack[stackIdx++] = 0;

		float3 invDir = 1. / ray.d;

		while (stackIdx > 0)
		{
			BVH_Node node = cudaNodes[stack[--stackIdx]];
			if (boxIntersectF(hit.t, ray, node.aabbMin, node.aabbMax, invDir) < hit.t) {
				if (node.triCount > 0) { // leaf node
					for (int i = 0; i < node.triCount; i++) { // leaf node
						if (triangleIntersect(hit, ray, cudaTriList[cudaTriIndex[i + node.leftFirst]], cudaMats[cudaTriList[cudaTriIndex[i + node.leftFirst]].matIndex])) {
							debug.z++;
							if (sunRay) return true; // this stops the process when doing shadowing because we don't care of what triangle we hit in this case
						}
					}
				}
				else {
					BVH_Node childLeft = cudaNodes[node.leftFirst];
					BVH_Node childRight = cudaNodes[node.leftFirst + 1];

					float dstLeft = boxIntersectF(hit.t, ray, childLeft.aabbMin, childLeft.aabbMax, invDir);
					float dstRight = boxIntersectF(hit.t, ray, childRight.aabbMin, childRight.aabbMax, invDir);
					int left = node.leftFirst, right = node.leftFirst + 1;

					if (dstLeft > dstRight) {
						if (dstLeft < hit.t) stack[stackIdx++] = left;
						if (dstRight < hit.t) stack[stackIdx++] = right;
					}
					else {
						if (dstRight < hit.t) stack[stackIdx++] = right;
						if (dstLeft < hit.t) stack[stackIdx++] = left;
					}

				}
				debug.x++;
			}
		}
		return false;
	}
	__device__ inline float3 shadowRay(Hit info, Ray sunRay, Ray ray, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats, int3& debug) {
		if (BVHIntersect(info, sunRay, debug, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, true)) {
			return make_float3(0.f, 0.f, 0.f);
		}
		float pdf;
		float3 value = Disney::DisneyEval_2(info, ray.d * 1.0f, info.normal, sunRay.d, pdf);
		return value;
	}
	__device__ inline float3 skyBoxSample(float3 direction, kernelParams params) {
		const float4 texVal = tex2D<float4>(params.cubeMap,
			atan2f(direction.z, direction.x) * (float)(0.5f / PI) + 0.5f, 1.f - (direction.y * 0.5f + 0.5f));
		return make_float3(texVal.x, texVal.y, texVal.z);
	}
	
	__device__ inline Hit map(Ray ray, int3 &debug, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		Hit hit;
		hit.mat = Material();
		
		BVHIntersect(hit, ray, debug, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, 0);

		/*Material sphMat;
		sphMat.metallic = 0.;
		sphMat.baseColor = make_float3(1., 1., 1.);
		sphMat.roughness = 0.8;
		sphMat.specTrans = 0.;
		sphMat.clearcoat = 1.f;
		sphMat.clearcoatRoughness = 0.01;
		sphereIntersect(hit, ray, make_float4(-400., 50., 0., 50.), sphMat);
		sphMat.clearcoatRoughness = 0.1;
		sphereIntersect(hit, ray, make_float4(-250., 50., 0., 50.), sphMat);
		sphMat.clearcoatRoughness = 0.2;
		sphereIntersect(hit, ray, make_float4(-100., 50., 0., 50.), sphMat);
		sphMat.clearcoatRoughness = 0.4;
		sphereIntersect(hit, ray, make_float4(50., 50., 0., 50.), sphMat);
		sphMat.clearcoatRoughness = 0.6;
		sphereIntersect(hit, ray, make_float4(200., 50., 0., 50.), sphMat);
		sphMat.clearcoatRoughness = 1.;
		sphereIntersect(hit, ray, make_float4(350., 50., 0., 50.), sphMat);
		*/

		return hit;
	}
	__device__ inline float3 sampleSkyboxOnBounce(Hit info,
												Ray ray,
												Triangle* cudaTriList,
												int* cudaTriIndex,
												BVH_Node* cudaNodes,
												Material* cudaMats,
												int3& debug,
												Rand_state& state,
												kernelParams params) {
		// generate a sample based on bsdf (for now)
		float3 L;
		float pdf;
		float3 bsdf = Disney::DisneySample_2(info, ray.d * -1.f, info.normal, L, pdf, state);
		float bsdf_pdf;

		Hit info0 = map(Ray(ray.o, L), debug, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
		if (info0.hit) return make_float3(0.f, 0.f, 0.f);

		float3 bsdfEv = Disney::DisneyEval_2(info, ray.d * -1.0, info.normal, L, bsdf_pdf);
		if (pdf > 0.f)
			return skyBoxSample(L, params) * bsdfEv / pdf;
		else return make_float3(0.f, 0.f, 0.f);
	}
	__device__ float HG(float g, float sundotrd) {
		float gg = g * g;	return (1. - gg) / pow(1. + gg - 2. * g * sundotrd, 1.5);
	}
	__device__ float rayleigh(float u) {
		return 0.75 * (1. + clamp(u * u, 0., 1.));
	}
	__device__ float3 skyGradient(float3 rd, kernelParams params) {
		float3 sunColor = mix(make_float3(1., 0.9, 0.8), make_float3(1., 0.2, 0.), exp(-abs(params.sunDirection.y) * 5.));
		float costh0 = dot(rd, params.sunDirection);
		float blend = smoothstep(-1., 0.6, costh0);
		float mult = smoothstep(-0.5, 0.7, costh0) * 0.5 + 0.5;
		float3 day = mix(make_float3(0.3, 0.6, 1.) * 0.7 * rayleigh(costh0), make_float3(0.9, 0.95, 1.), exp(-abs(rd.y) * 5.));
		float3 even = mix(make_float3(0.3, 0.6, 1.) * 0.7, mix(make_float3(0.2, 0.3, 0.4), make_float3(1., 0.2, 0.), blend), exp(-abs(rd.y) * 5.));
		float3 sun = 0.02 / length(rd - params.sunDirection) * sunColor ;

		float3 night = make_float3(0.05, 0.1, 0.2) + clamp(0.02 / length(rd + params.sunDirection), 0., 1.) * make_float3(0.7, 0.8, 1.) * 0.6;
		float3 dayF = mix(day, even, exp(-abs(params.sunDirection.y) * 5.)) + sun;

		float earth = smoothstep(-0.02, 0.02, rd.y) * 0.5 + 0.5;

		float mieM = 1. - abs(dot(params.sunDirection, make_float3(0., 1., 0.))) + 0.3;
		return mix(night, dayF + HG(0.6, costh0) * sunColor * 0.3 * mieM, smoothstep(-0.2, 0.0, params.sunDirection.y)) * earth * 0.75;
		//return vec3(HG(0.9,costh0)) ;
	}
	__device__ inline float3 skyColor(float3 rayDir, kernelParams params) {
		//return make_float3(0.f, 0.f, 0.f);
		//float3 skyGrad = mix(make_float3(0.4, 0.7, 1.), make_float3(0.8, 0.9, 1.), exp(-abs(rayDir.y) * 5.)) * 0.5;
		//return skyGrad * 3.;
		//float sineV = fmax(sin(rayDir.x * 7.) * sin(rayDir.z * 7.), 0.);
		//float sineV = fmaxf(sin(rayDir.x * 18.), 0.f);
		//return make_float3(sineV, sineV, sineV);
		//return make_float3(1., 1., 1.) * fmax((double)rayDir.y, 0.);
		//return dot(rayDir, params.sunDirection) > 0.95 ? make_float3(20., 20., 20.) : make_float3(0.,0.,0.);
		//return skyGrad + smoothstep(1., 0.9, length(rayDir - params.sunDirection) / 0.25) * 50.;
		//return skyGradient(rayDir, params);
		//return make_float3(5.f, 5.f, 5.f);
		/*const float4 texVal = tex2D<float4>(params.cubeMap,
			atan2f(rayDir.z, rayDir.x) * (float)(0.5f / PI) + 0.5f, 1.f - (rayDir.y * 0.5f + 0.5f));
		return make_float3(texVal.x, texVal.y, texVal.z);*/
		return skyBoxSample(rayDir, params);
	}
	__device__ inline void pathtrace(float& result, kernelParams params, Ray ray, Rand_state& state, int channel, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats, float uvX) {
		float rayColor = 1.0;

		Hit info;
		bool inside = false;
		int3 nullVal = make_int3(0, 0, 0);
		bool sampleSky = true;
		float x0 = ray.d.x;
		for (int i = 0; i < 50; i++) {
			info = map(ray, nullVal, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
			info.isInside = inside;

			if (!info.hit) {
				result += getN(skyColor(ray.d, params), channel) * rayColor * sampleSky;
				return;
			}

			ray.o += ray.d * info.t;
			info.mat.roughness = fmaxf(info.mat.roughness, 0.001f);

			if (info.isInside)
				rayColor *= exp(-getN(info.mat.extinction * info.t, channel));
			/*
			// material prop
			float specularChance = 0.;
			float refractionChance = info.mat.refractionProba;

			float rayProba = 1.;
			if (info.mat.f90 > 0.) {
				specularChance = FresnelReflectAmount(info.isInside ? getN(info.mat.IOR, channel) : 1.0f, 
					!info.isInside ? getN(info.mat.IOR, channel) : 1.0f, 
					ray, info);
			}
			//float isSpecular = rand(&state) < specularChance ? 1. : 0.;
			float isSpecular = 0.;
			float isRefractive = 0.;
			float zeta = randC(&state);
			if (specularChance > 0. && zeta < specularChance) {
				isSpecular = 1.;
				rayProba = specularChance;
				ray.o += info.normal * 0.1;
			}
			else if (refractionChance > 0. && zeta < refractionChance + specularChance) {
				isRefractive = 1.;
				rayProba = refractionChance;
				ray.o -= info.normal * 0.1;
			}
			else {
				rayProba = 1. - (specularChance + refractionChance);
				ray.o += info.normal * 0.1;
			}

			float3 diffuseRay = normalize(info.normal + generateUniformSample(state));
			float3 specularRay = reflect(ray.d, info.normal);
			specularRay = normalize(mix(diffuseRay, specularRay, info.mat.specularSmoothness * info.mat.specularSmoothness));

			float3 refractionRayDir = refract(ray.d, info.normal, getN(info.isInside ? info.mat.IOR : 1. / info.mat.IOR, channel));

			ray.d = mix(diffuseRay, specularRay, isSpecular);
			if (dot(refractionRayDir, refractionRayDir) == 0.f) refractionRayDir = ray.d; // should be useless unless f90 == 0
			ray.d = mix(ray.d, refractionRayDir, isRefractive);

			result += getN(info.mat.emissive, channel) * rayColor;

			if (isRefractive < 0.5)
				rayColor *= mix(getN(info.mat.albedo, channel), getN(info.mat.specularAlbedo, channel), isSpecular);
			else inside = !inside;

			rayProba = fmax(rayProba, 0.0001f);
			rayColor /= rayProba;
			*/

			float rC = randC(&state);
			if (rC < info.mat.alpha) {
				float3 L;
				float pdf;
				float3 bsdf = Disney::DisneySample_2(info, ray.d * -1.f, info.normal, L, pdf, state);

				result += getN(info.mat.emissive, channel) * rayColor;

				// disable it for now
				/*if (uvX > 0.5) {
					result += getN(sampleSkyboxOnBounce(info, Ray(ray.o + info.normal * 0.01f, ray.d), cudaTriList, cudaTriIndex, cudaNodes, cudaMats, nullVal, state, params), channel) * rayColor;

					sampleSky = false; // this would be useful when sampling the skybox at each bounce
				}*/

				if (pdf > 0.f) rayColor *= getN(bsdf / pdf, channel);
				else return;

				inside = info.isInside;
				ray.d = L;
			}
			ray.o += ray.d * 0.01f;

			// russian roulette 
			{
				float p = rayColor;
				if (randC(&state) > p)
					return;

				rayColor *= 1.0f / p;
			}
		}
		return;
	}
	__device__ inline float3 pixelColor(kernelParams params, Ray ray, Rand_state& state, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats, float uvX) {
		float3 color = make_float3(0., 0., 0.);
		if (!params.isRendering) {
			int3 debugV = make_int3(0, 0, 0);
			Hit info = map(ray, debugV, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
			//color = info.normal * 0.5 + 0.5;
			//float4 test = tex2D<float4>(cudaMats[0].diffuseTexture, 0.f, 0.f);
			//color = dot(ray.d, info.normal) < 0.f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f);
			//if (dot(info.normal, info.normal) == 0.) color = skyColor(ray.d, params);

			if (info.hit) {
				//float pdf;
				//color = Disney::DisneyEval_2(info, ray.d * -1.0f, info.normal, ray.d * -1.0f, pdf);
				//if (pdf > 0.f) color /= pdf;

				//float3 point = ray.o + ray.d * info.t + info.normal * 0.01;
				//color *= shadowRay(info, Ray(point, params.sunDirection), ray, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, debugV);
				//color = color * 0.5f + 0.5f;
				float NoV = fmaxf(dot(ray.d * -1.0, info.normal), 0.f);
				color = NoV * info.mat.baseColor;
			} else color = skyColor(ray.d, params);
			//color.x = (float)debugV.z * 0.5f;
			//color.z = (float)debugV.x * 0.05f;
		}
		else {
			int channel = int(randC(&state) * 3.);
			switch (channel)
			{
			case 0:
				pathtrace(color.x, params, ray, state, 0, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, uvX);
				break;
			case 1:
				pathtrace(color.y, params, ray, state, 1, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, uvX);
				break;
			case 2:
				pathtrace(color.z, params, ray, state, 2, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, uvX);
				break;
			}
			//pathtrace(color.x, params, ray, state, 0);
			//pathtrace(color.y, params, ray, state, 1);
			//pathtrace(color.z, params, ray, state, 2);
			color *= 3.f;
		}
		return color;
	}
	__global__ inline void renderPixel(kernelParams params, uint8_t* ptBuffer, float3* accumBuff, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		//__syncthread();
		if (x > params.windowSize.x && y > params.windowSize.y) return;

		int idx = x + y * params.windowSize.x;

		Rand_state rand_state;
		curand_init(idx, 0, 4096 * params.frameIndex, &rand_state);

		float2 uv = make_float2(((float)x + randC(&rand_state) * 2.0f - 1.f) / params.windowSize.x, 1.f - ((float)y + randC(&rand_state) * 2.0f - 1.f) / params.windowSize.y);
		float2 uvCam = uv - 0.5f;
		uvCam.x *= (float)params.windowSize.x / params.windowSize.y;

		// rayDirection computations
		float fov = params.fov;
		float3 cameraTarget = params.rayOrigin + make_float3(sin(params.cameraAngle.x) * cos(params.cameraAngle.y), sin(params.cameraAngle.y), -cos(params.cameraAngle.x) * cos(params.cameraAngle.y));
		float3 ww = normalize(cameraTarget - params.rayOrigin);
		float3 uu = normalize(cross(ww, make_float3(0.f, 1.f, 0.f)));
		float3 vv = normalize(cross(uu, ww));
		float3 rayDirection = normalize(uu * uvCam.x + vv * uvCam.y + ww * fov);
		
		if (params.DOF_strenght > 0.f) {
			float RdoT = dot(ww, rayDirection);
			float3 target = params.rayOrigin + rayDirection * params.focalDistance / RdoT;
			params.rayOrigin += generateUniformSample(rand_state) * params.DOF_strenght;
			rayDirection = normalize(target - params.rayOrigin);
		}
		float3 color0 = max3(pixelColor(params, Ray(params.rayOrigin, rayDirection), rand_state, cudaTriList, cudaTriIndex, cudaNodes, cudaMats, uv.x), make_float3(0.f,0.f,0.f));

		__syncthreads();
		if (params.frameIndex == 0 || !params.isRendering)
			accumBuff[idx] = color0;
		else
			accumBuff[idx] += color0;

		float3 color = accumBuff[idx] / (params.isRendering ? params.frameIndex + 1 : 1.);
		color = AgX_tonemap(color);

		// drawing to texture
		int colorX = color.x * 255;
		int colorY = color.y * 255;
		int colorZ = color.z * 255;

		ptBuffer[idx * 4] = uint8_t(clamp(colorX, 0, 255));
		ptBuffer[idx * 4 + 1] = uint8_t(clamp(colorY, 0, 255));
		ptBuffer[idx * 4 + 2] = uint8_t(clamp(colorZ, 0, 255));
		ptBuffer[idx * 4 + 3] = uint8_t(255);
	}

	void render(kernelParams params) {
		const int threadSize = 16;
		dim3 blockSize(8, 8, 1U);
		dim3 gridSize(int(params.windowSize.x / blockSize.x), int(params.windowSize.y / blockSize.y), 1U);
		renderPixel<<<gridSize, blockSize >>>(params, display_buffer_dev, accum_buffer_dev, cudaTriangleList, cudaTrianglesIndex, cudaBVHNodes, cudaMaterialList);
		
		//antialias<<<gridSize, blockSize>>>(params, pathtracer_buffer_dev, display_buffer_dev);
		if (cudaPeekAtLastError() != cudaSuccess) {
			printf("Error with kernel : %s \n", cudaGetErrorString(cudaGetLastError()));
		}

		cudaDeviceSynchronize();
		// copy device display buff. to host
		cudaError_t error = cudaMemcpy(params.pixelBuffer, display_buffer_dev, params.windowSize.x * params.windowSize.y * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("ERROR WHILE TRANSFERING DEVICE DATA TO HOST : %s \n", cudaGetErrorString(error));
		}
	}
}