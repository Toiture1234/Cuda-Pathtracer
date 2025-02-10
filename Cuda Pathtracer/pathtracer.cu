#include "pathtracer.cuh" // very important first include for cuda to work

namespace pathtracer {
	// constants
	__shared__ float3* accum_buffer_dev;
	__shared__ uint8_t* display_buffer_dev;

	Triangle* cudaTriangleList;
	int* cudaTrianglesIndex;
	BVH_Node* cudaBVHNodes;
	Material* cudaMaterialList;

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
		printf("Everything has been free !\n");
	}
	// device code

	// device constants (like triangles or something)
	

	//pathtracer
	/*__device__ inline float FresnelReflectAmount(float n1, float n2, Ray ray, Hit info)
	{
		// Schlick aproximation
		float r0 = (n1 - n2) / (n1 + n2);
		r0 *= r0;
		float cosX = -dot(info.normal, ray.d);
		if (n1 > n2)
		{
			float n = n1 / n2;
			float sinT2 = n * n * (1.0 - cosX * cosX);
			// Total internal reflection
			if (sinT2 > 1.0)
				return info.mat.f90;
			cosX = sqrtf(1.0 - sinT2);
		}
		float x = 1.0 - cosX;
		float ret = r0 + (1.0 - r0) * x * x * x * x * x;

		// adjust reflect multiplier for object reflectivity
		return mix(info.mat.f0, info.mat.f90, ret);
	}
	*/
	__device__ inline Hit map(Ray ray, int3 &debug, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		Hit hit;
		//hit.mat = Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0.0f, 0.0f, 0.0f, make_float3(0., 0., 0.), 0.0f, 1.0f, make_float3(0., 0., 0.), make_float3(1.5f, 1.5f, 1.5f));
		hit.mat = Material();
		
		/*sphereIntersect(hit, ray, make_float4(-4., 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 1., make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));
		sphereIntersect(hit, ray, make_float4(-2.5, 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 0.95, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));
		sphereIntersect(hit, ray, make_float4(-1., 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 0.9, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));
		sphereIntersect(hit, ray, make_float4(0.5, 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 0.85, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));
		sphereIntersect(hit, ray, make_float4(2., 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 0.8, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));
		sphereIntersect(hit, ray, make_float4(3.5, 0.5, 0., 0.5), Material(make_float3(0.2, 0.8, 0.4), make_float3(1., 1., 1.), 0.4, 1., 0.75, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));


		*/
		//triangleIntersect(hit, ray, Triangle(make_float3(0., 0., 0.), make_float3(3., 0., 0.), make_float3(0., 0., 3.)), Material(make_float3(0.2, 0.3, 0.7), make_float3(1., 1., 1.), 0.02, 0.6, 0.98, make_float3(0., 0., 0.), 0., 1., make_float3(0., 0., 0.), 1.5));

		//boxIntersect(hit, ray, make_float3(-100.0f, 270.0f, -100.0f), make_float3(100.0f, 290.f, 100.0f), Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0.0, 0., 0., make_float3(0.f, 0.f, 0.f), 0.f, 1.f, make_float3(15.f, 15.f, 15.f), make_float3(1.5f, 1.5f, 1.5f)));

		//boxIntersect(hit, ray, make_float3(-400.f, 100.f, 0.f), make_float3(-270.f, 200.f, 230.f), Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0.0, 0., 0., make_float3(0.f, 0.f, 0.f), 0.f, 1.f, make_float3(15.f, 15.f, 15.f), make_float3(1.5f, 1.5f, 1.5f)));
		//boxIntersect(hit, ray, make_float3(-100.0f, 270.0f, -100.0f), make_float3(100.0f, 290.f, 100.0f), Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0.0, 0., 0., make_float3(0.f, 0.f, 0.f), 0.f, 1.f, make_float3(15.f, 15.f, 15.f), make_float3(1.5f, 1.5f, 1.5f)));
		//sphereIntersect(hit, ray, make_float4(18.3, 8., 11.7, 0.75), Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0., 0., 0.9, make_float3(0., 0., 0.), 0., 1., make_float3(10., 2.8, 0.039) * 2., 1.5));
		//sphereIntersect(hit, ray, make_float4(2.6, 15.1, 20.3, 0.75), Material(make_float3(1., 1., 1.), make_float3(1., 1., 1.), 0., 0., 0.9, make_float3(0., 0., 0.), 0., 1., make_float3(10., 2.8, 0.039) * 2., 1.5));
		
		//planeIntersect(hit, ray, make_float3(0., 1., 0.), -2., Material(make_float3(0.1, 0.1, 0.2), make_float3(1., 1., 1.), 0.0, 0., 0., make_float3(0.f, 0.f, 0.f), 0.f, 1.f, make_float3(0.f, 0.f, 0.f), make_float3(1.5f, 1.5f, 1.5f)));
		BVHIntersect(hit, ray, debug, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
		//__syncthreads();

		/*Material sphMat;
		sphMat.metallic = 0.;
		sphMat.baseColor = make_float3(1., 1., 1.);
		sphMat.specTrans = 1.;
		sphMat.extinction = make_float3(0., 0.1, 0.1);
		sphereIntersect(hit, ray, make_float4(-400., 50., 0., 50.), sphMat);
		sphMat.extinction = make_float3(0., 0.1, 0.1) * 2.;
		sphereIntersect(hit, ray, make_float4(-250., 50., 0., 50.), sphMat);
		sphMat.extinction = make_float3(0., 0.1, 0.1) * 3.;
		sphereIntersect(hit, ray, make_float4(-100., 50., 0., 50.), sphMat);
		sphMat.extinction = make_float3(0., 0.1, 0.1) * 4.;
		sphereIntersect(hit, ray, make_float4(50., 50., 0., 50.), sphMat);
		sphMat.extinction = make_float3(0., 0.1, 0.1) * 5.;
		sphereIntersect(hit, ray, make_float4(200., 50., 0., 50.), sphMat);
		sphMat.extinction = make_float3(0., 0.1, 0.1) * 6.;
		sphereIntersect(hit, ray, make_float4(350., 50., 0., 50.), sphMat);
		*/
		//Material boxMat;
		//boxMat.emissive = make_float3(10.f, 10.f, 10.f);
		//boxMat.baseColor = make_float3(1.f, 1.f, 1.f);
		//boxIntersect(hit, ray, make_float3(-100.0f, 270.0f, -100.0f), make_float3(100.0f, 290.f, 100.0f), boxMat);
		
		return hit;
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
		return make_float3(0.f, 0.f, 0.f);
		//float3 skyGrad = mix(make_float3(0.4, 0.7, 1.), make_float3(0.8, 0.9, 1.), exp(-abs(rayDir.y) * 5.)) * 0.5;
		//return skyGrad * 3.;
		//float sineV = fmax(sin(rayDir.x * 7.) * sin(rayDir.z * 7.), 0.);
		//float sineV = fmaxf(sin(rayDir.x * 18.), 0.f);
		//return make_float3(sineV, sineV, sineV);
		//return make_float3(1., 1., 1.) * fmax((double)rayDir.y, 0.);
		//return dot(rayDir, params.sunDirection) > 0.95 ? make_float3(20., 20., 20.) : make_float3(0.,0.,0.);
		//return skyGrad + smoothstep(1., 0.9, length(rayDir - params.sunDirection) / 0.25) * 50.;
		//return skyGradient(rayDir, params);
	}
	__device__ inline void pathtrace(float& result, kernelParams params, Ray ray, Rand_state& state, int channel, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		float rayColor = 1.0;

		Hit info;
		bool inside = false;
		int3 nullVal = make_int3(0, 0, 0);
		for (int i = 0; i < 50; i++) {
			info = map(ray, nullVal, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
			info.isInside = inside;

			if (!info.hit) {
				result += getN(skyColor(ray.d, params), channel) * rayColor;
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


			float3 L;
			float pdf;
			float3 bsdf = Disney::DisneySample(info, ray.d * -1.0, info.normal, L, pdf, state, ray.o);
			inside = info.isInside;

			ray.d = L;

			result += getN(info.mat.emissive, channel) * rayColor;

			if (pdf > 0.f) rayColor *= getN(bsdf / pdf, channel);
			else return;

			ray.o += ray.d * 0.1f;

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
	__device__ inline float3 pixelColor(kernelParams params, Ray ray, Rand_state& state, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		float3 color = make_float3(0., 0., 0.);
		if (!params.isRendering) {
			int3 debugV = make_int3(0, 0, 0);
			Hit info = map(ray, debugV, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
			color = info.normal * 0.5 + 0.5;
			//color = dot(ray.d, info.normal) < 0.f ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f);
			if (dot(info.normal, info.normal) == 0.) color = skyGradient(ray.d, params);
			//if(info.t > params.focalDistance)color = mix(color, make_float3(0.5, 1., 0.5), 0.7);
			//color.x = info.mat.absorption.x;
			//color.y = clamp(info.t / 50., 0.0f, 1.0f);
			//color = info.mat.albedo;
			//color.x = clamp(info.t / 5., 0., 1.);
			//color = make_float3((float)debugV.x / 10.0f, (float)debugV.y / 10.0f, (float)debugV.z);
		}
		else {
			int channel = int(randC(&state) * 3.);
			switch (channel)
			{
			case 0:
				pathtrace(color.x, params, ray, state, 0, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
				break;
			case 1:
				pathtrace(color.y, params, ray, state, 1, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
				break;
			case 2:
				pathtrace(color.z, params, ray, state, 2, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);
				break;
			}
			//pathtrace(color.x, params, ray, state, 0);
			//pathtrace(color.y, params, ray, state, 1);
			//pathtrace(color.z, params, ray, state, 2);
			color *= 3.;
		}

		return color;
	}
	__global__ inline void renderPixel(kernelParams params, uint8_t* dispBuff, float3* accumBuff, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		//__syncthread();
		if (x > params.windowSize.x && y > params.windowSize.y) return;
		
		
		int idx = x + y * params.windowSize.x;

		Rand_state rand_state;
		curand_init(idx, 0, 4096 * params.frameIndex, &rand_state);

		float2 uv = make_float2((float)x / params.windowSize.x, 1. - (float)y / params.windowSize.y);
		float2 uvCam = uv - 0.5;
		uvCam.x *= (float)params.windowSize.x / params.windowSize.y;

		// rayDirection computations
		float fov = 1.;
		float3 cameraTarget = params.rayOrigin + make_float3(sin(params.cameraAngle.x) * cos(params.cameraAngle.y), sin(params.cameraAngle.y), -cos(params.cameraAngle.x) * cos(params.cameraAngle.y));
		float3 ww = normalize(cameraTarget - params.rayOrigin);
		float3 uu = normalize(cross(ww, make_float3(0., 1., 0.)));
		float3 vv = normalize(cross(uu, ww));
		float3 rayDirection = normalize(uu * uvCam.x + vv * uvCam.y + ww * fov);

		/*if (params.isRendering) {
			float3 target = params.rayOrigin + rayDirection * params.focalDistance;
			params.rayOrigin += generateUniformSample(rand_state) * params.DOF_strenght;
			rayDirection = normalize(target - params.rayOrigin);
		}*/
		float3 color0 = pixelColor(params, Ray(params.rayOrigin, rayDirection), rand_state, cudaTriList, cudaTriIndex, cudaNodes, cudaMats);

		__syncthreads();
		if (params.frameIndex == 0 || !params.isRendering)
			accumBuff[idx] = color0;
		else
			accumBuff[idx] += color0;

		float3 color = accumBuff[idx] / (params.isRendering ? params.frameIndex + 1 : 1.);
		aces(color);
		// drawing to texture
		dispBuff[idx * 4] = uint8_t(clamp(color.x * 255, 0, 255));
		dispBuff[idx * 4 + 1] = uint8_t(clamp(color.y * 255, 0, 255));
		dispBuff[idx * 4 + 2] = uint8_t(clamp(color.z * 255, 0, 255));
		dispBuff[idx * 4 + 3] = uint8_t(255);
	}
	
	void render(kernelParams params) {
		const int threadSize = 16;
		dim3 blockSize(8, 8, 1U);
		dim3 gridSize(int(params.windowSize.x / blockSize.x), int(params.windowSize.y / blockSize.y), 1U);
		renderPixel<<<gridSize, blockSize >>>(params, display_buffer_dev, accum_buffer_dev, cudaTriangleList, cudaTrianglesIndex, cudaBVHNodes, cudaMaterialList);
		
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