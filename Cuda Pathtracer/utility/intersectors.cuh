#pragma once

namespace pathtracer {

	

	// https://gelamisalami.github.io/blog/posts/ray-box-intersection/
	__device__ inline bool boxIntersect(Hit& hit, Ray ray, float3 min, float3 max, Material mat) {
		float3 planesMin = -1.0 * (ray.o - min) / ray.d;
		float3 planesMax = -1.0 * (ray.o - max) / ray.d;

		float3 planesNear = make_float3(fmin(planesMin.x, planesMax.x), fmin(planesMin.y, planesMax.y), fmin(planesMin.z, planesMax.z));
		float3 planesFar = make_float3(fmax(planesMin.x, planesMax.x), fmax(planesMin.y, planesMax.y), fmax(planesMin.z, planesMax.z));

		float tNear = fmax(fmax(planesNear.x, planesNear.y), planesNear.z);
		float tFar = fmin(fmin(planesFar.x, planesFar.y), planesFar.z);

		bool inside = tNear < 0.;
		if (tNear > tFar || (inside ? tFar : tNear) > hit.t || tFar < 0.) {
			return false; // no intersection
		}
	
		// normal
		/*float3 mask;
		if (!inside) {
			if (planesNear.x > planesNear.y && planesNear.x > planesNear.z) {
				mask = make_float3(1, 0, 0);
			}
			else if (planesNear.y > planesNear.z) {
				mask = make_float3(0, 1, 0);
			}
			else {
				mask = make_float3(0, 0, 1);
			}
		}
		else {
			float3 point = ray.o + ray.d * tFar;
		}*/

		hit.t = inside ? tFar : tNear;
		float3 point = ray.o + ray.d * hit.t;

		// normal calculations
		float e = 0.0005;
		float3 center = (max + min) * 0.5;
		float3 size = abs3(max - min) * 0.5;

		float3 pc = point - center;
		float3 normal = make_float3(0., 0., 0.);
		normal += make_float3(sign(pc.x), 0., 0.) * step(abs(abs(pc.x) - size.x), e);
		normal += make_float3(0., sign(pc.y), 0.) * step(abs(abs(pc.y) - size.y), e);
		normal += make_float3(0., 0., sign(pc.z)) * step(abs(abs(pc.z) - size.z), e);
		normal = normalize(normal);
		
		hit.normal = normal * (inside ? -1.0 : 1.0);

		float3 oldAbs = hit.mat.extinction;
		hit.mat = mat;
		//hit.mat.n = inside ? hit.mat.IOR : 1. / hit.mat.IOR;
		//hit.mat.absorption = inside ? hit.mat.absorption : oldAbs;
		hit.t = inside ? tFar : tNear;
		hit.hit = true;
		//hit.isInside = inside;
		return true;
	}
	__device__ inline bool boxIntersectNoMat(float t, Ray ray, float3 min, float3 max) {
		float3 planesMin = -1.0 * (ray.o - min) / ray.d;
		float3 planesMax = -1.0 * (ray.o - max) / ray.d;

		float3 planesNear = make_float3(fmin(planesMin.x, planesMax.x), fmin(planesMin.y, planesMax.y), fmin(planesMin.z, planesMax.z));
		float3 planesFar = make_float3(fmax(planesMin.x, planesMax.x), fmax(planesMin.y, planesMax.y), fmax(planesMin.z, planesMax.z));

		float tNear = fmax(fmax(planesNear.x, planesNear.y), planesNear.z);
		float tFar = fmin(fmin(planesFar.x, planesFar.y), planesFar.z);

		if (tFar > tNear && tNear < t && tFar > 0) return true;
		return false;
	}
	__device__ inline float boxIntersectF(float t, Ray ray, float3 aabbMin, float3 aabbMax, float3 invRayDir) {
		float3 planesMin = -1.0 * (ray.o - aabbMin) * invRayDir;
		float3 planesMax = -1.0 * (ray.o - aabbMax) * invRayDir;

		float3 planesNear = make_float3(fmin(planesMin.x, planesMax.x), fmin(planesMin.y, planesMax.y), fmin(planesMin.z, planesMax.z));
		float3 planesFar = make_float3(fmax(planesMin.x, planesMax.x), fmax(planesMin.y, planesMax.y), fmax(planesMin.z, planesMax.z));

		float tNear = fmax(fmax(planesNear.x, planesNear.y), planesNear.z);
		float tFar = fmin(fmin(planesFar.x, planesFar.y), planesFar.z);

		if (tFar > tNear && tFar > 0) return tNear;
		else return 1e30;
	}

	__device__ inline float boxIntersectF2(float t, Ray ray, float3 aabbMin, float3 aabbMax) {
		float3 invD = 1. / ray.d;
		float tx1 = (aabbMin.x - ray.o.x) * invD.x, tx2 = (aabbMax.x - ray.o.x) * invD.x;
		float tMin = fminf(tx1, tx2), tMax = fmaxf(tx1, tx2);
		float ty1 = (aabbMin.y - ray.o.y) * invD.y, ty2 = (aabbMax.y - ray.o.y) * invD.y;
		tMin = fmaxf(tMin, fminf(ty1, ty2)), tMax = fminf(tMax, fmaxf(ty1, ty2));
		float tz1 = (aabbMin.z - ray.o.z) * invD.z, tz2 = (aabbMax.z - ray.o.z) * invD.z;
		tMin = fmaxf(tMin, fminf(tz1, tz2)), tMax = fminf(tMax, max(tz1, tz2));
		if (tMax > tMin && tMax > 0.) return tMin;
		return 1e30;
	}
	__device__ inline bool sphereIntersect(Hit& hit, Ray ray, float4 sphere, Material mat) {
		float3 oc = ray.o - make_float3(sphere.x, sphere.y, sphere.z);
		float b = dot(oc, ray.d);
		float3 qc = oc - ray.d * b;
		float h = sphere.w * sphere.w - dot(qc, qc);
		if (h < 0.0) return false;
		h = sqrtf(h);
		float tNear = -b - h;
		float tFar = -b + h;
		bool inside = tNear < 0.;
		if (tFar < 0. || (inside ? tFar : tNear) > hit.t) return false;

		float3 oldAbs = hit.mat.extinction;
		hit.mat = mat;
		//hit.mat.n = inside ? hit.mat.IOR : 1. / hit.mat.IOR;
		//hit.mat.absorption = inside ? hit.mat.absorption : oldAbs;
		hit.t = inside ? tFar : tNear;
		hit.hit = true;
		//hit.isInside = inside;
		float3 pos = ray.o + ray.d * hit.t;
		hit.normal = normalize(pos - make_float3(sphere.x, sphere.y, sphere.z)) * (inside ? -1.0 : 1.0);
		return true;
	}

	__device__ inline bool planeIntersect(Hit& hit, Ray ray, float3 normal, float shift, Material mat) {
		float dis = dot(normal, ray.d);
		if (-dis > 0.001) {
			float t = dot(make_float3(shift, shift, shift) - ray.o, normal) / dis;
			if (t > hit.t || t < 0.) return false;
			hit.normal = normal;
			hit.t = t;
			hit.mat = mat;
			hit.hit = true;
			return true;
		}
		return false;
	}

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
		
		float w = 1. - u - v;

		//some shit to refine here
		float3 normalCalc = normalize(cross(edge1, edge2));
		float mult = -sign(dot(ray.d, normalCalc));
		normalCalc *= mult;
		float3 normalFile = normalize(tri.nA * w + tri.nB * u + tri.nC * v) * mult;
		hit.normal = dot(normalFile, ray.d) < 0.f ? normalFile : normalCalc;
		//hit.normal = normalCalc;

		return true;
	}
	__device__ inline float4 triangleIntersect_noMat(Ray ray, Triangle tri) {
		float3 edge1 = tri.b - tri.a;
		float3 edge2 = tri.c - tri.a;
		float3 h = cross(ray.d, edge2);
		float a = dot(edge1, h);
		if (a > -0.0001f && a < 0.0001f) return make_float4(0., 0., 0., -1.);
		float f = 1. / a;
		float3 s = ray.o - tri.a;
		float u = f * dot(s, h);
		if (u < 0. || u > 1) return make_float4(0., 0., 0., -1.);
		float3 q = cross(s, edge1);
		float v = f * dot(ray.d, q);
		if (v < 0 || u + v > 1) return make_float4(0., 0., 0., -1.);
		float t = f * dot(edge2, q);

		float3 normal = normalize(cross(edge1, edge2));
		normal *= -sign(dot(ray.d, normal));
		return make_float4(normal.x, normal.y, normal.z, t);
	}

	// CHANGE THISSSS
	//__device__ int dev_ABC[N];
	//__device__ Triangle dev_TRI[N];

	// BVH
	//__device__ BVH_Node Nodes_dev[N * 2 - 1];
	__device__ int RootandUsedNodes_dev[2];

	//__device__ Material matList[1];

	__device__ inline bool BVHIntersect(Hit& hit, Ray ray, int3& debug, Triangle* cudaTriList, int* cudaTriIndex, BVH_Node* cudaNodes, Material* cudaMats) {
		/*
		BVH_Node* node = &Nodes_dev[0], *stack[N];
		// slow asf
		//BVH_Node** stack = (BVH_Node**)malloc(64); 
		int stackPtr = 0;
		
		
		// DONT WORK FUCKKKKK
		if (boxIntersectF(hit.t, ray, node->aabbMin, node->aabbMax) < 1e30) {
			while (1) {
				//if (boxIntersectF(hit.t, ray, node->aabbMin, node->aabbMax) != 1e30) {
				debug.x++;
				if (node->triCount >= 1) {
					for (int i = 0; i < node->triCount; i++) {
						if (triangleIntersect(hit, ray, dev_TRI[dev_ABC[i + node->firstTriIdx]], mat)) {
							debug.y++;
						}
					}

					if (stackPtr == 0) break; else node = stack[--stackPtr];
					continue;
				}
				BVH_Node* child1 = &Nodes_dev[node->leftNode];
				BVH_Node* child2 = &Nodes_dev[node->leftNode + 1];
				float dist1 = boxIntersectF(hit.t, ray, child1->aabbMin, child1->aabbMax);
				float dist2 = boxIntersectF(hit.t, ray, child2->aabbMin, child2->aabbMax);
				if (dist1 > dist2) {
					float d = dist1; dist1 = dist2; dist2 = d;
					BVH_Node* c = child1; child1 = child2; child2 = c;
				}
				if (dist1 == 1e30) {
					if (stackPtr == 0) break;
					else node = stack[--stackPtr];
				}
				else {

					node = child1;
					if (dist2 != 1e30) {
						stack[stackPtr++] = child2;
					}
				}
			}
		}
		*/

		
		//BVH_Node stack[10];
		int stack[10];
		int stackIdx = 0;
		stack[stackIdx++] = 0;

		float dst = hit.t;
		float3 normal = make_float3(0., 0., 0.);
		bool hasIntersected = false;
		float2 minMax = make_float2(-1, dst);

		float3 invDir = 1. / ray.d;

		while (stackIdx > 0)
		{
			BVH_Node node = cudaNodes[stack[--stackIdx]];
			if (boxIntersectF(hit.t, ray, node.aabbMin, node.aabbMax, invDir) < hit.t) {
				if (node.triCount > 0) { // leaf node
					for (int i = 0; i < node.triCount; i++) { // leaf node
						triangleIntersect(hit, ray, cudaTriList[cudaTriIndex[i + node.leftFirst]], cudaMats[cudaTriList[cudaTriIndex[i + node.leftFirst]].matIndex]);
						/*float4 curr = triangleIntersect_noMat(ray, dev_TRI[dev_ABC[i + node.firstTriIdx]]);

						if(curr.w > 0.) {
							minMax = make_float2(fmaxf(minMax.x, dst), fminf(minMax.y, dst));
							if (curr.w < dst) {
								dst = curr.w;
								normal = make_float3(curr.x, curr.y, curr.z);
								hasIntersected = true;
							}
						}*/
					}
					debug.z++;
				}
				else {
					//stack[stackIdx++] = Nodes_dev[node.leftNode];
					//stack[stackIdx++] = Nodes_dev[node.leftNode + 1];

					BVH_Node childLeft = cudaNodes[node.leftFirst];
					BVH_Node childRight = cudaNodes[node.leftFirst + 1];

					float dstLeft = boxIntersectF(hit.t, ray, childLeft.aabbMin, childLeft.aabbMax, invDir);
					float dstRight = boxIntersectF(hit.t, ray, childRight.aabbMin, childRight.aabbMax, invDir);
					int left = node.leftFirst, right = node.leftFirst + 1;

					/*if (dstLeft > dstRight) {
						int c = right, right = left, left = c;
						float cf = dstRight, dstRight = dstLeft, dstLeft = cf;
					}
					if (dstLeft < hit.t) stack[stackIdx++] = left;
					if (dstRight < hit.t) stack[stackIdx++] = right;*/
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
		
		/*if (hasIntersected) {
			bool inside = minMax.x <= minMax.y;
			hit.normal = normal;
			hit.t = dst;

			hit.mat = mat;
			hit.mat.absorption = inside ? mat.absorption : make_float3(0., 0., 0.);
			//hit.mat.absorption = mat.absorption;
			hit.mat.n = inside ? hit.mat.IOR : 1. / hit.mat.IOR;
			hit.hit = true;
		}*/

	}
}