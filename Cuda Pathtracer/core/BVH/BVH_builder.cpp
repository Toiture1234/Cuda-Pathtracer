//#pragma once

// giga big thanks to https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/ 
namespace pathtracer {

	//static BVH_Node bvhNode[N * 2 - 1];
	//static int rootNodeIdx = 0, nodesUsed = 1;

	static BVH_Node* bvhNode;
	static int rootNodeIdx = 0, nodesUsed = 1;

	inline void updateNodesBounds(int nodeIdx) {
		BVH_Node& node = bvhNode[nodeIdx];
		node.aabbMin = make_float3(1e30, 1e30, 1e30);
		node.aabbMax = make_float3(-1e30, -1e30, -1e30);
		for (int first = node.leftFirst, i = 0; i < node.triCount; i++) {
			int leafTriIdx = TriangleIdx[first + i];
			Triangle& leafTri = allTriangles[leafTriIdx];
			node.aabbMin = min3(node.aabbMin, leafTri.a - 0.1);
			node.aabbMin = min3(node.aabbMin, leafTri.b - 0.1);
			node.aabbMin = min3(node.aabbMin, leafTri.c - 0.1);
			node.aabbMax = max3(node.aabbMax, leafTri.a + 0.1);
			node.aabbMax = max3(node.aabbMax, leafTri.b + 0.1);
			node.aabbMax = max3(node.aabbMax, leafTri.c + 0.1);
		}
	}

	inline float evalSAH(BVH_Node& node, int axis, float pos) {
		aabb leftB, rightB;
		int leftCount = 0, rightCount = 0;
		for (unsigned int i = 0; i < node.triCount; i++) {
			Triangle& tri = allTriangles[TriangleIdx[node.leftFirst + i]];
			if (getN(tri.origin, axis) < pos) {
				leftCount++;
				leftB.grow(tri.a);
				leftB.grow(tri.b);
				leftB.grow(tri.c);
			}
			else {
				rightCount++;
				rightB.grow(tri.a);
				rightB.grow(tri.b);
				rightB.grow(tri.c);
			}
		}
		float cost = leftCount * leftB.area() + rightCount * rightB.area();
		return cost > 0 ? cost : 1e30;
	}
	inline float findBestPlane(BVH_Node& node, int& axis, float& split) {
		float bCost = 1e30;
		for (int a = 0; a < 3; a++) {
			float bMin = getN(node.aabbMin, a);
			float bMax = getN(node.aabbMax, a);

			if (bMin == bMax) continue;
			float scale = (bMax - bMin) / 100.;
			for (int i = 1; i < 100; i++) {
				float cP = bMin + i * scale;
				float cost = evalSAH(node, a, cP);
				if (cost < bCost) 
					split = cP, axis = a, bCost = cost;
			}
		}
		return bCost;
	}
	inline void subdivide(int nodeIdx) {
		BVH_Node& node = bvhNode[nodeIdx];
		//if (node.triCount <= 20) return;
		float3 e = node.aabbMax - node.aabbMin;
		float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
		float parentCost = node.triCount * parentArea;

		// cut choice
		/*
		float3 extent = node.aabbMax - node.aabbMin;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > getN(extent, axis)) axis = 2;
		float splitPos = getN(node.aabbMin + node.aabbMax, axis) * 0.5;
		*/

		// using SAH
		int axis;
		float splitPos;
		float splitCost = findBestPlane(node, axis, splitPos);
		if (splitCost >= parentCost) return;;

		int i = node.leftFirst;
		int j = i + node.triCount - 1;
		while (i <= j)
		{
			if (getN(allTriangles[TriangleIdx[i]].origin, axis) < splitPos) i++;
			else std::swap(TriangleIdx[i], TriangleIdx[j--]);
		}

		int leftCount = i - node.leftFirst;
		if (leftCount == 0 || leftCount == node.triCount) return;
		int leftChildIdx = nodesUsed++;
		int rightChildIdx = nodesUsed++;
		
		bvhNode[leftChildIdx].leftFirst = node.leftFirst;
		bvhNode[leftChildIdx].triCount = leftCount;
		bvhNode[rightChildIdx].leftFirst = i;
		bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
		node.leftFirst = leftChildIdx;
		node.triCount = 0;

		updateNodesBounds(leftChildIdx);
		updateNodesBounds(rightChildIdx);

		subdivide(leftChildIdx);
		subdivide(rightChildIdx);
	}

	inline void buildBVH(int nbTri){

		pathtracer::bvhNode = new BVH_Node[nbTri * 2 - 1];
		for (int i = 0; i < nbTri; i++) {
			allTriangles[i].origin = (allTriangles[i].a + allTriangles[i].b + allTriangles[i].c) * 0.3333f;
			//allTriangles[i].matIndex = 0;
		}
		BVH_Node& root = bvhNode[rootNodeIdx];
		root.leftFirst = 0.;
		root.triCount = nbTri;
		
		updateNodesBounds(rootNodeIdx);
		subdivide(rootNodeIdx);
	}

	
}
