#pragma once


namespace pathtracer {

	//static float3* verticies_host;
	static Triangle* allTriangles;
	static int* TriangleIdx;
	static Material* materialList;
	//static const std::string SUB_NAME = "";

	struct readableMaterial {
		std::string name;
		Material matProperties;
	};
	static readableMaterial* rMatList;

	inline std::string readUntil(std::string line, int firstIndex, char stopper, int* end) {
		int txtLength = line.length();

		string output;
		for (int i = firstIndex; i < txtLength; i++) {
			char c = line[i];
			*end = i;
			if (c == stopper) return output;
			else output.push_back(c);
		}
		return output;
	}

	inline bool readMtlFile(int* numOfMaterials, std::string path, std::string subDir) { // perform memory allocations inside this, sounds risky
		std::vector<readableMaterial> matList;

		std::string pathC = "assets/models/" + subDir + path;
		ifstream mtlFile(pathC);
		string line;

		if (mtlFile.is_open()) {
			readableMaterial current;
			current.matProperties = Material();

			// reading line by line
			while (getline(mtlFile, line)) {
				int lineLength = strlen(line.c_str()); 
				
				int tokenEnd = 0;
				if (readUntil(line, 0, ' ', &tokenEnd) == "newmtl") {
					if (!current.name.empty()) {
						matList.push_back(current);
						current.matProperties = Material();
					}
					std::string materialName = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
					std::cout << "New material named " << materialName << "\n";
					current.name = materialName;
 				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Kd") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Kd of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					current.matProperties.baseColor = values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ke") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Ke of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					current.matProperties.emissive = values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "d") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "d of " << current.name << " : " << value << "\n";
					current.matProperties.specTrans = 1. - value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Tf") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Ke of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					current.matProperties.medium.sigmaA = 1.f - values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ni") {
					float value = 1.0f;
					value = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					//std::cout << "IOR of " << current.name << " : " << value << "\n";
					current.matProperties.ior = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Pr") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Pr of " << current.name << " : " << value << "\n";
					current.matProperties.roughness = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Pm") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Pm of " << current.name << " : " << value << "\n";
					current.matProperties.metallic = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ps") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "Ps of " << current.name << " : " << value << "\n";
					current.matProperties.sheen = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "aniso") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					//std::cout << "aniso of " << current.name << " : " << value << "\n";
					current.matProperties.anisotropic = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "map_Kd") {
					std::string path = "assets/models/" + subDir + readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
					
					cudaArray_t cuArr = 0;
					current.matProperties.useTexture = genTexture(&current.matProperties.diffuseTexture, &cuArr, path);
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "map_Pr") {
					std::string path = "assets/models/" + subDir + readUntil(line, tokenEnd + 1, '\n', &tokenEnd);

					cudaArray_t cuArr = 0;
					current.matProperties.use_mapPr = genTexture_float(&current.matProperties.roughnessTexture, &cuArr, path);
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "map_Pm") {
					std::string path = "assets/models/" + subDir + readUntil(line, tokenEnd + 1, '\n', &tokenEnd);

					cudaArray_t cuArr = 0;
					current.matProperties.use_mapPm = genTexture_float(&current.matProperties.metallicTexture, &cuArr, path);
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "map_Bump") {
					/*std::string nextToken = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
					std::string tmpPath = nextToken;
					if(nextToken == "-bm")
						readUntil(line, tokenEnd + 1, ' ', &tokenEnd);*/
					std::string tmpPath = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
					
					std::string path = "assets/models/" + subDir + tmpPath;
					cudaArray_t cuArr = 0;
					current.matProperties.use_mapNor = genTexture(&current.matProperties.normalTexture, &cuArr, path);
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "map_Ke") {
					std::string path = "assets/models/" + subDir + readUntil(line, tokenEnd + 1, '\n', &tokenEnd);

					cudaArray_t cuArr = 0;
					current.matProperties.use_mapKe = genTexture_float(&current.matProperties.emissiveTexture, &cuArr, path);
				}
			}
			matList.push_back(current);

			if (matList.size() > 0) {
				// memory allocations
				rMatList = new readableMaterial[matList.size()];
				materialList = new Material[matList.size()];

				std::cout << matList.size() << "\n";
				*numOfMaterials = matList.size();
				for (int i = 0; i < matList.size(); i++) {
					if (i < matList.size()) {
						rMatList[i] = matList[i];
						materialList[i] = matList[i].matProperties;
					}
				}
			}
			else {
				rMatList = new readableMaterial[1];
				materialList = new Material[1];
				*numOfMaterials = 1;
				materialList[0] = Material(/*make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f), 0.f, 0.f, 0.f, make_float3(0.f, 0.f, 0.f), 0.f, 0.f, make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f)*/);
			}
			return true;

			mtlFile.close();
		}
		return false;
	}

	inline int getMatIndex(int matNb, std::string thisName) {
		for (int i = 0; i < matNb; i++) {
			if (rMatList[i].name == thisName) return i;
		}
		return 0;
	}

	inline bool readObjFile(const char* path, const char* subDir, int& nbTri, int& nbMat) {
		float size = 100.f;
		float3 position = make_float3(0.f, 0.f, 0.f);

		int triangleCount = 0; // for avoiding memory issues
		int verticesCount = 0;
		string line;
		ifstream objFile(path);
		std::vector<float3> pointsVec;
		std::vector<int3> faceVec;
		std::vector<float3> normals;
		std::vector<int3> normalPtr;
		std::vector<float2> texCoord;
		std::vector<int3> texCoordPtr; 
		std::vector<int> matIndexes;

		int matNb = 0;
		bool anyNormals = false;
		bool anyTextures = false; // textures are loaded if normals are

		if (objFile.is_open()) {
			int currentIndex = 0;
			while (getline(objFile, line)) {
				int len = strlen(line.c_str());
				
				int tokenEnd = 0;

				// to rework
				if (readUntil(line, 0, ' ', &tokenEnd) == "f") {
					int3 ptsLink = make_int3(0, 0, 0);
					int3 vtLink = make_int3(0, 0, 0);
					int3 nLink = make_int3(0, 0, 0);

					if (anyNormals) {
						std::string read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if(!read.empty())
							vtLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							nLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							vtLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							nLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.z = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							vtLink.z = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
						if (!read.empty())
							nLink.z = stoi(read) - 1;
					}
					else if (anyTextures) {
						std::string read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							vtLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							vtLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.z = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
						if (!read.empty())
							vtLink.z = stoi(read) - 1;
					}
					else {
						std::string read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							ptsLink.x = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							ptsLink.y = stoi(read) - 1;

						read = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
						if (!read.empty())
							ptsLink.z = stoi(read) - 1;
					}
					faceVec.push_back(ptsLink);
					normalPtr.push_back(nLink);
					texCoordPtr.push_back(vtLink);
					matIndexes.push_back(currentIndex);
					triangleCount++;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "v") {
					float3 point = make_float3(0.f, 0.f, 0.f);
					point.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					point.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					point.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));

					pointsVec.push_back(point * size + position);
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "vn") {
					float3 norm = make_float3(0.f, 0.f, 0.f);
					norm.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					norm.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					norm.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));

					normals.push_back(norm);
					anyNormals = true;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "vt") {
					float2 coords = make_float2(0.f, 0.f);
					coords.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					coords.y = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));

					texCoord.push_back(coords);
					anyTextures = true;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "usemtl") {
					std::string thisName = readUntil(line, 7, '\n', &tokenEnd);
					currentIndex = getMatIndex(matNb, thisName);
					std::cout << "Mat index is now " << currentIndex << "\n";
					std::cout << thisName << "\n";
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "mtllib") { // consider this being before usemtl
					std::string mtlName = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
					if (!readMtlFile(&matNb, mtlName, subDir)) return false;
					std::cout << matNb << "\n";
					nbMat = matNb;
				}
			}
			std::cout << "Model has " << triangleCount << " triangles, " << triangleCount << " triangles counted." << "\n";
			objFile.close();

			nbTri = triangleCount; // really important line actually !!
			pathtracer::allTriangles = new pathtracer::Triangle[nbTri];
			pathtracer::TriangleIdx = new int[nbTri];
			for (int i = 0; i < nbTri; i++) {
				allTriangles[i] = Triangle(make_float3(0, 0., 0.), make_float3(0, 0, 0.), make_float3(0, 0., 0.));
				allTriangles[i].matIndex = 0;
				TriangleIdx[i] = i;
			}

			// transfering triangles
			for (int i = 0; i < nbTri; i++) { // actually N if everything is ok
				int3 alloc = make_int3(faceVec[i].x, faceVec[i].y, faceVec[i].z);
				int3 nPtr = normalPtr[i];
				int3 tPtr = texCoordPtr[i];
				if (alloc.x > nbTri) {
					alloc.x = 0;
					//exit(2);
				}
				if (alloc.y > nbTri) {
					alloc.y = 0;
					//exit(2);
				}
				if (alloc.z > nbTri) {
					alloc.z = 0;
					//exit(2);
				}
				if (nPtr.x < 0) nPtr.x = normals.size() - nPtr.x - 1;
				if (nPtr.y < 0) nPtr.y = normals.size() - nPtr.y - 1;
				if (nPtr.z < 0) nPtr.z = normals.size() - nPtr.z - 1;

				if (alloc.x < 0) alloc.x = pointsVec.size() - alloc.x - 1;
				if (alloc.y < 0) alloc.y = pointsVec.size() - alloc.y - 1;
				if (alloc.z < 0) alloc.z = pointsVec.size() - alloc.z - 1;

				allTriangles[i] = Triangle(pointsVec[alloc.x], pointsVec[alloc.y], pointsVec[alloc.z]);
				allTriangles[i].matIndex = matIndexes[i];

				if (anyNormals) {
					allTriangles[i].nA = normalize(normals[nPtr.x]);
					allTriangles[i].nB = normalize(normals[nPtr.y]);
					allTriangles[i].nC = normalize(normals[nPtr.z]);
				}
				if (anyTextures) {
					allTriangles[i].tA = texCoord[tPtr.x];
					allTriangles[i].tB = texCoord[tPtr.y];
					allTriangles[i].tC = texCoord[tPtr.z];
				}

				TriangleIdx[i] = i;
			}
			return true;
		}
		else return false;
	}
}