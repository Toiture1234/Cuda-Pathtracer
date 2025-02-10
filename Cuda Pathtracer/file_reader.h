#pragma once

namespace pathtracer {
	
	//static float3* verticies_host;
	static Triangle* allTriangles;
	static int* TriangleIdx;
	static Material* materialList;

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

	inline bool readMtlFile(int* numOfMaterials, std::string path) { // perform memory allocations inside this, sounds risky
		std::vector<readableMaterial> matList;

		std::string pathC = "assets/models/" + path;
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
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ks") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Ks of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					//current.matProperties.specularAlbedo = values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Kd") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Kd of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					current.matProperties.baseColor = values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ke") {
					float3 values = make_float3(0.f, 0.f, 0.f);

					values.x = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.y = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					values.z = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Ke of " << current.name << " : " << values.x << " " << values.y << " " << values.z << "\n";
					current.matProperties.emissive = values;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "d") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "d of " << current.name << " : " << value << "\n";
					current.matProperties.specTrans = 1. - value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ni") {
					float value = 1.0f;
					value = stof(readUntil(line, tokenEnd + 1, ' ', &tokenEnd));
					std::cout << "IOR of " << current.name << " : " << value << "\n";
					current.matProperties.ior = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Pr") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Pr of " << current.name << " : " << value << "\n";
					current.matProperties.roughness = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Pm") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Pm of " << current.name << " : " << value << "\n";
					current.matProperties.metallic = value;
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "Ps") {
					float value = stof(readUntil(line, tokenEnd + 1, '\n', &tokenEnd));
					std::cout << "Ps of " << current.name << " : " << value << "\n";
					current.matProperties.sheen = value;
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
		}
		return false;
	}

	inline int getMatIndex(int matNb, std::string thisName) {
		for (int i = 0; i < matNb; i++) {
			//std::cout << rMatList[i].name << " / " << thisName << "\n";
			if (rMatList[i].name == thisName) return i;
		}
		return 0;
	}

	inline bool readObjFile(const char* path, int& nbTri, int& nbMat) {
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
		std::vector<int> matIndexes;

		int matNb = 0;
		bool anyNormals = true;
		bool anyTextures = false;

		if (objFile.is_open()) {
			int currentIndex = 0;
			while (getline(objFile, line)) {
				int len = strlen(line.c_str());
				int tokenEnd = 0;

				// to rework
				/*if (readUntil(line, 0, ' ', &tokenEnd) == "f") {
					int3 ptsLink = make_int3(0, 0, 0);
					int3 vtLink = make_int3(0, 0, 0);
					int3 nLink = make_int3(0, 0, 0);

					if (anyNormals) {
						std::string read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.x = stoi(read);

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if(!read.empty())
							vtLink.x = stoi(read);

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							nLink.x = stoi(read);

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.y = stoi(read);

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							vtLink.y = stoi(read);

						read = readUntil(line, tokenEnd + 1, ' ', &tokenEnd);
						if (!read.empty())
							nLink.y = stoi(read);

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							ptsLink.z = stoi(read);

						read = readUntil(line, tokenEnd + 1, '/', &tokenEnd);
						if (!read.empty())
							vtLink.z = stoi(read);

						read = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
						if (!read.empty())
							nLink.z = stoi(read);
					}
					faceVec.push_back(ptsLink);
					normalPtr.push_back(nLink);
					matIndexes.push_back(currentIndex);
					triangleCount++;
				}*/
				if (line[0] == 'f') {
					int x = 0, y = 0, z = 0;
					int vx = 0, vy = 0, vz = 0;
					int tx = 0, ty = 0, tz = 0; // use later (maybe)

					int stage = 0;
					std::string buffer;
					triangleCount++;

					for (int i = 2; i < len + 1; i++) {
						char c = line[i];
						if (c == ' ' || c == '\0' || c == '/') {
							switch (stage) {
							case 0:
								if (buffer.size() != 0)
									x = stoi(buffer);
								stage++;
								break;
							case 1:
								if (buffer.size() != 0)
									tx = stoi(buffer);
								stage++;
								break;
							case 2:
								if (buffer.size() != 0)
									vx = stoi(buffer);
								stage++;
								break;
							case 3:
								if (buffer.size() != 0)
									y = stoi(buffer);
								stage++;
								break;
							case 4:
								if (buffer.size() != 0)
									ty = stoi(buffer);
								stage++;
								break;
							case 5:
								if (buffer.size() != 0)
									vy = stoi(buffer);
								stage++;
								break;
							case 6:
								if (buffer.size() != 0)
									z = stoi(buffer);
								stage++;
								break;
							case 7:
								if (buffer.size() != 0)
									tz = stoi(buffer);
								stage++;
								break;
							case 8:
								if (buffer.size() != 0)
									vz = stoi(buffer);
								stage = -1; // stopper
								break;
							}
							buffer = "";
						}
						else buffer.push_back(c);
						if (stage == -1) break;
					}
					faceVec.push_back(make_int3(x - 1, y - 1, z - 1));
					normalPtr.push_back(make_int3(vx - 1, vy - 1, vz - 1));
					matIndexes.push_back(currentIndex);
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
				}
				// old version
				/*
				else if (line[0] == 'v' && line[1] == ' ') {
					float x = 0, y = 0, z = 0;
					int stage = 0;
					std::string buffer;
					verticesCount++;

					// read line
					for (int i = 2; i < len + 1; i++) {
						char c = line[i];
						if (c == ' ' || c == '\0') {
							switch (stage) {
							case 0:
								x = stof(buffer);
								stage = 1;
								break;
							case 1:
								y = stof(buffer);
								stage = 2;
								break;
							case 2:
								z = stof(buffer);
								break;

							}
							buffer = "";
						}
						buffer.push_back(c);
					}
					pointsVec.push_back(make_float3(x, y, z) * size + position);
				}
				*/
				/*else if (line[0] == 'v' && line[1] == 'n') {
					float vx = 0, vy = 0, vz = 0;
					int stage = 0;
					std::string buffer;

					// read line
					for (int i = 3; i < len + 1; i++) {
						char c = line[i];
						if (c == ' ' || c == '\0') {
							switch (stage) {
							case 0:
								vx = stof(buffer);
								stage = 1;
								break;
							case 1:
								vy = stof(buffer);
								stage = 2;
								break;
							case 2:
								vz = stof(buffer);
								break;

							}
							buffer = "";
						}
						buffer.push_back(c);
					}
					normals.push_back(make_float3(vx, vy, vz));
				}*/
				else if (readUntil(line, 0, ' ', &tokenEnd) == "usemtl") {
					std::string thisName = readUntil(line, 7, '\n', &tokenEnd);
					currentIndex = getMatIndex(matNb, thisName);
					std::cout << "Mat index is now " << currentIndex << "\n";
					std::cout << thisName << "\n";
				}
				else if (readUntil(line, 0, ' ', &tokenEnd) == "mtllib") { // consider this being before usemtl
					std::string mtlName = readUntil(line, tokenEnd + 1, '\n', &tokenEnd);
					if (!readMtlFile(&matNb, mtlName)) return false;
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
				allTriangles[i].nA = normalize(normals[nPtr.x]);
				allTriangles[i].nB = normalize(normals[nPtr.y]);
				allTriangles[i].nC = normalize(normals[nPtr.z]);
				allTriangles[i].matIndex = matIndexes[i];

				TriangleIdx[i] = i;
			}
			return true;
		}
		else return false;
	}
}