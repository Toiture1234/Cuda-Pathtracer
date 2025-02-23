#pragma once

namespace pathtracer {
	inline bool genTexture(cudaTextureObject_t* tex, cudaArray_t* texData, std::string path) {
		sf::Image img;;
		if (!img.loadFromFile(path)) return false;

		sf::Vector2u size = img.getSize();

		// array building
		float* imgArray = new float[size.x * size.y * 4];
		for (int x = 0; x < size.x; x++) {
			for (int y = 0; y < size.y; y++) {
				sf::Color pxCol = img.getPixel(x, size.y - y - 1);

				int idx = x + y * size.x;
				imgArray[idx * 4] = (float)pxCol.r / 255.0f;
				imgArray[idx * 4 + 1] = (float)pxCol.g / 255.0f;
				imgArray[idx * 4 + 2] = (float)pxCol.b / 255.0f;
				imgArray[idx * 4 + 3] = (float)pxCol.a / 255.0f;
			}
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		size_t spitch = size.x * sizeof(float4);
		cudaMallocArray(texData, &channelDesc, size.x, size.y);

		cudaMemcpy2DToArray(*texData, 0, 0, imgArray, spitch, size.x * sizeof(float4), size.y, cudaMemcpyHostToDevice);
		//cudaMemcpyToArray(*texData, 0, 0, imgArray, size.x * size.y * sizeof(float4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = *texData;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);

		return true;;
	}
	inline bool genTexture_float(cudaTextureObject_t* tex, cudaArray_t* texData, std::string path) {
		sf::Image img;;
		if (!img.loadFromFile(path)) return false;

		sf::Vector2u size = img.getSize();

		// array building
		float* imgArray = new float[size.x * size.y];
		for (int x = 0; x < size.x; x++) {
			for (int y = 0; y < size.y; y++) {
				sf::Color pxCol = img.getPixel(x, size.y - y - 1);

				int idx = x + y * size.x;
				imgArray[idx] = (float)pxCol.r / 255.0f;
			}
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		size_t spitch = size.x * sizeof(float);
		cudaMallocArray(texData, &channelDesc, size.x, size.y);

		cudaMemcpy2DToArray(*texData, 0, 0, imgArray, spitch, size.x * sizeof(float), size.y, cudaMemcpyHostToDevice);
		//cudaMemcpyToArray(*texData, 0, 0, imgArray, size.x * size.y * sizeof(float4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = *texData;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);

		return true;;
	}
	inline bool genTextureFromHDR(cudaTextureObject_t* tex, cudaArray_t* texData, std::string path) {

		// array building
		float* imgArray;
		sf::Vector2u size;
		if (!load_hdr_float4(&imgArray, &size.x, &size.y, path.c_str())) {
			return false;
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		size_t spitch = size.x * sizeof(float4);
		cudaMallocArray(texData, &channelDesc, size.x, size.y);

		cudaMemcpy2DToArray(*texData, 0, 0, imgArray, spitch, size.x * sizeof(float4), size.y, cudaMemcpyHostToDevice);
		//cudaMemcpyToArray(*texData, 0, 0, imgArray, size.x * size.y * sizeof(float4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = *texData;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);

		return true;;
	}
}