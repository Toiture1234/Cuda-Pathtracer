#include <string>
#include <stdio.h>

#include "envmap.h"

namespace pathtracer {
    inline float Luminance(float r, float g, float b)
    {
        return 0.212671 * r + 0.715160 * g + 0.072169 * b;
    }
    void envMap::buildCDF() {
        float* weights = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = x * 4 + y * width * 4;
                weights[x + y * width] = Luminance(data[idx], data[idx + 1], data[idx + 2]);
            }
        }

        cdf = new float[width * height];
        cdf[0] = weights[0];
        for (int i = 1; i < width * height; i++) {
            cdf[i] = cdf[i - 1] + weights[i];
        }
        sum = cdf[width * height - 1];

        delete[] weights;
    }
    bool envMap::loadMap(const std::string path) {
        if (!load_hdr_float4(&data, &width, &height, path.c_str())) {
            return false;
        }

        buildCDF();
        return true;
    }
    bool envMap::generateCUDAtexture(cudaTextureObject_t* texObj, cudaArray_t* texData) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        size_t spitch = width * sizeof(float4);
        cudaMallocArray(texData, &channelDesc, width, height);

        cudaMemcpy2DToArray(*texData, 0, 0, data, spitch, width * sizeof(float4), height, cudaMemcpyHostToDevice);

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

        cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
        return true;
    }
    bool envMap::generateCUDAtexture_cdf(cudaTextureObject_t* cdfObj, cudaArray_t* cdfData) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        size_t spitch = width * sizeof(float);
        cudaMallocArray(cdfData, &channelDesc, width, height);

        cudaMemcpy2DToArray(*cdfData, 0, 0, cdf, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
        //cudaMemcpyToArray(*texData, 0, 0, imgArray, size.x * size.y * sizeof(float4), cudaMemcpyHostToDevice);
        //cudaMemcpyToArray(*cdfData, 0, 0, cdf, width * height * sizeof(float), cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = *cdfData;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = true;

        cudaCreateTextureObject(cdfObj, &resDesc, &texDesc, NULL);

        return true;;
    }
    void envMap::generateCUDAenvmap(cudaTextureObject_t* texObj, cudaArray_t* texData, cudaTextureObject_t* cdfObj, cudaArray_t* cdfData) {
        generateCUDAtexture(texObj, texData);
        generateCUDAtexture_cdf(cdfObj, cdfData);
    }

    void envMap::transfertToParams(int2* size, float* tot_sum) {
        *size = make_int2(width, height);
        *tot_sum = sum;
    }
}