#include "../pathtracer.cuh"

namespace pathtracer {
    __device__ inline float evalHG(float mu, float g) {
        return INV_4_PI * (1. - g * g) / (powf(1. + g * g - 2.0f * g * mu, 1.5f));
        
    }
    __device__ inline float3 sampleHG(Ray ray, float g, Rand_state& state) {
        float xi = randC(&state);
        float t = (1.f - g * g) / (1.f - g + 2.0f * g * xi);
        float mu =  (0.5f / g) * ((1.f + g * g) - t * t);

        float phi = TWO_PI * randC(&state);
        float sinTheta = sqrtf(fmaxf(0.f, 1.f - mu * mu));

        float3 T, B;
        Onb(ray.d, T, B);
        return normalize(sinTheta * cosf(phi) * T + sinTheta * sinf(phi) * B + mu * ray.d);
    }
}