#include "../pathtracer.cuh"

namespace pathtracer {
    // HG
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

    // draine-hg blended
    __device__ inline float4 getDraineParams(float dropletSize) { // returns gHG, gDraine, alpha and wDraine
        float gHG = expf(-0.0990567f / (dropletSize - 1.67154f));
        float gDraine = expf(-2.20679f / (dropletSize + 3.91029f) - 0.428934f);
        float alpha = expf(3.62489f - 8.29288f / (dropletSize + 5.52825));
        float wDraine = expf(-0.599085f / (dropletSize - 0.641583f) - 0.665888f);
        return make_float4(gHG, gDraine, alpha, wDraine);
    }


    /*
     * SPDX-FileCopyrightText: Copyright (c) <2023> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     * SPDX-License-Identifier: MIT
     *
     * Permission is hereby granted, free of charge, to any person obtaining a
     * copy of this software and associated documentation files (the "Software"),
     * to deal in the Software without restriction, including without limitation
     * the rights to use, copy, modify, merge, publish, distribute, sublicense,
     * and/or sell copies of the Software, and to permit persons to whom the
     * Software is furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in
     * all copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
     * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
     * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     * DEALINGS IN THE SOFTWARE.
     */

    // [Jendersie and d'Eon 2023]
    //   SIGGRAPH 2023 Talks
    //   https://doi.org/10.1145/3587421.3595409

    // EVAL and SAMPLE for the Draine (and therefore Cornette-Shanks) phase function
    //   g = HG shape parameter
    //   a = "alpha" shape parameter

    // Warning: these functions don't special case isotropic scattering and can numerically fail for certain inputs

    // eval:
    //   u = dot(prev_dir, next_dir)
    __device__ inline float evalDraine(float u, float g, float a)
    {
        return ((1.f - g * g) * (1.f + a * u * u)) / (4.f * (1.f + (a * (1.f + 2.f * g * g)) / 3.f) * PI * powf(1.f + g * g - 2.f * g * u, 1.5f));
    }

    // sample: (sample an exact deflection cosine)
    //   xi = a uniform random real in [0,1]
    __device__ inline float sampleDraineCos(float xi, float g, float a)
    {
        const float g2 = g * g;
        const float g3 = g * g2;
        const float g4 = g2 * g2;
        const float g6 = g2 * g4;
        const float pgp1_2 = (1.f + g2) * (1.f + g2);
        const float T1 = (-1.f + g2) * (4.f * g2 + a * pgp1_2);
        const float T1a = -a + a * g4;
        const float T1a3 = T1a * T1a * T1a;
        const float T2 = -1296.f * (-1.f + g2) * (a - a * g2) * (T1a) * (4.f * g2 + a * pgp1_2);
        const float T3 = 3.f * g2 * (1.f + g * (-1.f + 2.f * xi)) + a * (2.f + g2 + g3 * (1.f + 2.f * g2) * (-1.f + 2.f * xi));
        const float T4a = 432.f * T1a3 + T2 + 432.f * (a - a * g2) * T3 * T3;
        const float T4b = -144.f * a * g2 + 288.f * a * g4 - 144.f * a * g6;
        const float T4b3 = T4b * T4b * T4b;
        const float T4 = T4a + sqrtf(-4.f * T4b3 + T4a * T4a);
        const float T4p3 = powf(T4, 1.0f / 3.0f);
        const float T6 = (2 * T1a + (48.f * powf(2, 1.0f / 3.0f) *
            (-(a * g2) + 2.f * a * g4 - a * g6)) / T4p3 + T4p3 / (3.f * powf(2.f, 1.0f / 3.0f))) / (a - a * g2);
        const float T5 = 6.f * (1.f + g2) + T6;
        return (1.f + g2 - powf(-0.5f * sqrtf(T5) + sqrtf(6.f * (1.f + g2) - (8.f * T3) / (a * (-1.f + g2) * sqrtf(T5)) - T6) / 2.f, 2.f)) / (2.f * g);
    }

    __device__ inline float evalDraineHG(float mu, float dS) {
        float4 params = getDraineParams(dS);
        return mix(evalHG(mu, params.x), evalDraine(mu, params.y, params.z), params.w);
    }

    __device__ inline float3 sampleDraineHG(Ray ray, float dS, Rand_state& state) {
        float4 params = getDraineParams(dS);

        if (randC(&state) < params.w) {// do draine
            float xi = randC(&state);
            float mu = sampleDraineCos(xi, params.y, params.z);

            float phi = TWO_PI * randC(&state);
            float sinTheta = sqrtf(fmaxf(0.f, 1.f - mu * mu));

            float3 T, B;
            Onb(ray.d, T, B);
            return normalize(sinTheta * cosf(phi) * T + sinTheta * sinf(phi) * B + mu * ray.d);
        }
        else {
            return sampleHG(ray, params.x, state);
        }
    }
}