#pragma once

/*
 * MIT License
 *
 * Copyright(c) 2019 Asif Ali
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

namespace pathtracer {
    namespace Disney {
        __device__ inline float Luminance(float3 c)
        {
            return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
        }

        // sampling
        __device__ inline float SchlickFresnel(float u)
        {
            float m = clamp(1.0 - u, 0.0, 1.0);
            float m2 = m * m;
            return m2 * m2 * m;
        }
        __device__ inline float GTR1(float NDotH, float a)
        {
            if (a >= 1.0)
                return INV_PI;
            float a2 = a * a;
            float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
            return (a2 - 1.0) / (PI * log(a2) * t);
        }

        __device__ inline float3 SampleGTR1(float rgh, float r1, float r2)
        {
            float a = fmaxf(0.001, rgh);
            float a2 = a * a;

            float phi = r1 * TWO_PI;

            float cosTheta = sqrtf((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
            float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        }

        __device__ inline float GTR2(float NDotH, float a)
        {
            float a2 = a * a;
            float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
            return a2 / (PI * t * t);
        }

        __device__ inline float3 SampleGTR2(float rgh, float r1, float r2)
        {
            float a = fmaxf(0.001, rgh);

            float phi = r1 * TWO_PI;

            float cosTheta = sqrtf((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
            float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        }

        __device__ inline float3 SampleGGXVNDF(float3 V, float ax, float ay, float r1, float r2)
        {
            float3 Vh = normalize(make_float3(ax * V.x, ay * V.y, V.z));

            float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
            float3 T1 = lensq > 0.0 ? make_float3(-Vh.y, Vh.x, 0.0) * rsqrtf(lensq) : make_float3(1, 0, 0);
            float3 T2 = cross(Vh, T1);

            float r = sqrtf(r1);
            float phi = 2.0 * PI * r2;
            float t1 = r * cosf(phi);
            float t2 = r * sinf(phi);
            float s = 0.5 * (1.0 + Vh.z);
            t2 = (1.0 - s) * sqrtf(1.0 - t1 * t1) + s * t2;

            float3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

            return normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0, Nh.z)));
        }
        __device__ inline float SmithG(float NDotV, float alphaG)
        {
            float a = alphaG * alphaG;
            float b = NDotV * NDotV;
            return (2.0f * NDotV) / (NDotV + sqrtf(a + b - a * b));
        }
        __device__ inline float DielectricFresnel(float cosThetaI, float eta)
        {
            float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

            // Total internal reflection
            if (sinThetaTSq > 1.0f)
                return 1.0f;

            float cosThetaT = sqrtf(max(1.0 - sinThetaTSq, 0.0));

            float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
            float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

            return 0.5f * (rs * rs + rp * rp);
        }
        __device__ inline float3 CosineSampleHemisphere(float r1, float r2)
        {
            float3 dir;
            float r = sqrtf(r1);
            float phi = TWO_PI * r2;
            dir.x = r * cosf(phi);
            dir.y = r * sinf(phi);
            dir.z = sqrtf(fmaxf(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));
            return dir;
        }



        // Disney
        __device__ inline float FresnelMix(Material mat, float eta, float VDotH)
        {
            float metallicFresnel = SchlickFresnel(VDotH);
            float dielectricFresnel = DielectricFresnel(VDotH, eta);
            return mix(dielectricFresnel, metallicFresnel, mat.metallic);
        }

        __device__ inline float3 EvalDiffuse(Material mat, float3 Csheen, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0f;
            if (L.z <= 0.0f)
                return make_float3(0.f, 0.f, 0.f);

            // Diffuse
            float FL = SchlickFresnel(L.z);
            float FV = SchlickFresnel(V.z);
            float FH = SchlickFresnel(dot(L, H));
            float Fd90 = 0.5 + 2.0 * dot(L, H) * dot(L, H) * mat.roughness;
            float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

            // Fake Subsurface TODO: Replace with volumetric scattering
            float Fss90 = dot(L, H) * dot(L, H) * mat.roughness;
            float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
            float ss = 1.25f * (Fss * (1.0f / (L.z + V.z) - 0.5f) + 0.5f);

            // Sheen
            float3 Fsheen = FH * mat.sheen * Csheen;

            pdf = L.z * INV_PI;
            return (INV_PI * mix(Fd, ss, mat.subsurface) * mat.baseColor + Fsheen) * (1.0f - mat.metallic) * (1.0f - mat.specTrans);
        }

        __device__ inline float3 EvalSpecReflection(Material mat, float eta, float3 specCol, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0f;
            if (L.z <= 0.0f)
                return make_float3(0.f, 0.f, 0.f);

            float FM = FresnelMix(mat, eta, dot(L, H));
            float3 F = mix(specCol, make_float3(1.0f, 1.0f, 1.0f), FM);
            float D = GTR2(H.z, mat.roughness);
            float G1 = SmithG(abs(V.z), mat.roughness);
            float G2 = G1 * SmithG(abs(L.z), mat.roughness);
            float jacobian = 1.0 / (4.0 * dot(V, H));

            pdf = G1 * fmaxf(0.0, dot(V, H)) * D * jacobian / V.z;
            return F * D * G2 / (4.0 * L.z * V.z);
        }

        __device__ inline float3 EvalSpecRefraction(Material mat, float eta, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0f;
            if (L.z >= 0.0f)
                return  make_float3(0.f, 0.f, 0.f);

            float F = DielectricFresnel(abs(dot(V, H)), eta);
            float D = GTR2(H.z, mat.roughness);
            float denom = dot(L, H) + dot(V, H) * eta;
            denom *= denom;
            float G1 = SmithG(abs(V.z), mat.roughness);
            float G2 = G1 * SmithG(abs(L.z), mat.roughness);
            float jacobian = abs(dot(L, H)) / denom;

            pdf = G1 * fmaxf(0.0, dot(V, H)) * D * jacobian / V.z;

            float3 specColor = pow3(mat.baseColor, 0.5f);
            return specColor * (1.0 - mat.metallic) * mat.specTrans * (1.0 - F) * D * G2 * abs(dot(V, H)) * abs(dot(L, H)) * eta * eta / (denom * abs(L.z) * abs(V.z));
        }
        __device__ inline float3 EvalClearcoat(Material mat, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0f;
            if (L.z <= 0.0f)
                return make_float3(0.f, 0.f, 0.f);

            float FH = DielectricFresnel(dot(V, H), 1.0f / 1.5f);
            float F = mix(0.04f, 1.0f, FH);
            float D = GTR1(H.z, mat.clearcoatRoughness);
            float G = SmithG(L.z, 0.25f) * SmithG(V.z, 0.25f);
            float jacobian = 1.0f / (4.0f * dot(V, H));

            pdf = D * H.z * jacobian;
            return make_float3(0.25f, 0.25f, 0.25f) * mat.clearcoat * F * D * G / (4.0f * L.z * V.z);
        }
        __device__ inline void GetSpecColor(Material mat, float eta, float3& specCol, float3& sheenCol)
        {
            float F0 = (1.0 - eta) / (1.0f + eta);
            specCol = mix(F0 * F0 * make_float3(1.0f, 1.0f, 1.0f), mat.baseColor, mat.metallic);
            sheenCol = make_float3(1.f,1.f,1.f);
        }

        __device__ inline void GetLobeProbabilities(Material mat, float eta, float3 specCol, float approxFresnel, float& diffuseWt, float& specReflectWt, float& specRefractWt, float& clearcoatWt)
        {
            diffuseWt = Luminance(mat.baseColor) * (1.0f - mat.metallic) * (1.0f - mat.specTrans);
            specReflectWt = Luminance(mix(specCol, make_float3(1.0f, 1.0f, 1.0f), approxFresnel));
            specRefractWt = (1.0 - approxFresnel) * (1.0f - mat.metallic) * mat.specTrans * Luminance(mat.baseColor);
            clearcoatWt = mat.clearcoat * (1.0f - mat.metallic);
            float totalWt_inv = 1. / (diffuseWt + specReflectWt + specRefractWt + clearcoatWt);

            diffuseWt *= totalWt_inv;
            specReflectWt *= totalWt_inv;
            specRefractWt *= totalWt_inv;
            clearcoatWt *= totalWt_inv;
        }

        __device__ inline float3 DisneySample(Hit& info, float3 V, float3 N, float3& L, float& pdf, Rand_state& state, float3& rayO)
        {
            pdf = 0.0;
            float3 f = make_float3(0.0f, 0.0f, 0.0f);
            float eta = !info.isInside ? 1. / info.mat.ior : info.mat.ior;

            // anisotropy
            float aspect = rsqrtf(1.f - 0.9f * info.mat.anisotropic);
            float ax = info.mat.roughness * info.mat.roughness * aspect;
            float ay = info.mat.roughness * info.mat.roughness / aspect;

            float r1 = randC(&state);
            float r2 = randC(&state);

            float3 T, B;
            Onb(N, T, B);
            V = ToLocal(T, B, N, V); // NDotL = L.z; NDotV = V.z; NDotH = H.z

            // Specular and sheen color
            float3 specCol, sheenCol;
            GetSpecColor(info.mat, eta, specCol, sheenCol);

            // Lobe weights
            float diffuseWt, specReflectWt, specRefractWt, clearcoatWt;
            float approxFresnel = FresnelMix(info.mat, eta, V.z);
            GetLobeProbabilities(info.mat, eta, specCol, approxFresnel, diffuseWt, specReflectWt, specRefractWt, clearcoatWt);

            // CDF for picking a lobe
            float cdf[4];
            cdf[0] = diffuseWt;
            cdf[1] = cdf[0] + specReflectWt;
            cdf[2] = cdf[1] + specRefractWt;
            cdf[3] = cdf[2] + clearcoatWt;

            float r3 = randC(&state);

            if (r3 < cdf[0]) // Diffuse Reflection Lobe
            {
                L = CosineSampleHemisphere(r1, r2);

                float3 H = normalize(L + V);

                f = EvalDiffuse(info.mat, sheenCol, V, L, H, pdf);
                pdf *= diffuseWt;
            }
            else if (r3 < cdf[1]) // Specular Reflection Lobe
            {
                float3 H = SampleGGXVNDF(V, ax, ay, r1, r2);
                H *= sign(H.z);

                L = normalize(reflect(V * -1.0f, H));

                f = EvalSpecReflection(info.mat, eta, specCol, V, L, H, pdf);
                pdf *= specReflectWt;
            }
            else if (r3 < cdf[2]) // Specular Refraction Lobe
            {
                float3 H = SampleGGXVNDF(V, ax, ay, r1, r2);
                H *= sign(H.z);

                L = normalize(refract(V * -1.0f, H, eta));

                f = EvalSpecRefraction(info.mat, eta, V, L, H, pdf);
                pdf = specRefractWt;

                info.isInside = !info.isInside;
            }
            else // Clearcoat Lobe
            {
                float3 H = SampleGTR1(info.mat.clearcoatRoughness, r1, r2);
                H *= sign(H.z);

                L = normalize(reflect(V * -1.0f, H));

                f = EvalClearcoat(info.mat, V, L, H, pdf);
                pdf *= clearcoatWt;
            }

            L = ToWorld(T, B, N, L);
            return f * abs(dot(N, L));
        }

        __device__ inline float3 DisneyEval(Hit info, float3 V, float3 N, float3 L, float& bsdfPdf)
        {
            float eta = !info.isInside ? 1. / info.mat.ior : info.mat.ior;
            bsdfPdf = 0.0f;
            float3 f = make_float3(0.f, 0.f, 0.f);

            float3 T, B;
            Onb(N, T, B);
            V = ToLocal(T, B, N, V); // NDotL = L.z; NDotV = V.z; NDotH = H.z
            L = ToLocal(T, B, N, L);

            float3 H;
            if (L.z > 0.0)
                H = normalize(L + V);
            else
                H = normalize(L + V * eta);

            H *= sign(H.z);

            // Specular and sheen color
            float3 specCol, sheenCol;
            GetSpecColor(info.mat, eta, specCol, sheenCol);

            // Lobe weights
            float diffuseWt, specReflectWt, specRefractWt, clearcoatWt;
            float fresnel = FresnelMix(info.mat, eta, dot(V, H));
            GetLobeProbabilities(info.mat, eta, specCol, fresnel, diffuseWt, specReflectWt, specRefractWt, clearcoatWt);

            float pdf;

            // Diffuse
            if (diffuseWt > 0.0 && L.z > 0.0)
            {
                f += EvalDiffuse(info.mat, sheenCol, V, L, H, pdf);
                bsdfPdf += pdf * diffuseWt;
            }

            // Specular Reflection
            if (specReflectWt > 0.0 && L.z > 0.0 && V.z > 0.0)
            {
                f += EvalSpecReflection(info.mat, eta, specCol, V, L, H, pdf);
                bsdfPdf += pdf * specReflectWt;
            }

            // Specular Refraction
            if (specRefractWt > 0.0 && L.z < 0.0)
            {
                f += EvalSpecRefraction(info.mat, eta, V, L, H, pdf);
                bsdfPdf += pdf * specRefractWt;
            }

            // Clearcoat
            if (clearcoatWt > 0.0 && L.z > 0.0 && V.z > 0.0)
            {
                f += EvalClearcoat(info.mat, V, L, H, pdf);
                bsdfPdf += pdf * clearcoatWt;
            }

            return f * abs(L.z);
        }
    }

}