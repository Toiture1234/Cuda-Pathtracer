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
        __device__ inline float SchlickWeight(float u)
        {
            float m = clamp(1.0f - u, 0.0f, 1.0f);
            float m2 = m * m;
            return m2 * m2 * m;
        }
        __device__ inline void TintColors(Material mat, float eta, float& F0, float3& Csheen, float3& Cspec0)
        {
            float lum = Luminance(mat.baseColor);
            float3 ctint = lum > 0.0 ? mat.baseColor / lum : make_float3(1.f, 1.f, 1.f);

            F0 = (1.0 - eta) / (1.0 + eta);
            F0 *= F0;

            Cspec0 = F0 * make_float3(1.f, 1.f, 1.f);
            Csheen = make_float3(1.f, 1.f, 1.f);
        }
        __device__ inline float GTR2Aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
        {
            float a = HDotX / ax;
            float b = HDotY / ay;
            float c = a * a + b * b + NDotH * NDotH;
            return 1.0 / (PI * ax * ay * c * c);
        }
        __device__ inline float SmithG_2(float NDotV, float alphaG)
        {
            float a = alphaG * alphaG;
            float b = NDotV * NDotV;
            return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
        }
        __device__ inline float SmithGAniso(float NDotV, float VDotX, float VDotY, float ax, float ay)
        {
            float a = VDotX * ax;
            float b = VDotY * ay;
            float c = NDotV;
            return (2.0 * NDotV) / (NDotV + sqrt(a * a + b * b + c * c));
        }
        __device__ inline float GTR1_2(float NDotH, float a)
        {
            if (a >= 1.0)
                return INV_PI;
            float a2 = a * a;
            float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
            return (a2 - 1.0) / (PI * log(a2) * t);
        }

        __device__ inline float3 SampleGTR1_2(float rgh, float r1, float r2)
        {
            float a = fmaxf(0.001, rgh);
            float a2 = a * a;

            float phi = r1 * TWO_PI;

            float cosTheta = sqrtf((1.0 - pow(a2, 1.0 - r2)) / (1.0 - a2));
            float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        }

        // Disney
        __device__ inline float3 EvalDisneyDiffuse(Material mat, float3 Csheen, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0;
            if (L.z <= 0.0)
                return make_float3(0.f, 0.f, 0.f);

            float LDotH = dot(L, H);

            float Rr = 2.0f * mat.roughness * LDotH * LDotH;

            // Diffuse
            float FL = SchlickWeight(L.z);
            float FV = SchlickWeight(V.z);
            float Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.0f));
            float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);

            // Fake subsurface
            float Fss90 = 0.5 * Rr;
            float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
            float ss = 1.25f * (Fss * (1.0f / (L.z + V.z) - 0.5f) + 0.5f);

            // Sheen
            float FH = SchlickWeight(LDotH);
            float3 Fsheen = FH * mat.sheen * Csheen;

            pdf = L.z * INV_PI;
            return INV_PI * mat.baseColor * mix(Fd + Fretro, ss, mat.subsurface) + Fsheen;
        }

        __device__ inline float3 EvalMicrofacetReflection(Material mat, float3 V, float3 L, float3 H, float3 F, float ax, float ay, float& pdf)
        {
            pdf = 0.0;
            if (L.z <= 0.0)
                return make_float3(0.f, 0.f, 0.f);

            float D = GTR2Aniso(H.z, H.x, H.y, ax, ay);
            float G1 = SmithGAniso(abs(V.z), V.x, V.y, ax, ay);
            float G2 = G1 * SmithGAniso(abs(L.z), L.x, L.y, ax, ay);

            pdf = G1 * D / (4.0 * V.z);
            return F * D * G2 / (4.0 * L.z * V.z);
        }

        __device__ inline float3 EvalMicrofacetRefraction(Material mat, float eta, float3 V, float3 L, float3 H, float3 F, float ax, float ay, float& pdf)
        {
            pdf = 0.0;
            if (L.z >= 0.0)
                return make_float3(0.f, 0.f, 0.f);

            float LDotH = dot(L, H);
            float VDotH = dot(V, H);

            float D = GTR2Aniso(H.z, H.x, H.y, ax, ay);
            float G1 = SmithGAniso(abs(V.z), V.x, V.y, ax, ay);
            float G2 = G1 * SmithGAniso(abs(L.z), L.x, L.y, ax, ay);
            float denom = LDotH + VDotH * eta;
            denom *= denom;
            float eta2 = eta * eta;
            float jacobian = abs(LDotH) / denom;

            pdf = G1 * fmaxf(0.0, VDotH) * D * jacobian / V.z;
            return pow3(mat.baseColor, 0.5f) * (1.0f - F) * D * G2 * abs(VDotH) * jacobian * eta2 / abs(L.z * V.z);
        }

        __device__ inline float3 EvalClearcoat_2(Material mat, float3 V, float3 L, float3 H, float& pdf)
        {
            pdf = 0.0;
            if (L.z <= 0.0)
                return make_float3(0.f, 0.f, 0.f);

            float VDotH = dot(V, H);

            float F = mix(0.04, 1.0, SchlickWeight(VDotH));
            float D = GTR1_2(H.z, mat.clearcoatRoughness);
            float G = SmithG_2(L.z, 0.25) * SmithG_2(V.z, 0.25f);
            float jacobian = 1.0 / (4.0 * VDotH);

            pdf = D * H.z * jacobian;
            return make_float3(F,F,F) * D * G;
        }
        __device__ inline float3 DisneyEval_2(Hit info, float3 V, float3 N, float3 L, float& pdf)
        {
            pdf = 0.0;
            float eta = info.isInside ? info.mat.ior : 1.f / info.mat.ior;
            float3 f = make_float3(0.f, 0.f, 0.f);

            // TODO: Tangent and bitangent should be calculated from mesh (provided, the mesh has proper uvs)
            float3 T, B;
            Onb(N, T, B);

            // Transform to shading space to simplify operations (NDotL = L.z; NDotV = V.z; NDotH = H.z)
            V = ToLocal(T, B, N, V);
            L = ToLocal(T, B, N, L);

            float3 H;
            if (L.z > 0.0)
                H = normalize(L + V);
            else
                H = normalize(L + V * eta);

            if (H.z < 0.0)
                H = -1.f * H;

            // Tint colors
            float3 Csheen, Cspec0;
            float F0;
            TintColors(info.mat, eta, F0, Csheen, Cspec0);

            // Model weights
            float dielectricWt = (1.0f - info.mat.metallic) * (1.0f - info.mat.specTrans);
            float metalWt = info.mat.metallic;
            float glassWt = (1.0f - info.mat.metallic) * info.mat.specTrans;

            // Lobe probabilities
            float schlickWt = SchlickWeight(V.z);

            float diffPr = dielectricWt * Luminance(info.mat.baseColor);
            float dielectricPr = dielectricWt * Luminance(mix(Cspec0, make_float3(1.f, 1.f, 1.f), schlickWt));
            float metalPr = metalWt * Luminance(mix(info.mat.baseColor, make_float3(1.f, 1.f, 1.f), schlickWt));
            float glassPr = glassWt;
            float clearCtPr = 0.25f * info.mat.clearcoat;

            // Normalize probabilities
            float invTotalWt = 1.0f / (diffPr + dielectricPr + metalPr + glassPr + clearCtPr);
            diffPr *= invTotalWt;
            dielectricPr *= invTotalWt;
            metalPr *= invTotalWt;
            glassPr *= invTotalWt;
            clearCtPr *= invTotalWt;

            bool reflect = L.z * V.z > 0;

            float tmpPdf = 0.0;
            float VDotH = abs(dot(V, H));

            // Diffuse
            if (diffPr > 0.0 && reflect)
            {
                f += EvalDisneyDiffuse(info.mat, Csheen, V, L, H, tmpPdf) * dielectricWt;
                pdf += tmpPdf * diffPr;
            }

            // Dielectric Reflection
            if (dielectricPr > 0.0 && reflect)
            {
                // Normalize for interpolating based on Cspec0
                float F = (DielectricFresnel(VDotH, 1.0 / info.mat.ior) - F0) / (1.0 - F0);

                f += EvalMicrofacetReflection(info.mat, V, L, H, mix(Cspec0, make_float3(1.f, 1.f, 1.f), F), info.mat.roughness, info.mat.roughness, tmpPdf) * dielectricWt;
                pdf += tmpPdf * dielectricPr;
            }

            // Metallic Reflection
            if (metalPr > 0.0 && reflect)
            {
                // Tinted to base color
                float3 F = mix(info.mat.baseColor, make_float3(1.f, 1.f, 1.f), SchlickWeight(VDotH));

                f += EvalMicrofacetReflection(info.mat, V, L, H, F, info.mat.roughness, info.mat.roughness, tmpPdf) * metalWt;
                pdf += tmpPdf * metalPr;
            }

            // Glass/Specular BSDF
            if (glassPr > 0.0)
            {
                // Dielectric fresnel (achromatic)
                float F = DielectricFresnel(VDotH, eta);

                if (reflect)
                {
                    f += EvalMicrofacetReflection(info.mat, V, L, H, make_float3(F, F, F), info.mat.roughness, info.mat.roughness, tmpPdf) * glassWt;
                    pdf += tmpPdf * glassPr * F;
                }
                else
                {
                    f += EvalMicrofacetRefraction(info.mat, eta, V, L, H, make_float3(F, F, F), info.mat.roughness, info.mat.roughness, tmpPdf) * glassWt;
                    pdf += tmpPdf * glassPr * (1.0f - F);
                }
            }

            // Clearcoat
            if (clearCtPr > 0.0 && reflect)
            {
                f += EvalClearcoat_2(info.mat, V, L, H, tmpPdf) * 0.25f * info.mat.clearcoat;
                pdf += tmpPdf * clearCtPr;
            }

            return f * abs(L.z);
        }
        __device__ inline float3 DisneySample_2(Hit info, float3 V, float3 N, float3& L, float& pdf, Rand_state& state)
        {
            pdf = 0.0;
            float eta = info.isInside ? info.mat.ior : 1.f / info.mat.ior;

            float r1 = randC(&state);
            float r2 = randC(&state);

            // TODO: Tangent and bitangent should be calculated from mesh (provided, the mesh has proper uvs)
            float3 T, B;
            Onb(N, T, B);

            // Transform to shading space to simplify operations (NDotL = L.z; NDotV = V.z; NDotH = H.z)
            V = ToLocal(T, B, N, V);

            // Tint colors
            float3 Csheen, Cspec0;
            float F0;
            TintColors(info.mat, eta, F0, Csheen, Cspec0);

            // Model weights
            float dielectricWt = (1.0 - info.mat.metallic) * (1.0 - info.mat.specTrans);
            float metalWt = info.mat.metallic;
            float glassWt = (1.0 - info.mat.metallic) * info.mat.specTrans;

            // Lobe probabilities
            float schlickWt = SchlickWeight(V.z);

            float diffPr = dielectricWt * Luminance(info.mat.baseColor);
            float dielectricPr = dielectricWt * Luminance(mix(Cspec0, make_float3(1.f, 1.f, 1.f), schlickWt));
            float metalPr = metalWt * Luminance(mix(info.mat.baseColor, make_float3(1.f, 1.f, 1.f), schlickWt));
            float glassPr = glassWt;
            float clearCtPr = 0.25 * info.mat.clearcoat;

            // Normalize probabilities
            float invTotalWt = 1.0 / (diffPr + dielectricPr + metalPr + glassPr + clearCtPr);
            diffPr *= invTotalWt;
            dielectricPr *= invTotalWt;
            metalPr *= invTotalWt;
            glassPr *= invTotalWt;
            clearCtPr *= invTotalWt;

            // CDF of the sampling probabilities
            float cdf[5];
            cdf[0] = diffPr;
            cdf[1] = cdf[0] + dielectricPr;
            cdf[2] = cdf[1] + metalPr;
            cdf[3] = cdf[2] + glassPr;
            cdf[4] = cdf[3] + clearCtPr;

            // Sample a lobe based on its importance
            float r3 = randC(&state);

            if (r3 < cdf[0]) // Diffuse
            {
                L = CosineSampleHemisphere(r1, r2);
            }
            else if (r3 < cdf[2]) // Dielectric + Metallic reflection
            {
                float3 H = SampleGGXVNDF(V, info.mat.roughness, info.mat.roughness, r1, r2);

                if (H.z < 0.0)
                    H = -1.f * H;

                L = normalize(reflect(-1.f * V, H));
            }
            else if (r3 < cdf[3]) // Glass
            {
                float3 H = SampleGGXVNDF(V, info.mat.roughness, info.mat.roughness, r1, r2);
                float F = DielectricFresnel(abs(dot(V, H)), eta);

                if (H.z < 0.0)
                    H = -1.f * H;

                // Rescale random number for reuse
                r3 = (r3 - cdf[2]) / (cdf[3] - cdf[2]);

                // Reflection
                if (r3 < F)
                {
                    L = normalize(reflect(-1.f * V, H));
                }
                else // Transmission
                {
                    L = normalize(refract(-1.f * V, H, eta));
                }
            }
            else // Clearcoat
            {
                float3 H = SampleGTR1_2(info.mat.clearcoatRoughness, r1, r2);

                if (H.z < 0.0)
                    H = -1.f * H;

                L = normalize(reflect(-1.f * V, H));
            }

            L = ToWorld(T, B, N, L);
            V = ToWorld(T, B, N, V);

            return DisneyEval_2(info, V, N, L, pdf);
        }
    }

}