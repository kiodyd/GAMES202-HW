#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include "vec.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int resolution = 128;
int channel = 3;

Vec2f Hammersley(uint32_t i, uint32_t N) {
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10;
    return {float(i) / float(N), rdi};
}

Vec3f ImportanceSampleGGX(Vec2f Xi, Vec3f N, float roughness) {

    float a = roughness * roughness;

    //TODO: in spherical space - Bonus 1
    // 通过GGX采样半程向量
    double theta = atan(a * sqrt(Xi.x) / sqrt(1-Xi.x));
    double phi =  2 * PI * Xi.y;

    //TODO: from spherical space to cartesian space - Bonus 1
    Vec3f H = Vec3f(sin(theta) * cos(phi),
                    sin(theta) * sin(phi),
                    cos(theta));

    //TODO: tangent coordinates - Bonus 1
    Vec3f up = std::abs(N.z) <0.999? Vec3f(0.0f, 0.0f, 1.0f) : Vec3f(1.0f, 0.0f, 0.0f) ;
    Vec3f T = normalize(cross(up, N));
    Vec3f B = cross(N, T);

    //TODO: transform H to tangent space - Bonus 1
    return Vec3f(dot(T, H), dot(B, H), dot(N, H));
}


Vec3f IntegrateEmu(Vec3f V, float roughness, float NdotV, Vec3f Ei) {
    Vec3f Eavg = Vec3f(0.0f);
    const int sample_count = 1024;
    Vec3f N = Vec3f(0.0, 0.0, 1.0);

    for (int i = 0; i < sample_count; i++) 
    {
        Vec2f Xi = Hammersley(i, sample_count);
        Vec3f H = ImportanceSampleGGX(Xi, N, roughness);
        Vec3f L = normalize(H * 2.0f * dot(V, H) - V);

        float NoL = std::max(L.z, 0.0f);
        float NoH = std::max(H.z, 0.0f);
        float VoH = std::max(dot(V, H), 0.0f);
        float NoV = std::max(dot(N, V), 0.0f);

        // TODO: To calculate Eavg here - Bonus 1
        Eavg += Ei * NoL * 2.0f;
    }

    return Eavg / sample_count;
}

void setRGB(int x, int y, float alpha, unsigned char *data) {
	data[3 * (resolution * x + y) + 0] = uint8_t(alpha);
    data[3 * (resolution * x + y) + 1] = uint8_t(alpha);
    data[3 * (resolution * x + y) + 2] = uint8_t(alpha);
}

void setRGB(int x, int y, Vec3f alpha, unsigned char *data) {
	data[3 * (resolution * x + y) + 0] = uint8_t(alpha.x);
    data[3 * (resolution * x + y) + 1] = uint8_t(alpha.y);
    data[3 * (resolution * x + y) + 2] = uint8_t(alpha.z);
}

Vec3f getEmu(int x, int y, int alpha, unsigned char *data, float NdotV, float roughness) {
    return Vec3f(data[3 * (resolution * x + y) + 0],
                 data[3 * (resolution * x + y) + 1],
                 data[3 * (resolution * x + y) + 2]);
}

int main() {
    unsigned char *Edata = stbi_load("./GGX_E_LUT.png", &resolution, &resolution, &channel, 3);
    if (Edata == NULL) 
    {
		std::cout << "ERROE_FILE_NOT_LOAD" << std::endl;
		return -1;
	}
	else 
    {
		std::cout << resolution << " " << resolution << " " << channel << std::endl;
        // | -----> mu(j)
        // | 
        // | rough（i）
        // Flip it, if you want the data written to the texture
        uint8_t data[resolution * resolution * 3];
        float step = 1.0 / resolution;
        Vec3f Eavg = Vec3f(0.0);
		for (int i = 0; i < resolution; i++) 
        {
            float roughness = step * (static_cast<float>(i) + 0.5f);
			for (int j = 0; j < resolution; j++) 
            {
                float NdotV = step * (static_cast<float>(j) + 0.5f);
                Vec3f V = Vec3f(std::sqrt(1.f - NdotV * NdotV), 0.f, NdotV);

                Vec3f Ei = getEmu((resolution - 1 - i), j, 0, Edata, NdotV, roughness);
                Eavg += IntegrateEmu(V, roughness, NdotV, Ei) * step;
                setRGB(i, j, 0.0, data);
			}

            for(int k = 0; k < resolution; k++)
            {
                setRGB(i, k, Eavg, data);
            }

            Eavg = Vec3f(0.0);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_png("GGX_Eavg_LUT.png", resolution, resolution, channel, data, 0);
	}
	stbi_image_free(Edata);
    return 0;
}