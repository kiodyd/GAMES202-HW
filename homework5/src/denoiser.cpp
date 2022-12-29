#include "denoiser.h"
Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            Float3 currWorldPos = frameInfo.m_position(x, y);
            int currObjId = frameInfo.m_id(x, y);
            if (currObjId >= 0.){
                Matrix4x4 currWorld2ObjMatrixInv = Inverse(frameInfo.m_matrix[currObjId]);
                Matrix4x4 preWorld2ObjMatrix = m_preFrameInfo.m_matrix[currObjId];
                Float3 preScreenCoord = currWorld2ObjMatrixInv(currWorldPos, Float3::EType::Point);
                preScreenCoord = preWorld2ObjMatrix(preScreenCoord, Float3::EType::Point);
                preScreenCoord = preWorldToScreen(preScreenCoord, Float3::EType::Point);

                preScreenCoord.x = int(preScreenCoord.x);
                preScreenCoord.y = int(preScreenCoord.y);
                if (preScreenCoord.x >= 0 && preScreenCoord.x < width
                    && preScreenCoord.y >= 0 && preScreenCoord.y < height
                    && int(m_preFrameInfo.m_id(preScreenCoord.x, preScreenCoord.y)) == int(currObjId))
                {
                    m_valid(x, y) = true;
                    m_misc(x, y) = m_accColor(preScreenCoord.x, preScreenCoord.y);
                    continue;
                }

            }
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);

            // 求方差和均值
            Float3 uColor, aColor = Float3(0.f);
            float count = Sqr(kernelRadius * 2 + 1);
            for (int yy = y - kernelRadius; yy < y + kernelRadius; ++yy) {
                for (int xx = x - kernelRadius; xx <= x + kernelRadius; ++xx) {
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        uColor += curFilteredColor(xx, yy);
                        aColor += Sqr(curFilteredColor(xx, yy));
                    }
                }
            }
            uColor /= count;
            // 方差=平方的均值-均值的平方
            aColor = SafeSqrt(aColor/count - Sqr(uColor));

            // Clamp, 避免异常噪点干扰，以及其他因temporal导致的异常点
            color = Clamp(color, uColor - aColor * m_colorBoxK, uColor + aColor * m_colorBoxK);


            // TODO: Exponential moving average
            float alpha = m_valid(x, y) ? m_alpha : 1.0f;
            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
    float nomSigmaCoord     = -2 * m_sigmaCoord * m_sigmaCoord;
    float nomSigmaColor     = -2 * m_sigmaColor * m_sigmaColor;
    float nomSigmaNormal    = -2 * m_sigmaNormal * m_sigmaNormal;
    float nomSigmaPlane     = -2 * m_sigmaPlane * m_sigmaPlane;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            // 注意：需要将权值归一化！！！以及考虑自身的贡献
            Float3 filterValue = 0;
            float weightSum = 0;
            for (int yy = y - kernelRadius; yy <= y + kernelRadius; ++yy) {
                for (int xx = x - kernelRadius; xx <= x + kernelRadius ; ++xx) {
                    if (yy < 0 || yy >= height || xx < 0 || xx >= width)
                        continue;
                    // 自身的贡献
                    if (yy == y && xx == x)
                    {
                        filterValue += frameInfo.m_beauty(x, y);
                        weightSum += 1;
                        continue;
                    }

                    float distance = Distance(frameInfo.m_position(xx,yy), frameInfo.m_position(x,y));

                    float coordSqr = Sqr(distance);

                    float colorSqr = SqrDistance(frameInfo.m_beauty(x, y), frameInfo.m_beauty(xx,yy));

                    float normalSqr = SafeAcos(Dot(frameInfo.m_normal(x,y), frameInfo.m_normal(xx, yy)));
                    normalSqr = normalSqr * normalSqr;

                    float planeSqr = 0.f;
                    if (distance > 0.f) {
                        planeSqr = Sqr(Dot(frameInfo.m_normal(x,y),
                                           Normalize(frameInfo.m_position(xx,yy) - frameInfo.m_position(x,y))));
                    }

                    float J_ij = coordSqr / nomSigmaCoord + colorSqr / nomSigmaColor
                                 + normalSqr / nomSigmaNormal + planeSqr / nomSigmaPlane;
                    float weight = expf(J_ij);
                    weightSum += weight;
                    filterValue += frameInfo.m_beauty(xx,yy) * weight;
                }
            }
            filteredImage(x,y) = filterValue / weightSum;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }
Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
