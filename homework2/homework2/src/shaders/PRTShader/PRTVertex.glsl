attribute vec3 aVertexPosition;

attribute mat3 aPrecomputeLT;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform mat3 uPrecomputeLR;
uniform mat3 uPrecomputeLG;
uniform mat3 uPrecomputeLB;

varying highp vec3 vColor;

float dot(mat3 L, mat3 LT) {
  return L[0][0] * LT[0][0] + L[0][1] * LT[0][1] + L[0][2] * LT[0][2]
        + L[1][0] * LT[1][0] + L[1][1] * LT[1][1] + L[1][2] * LT[1][2]
        + L[2][0] * LT[2][0] + L[2][1] * LT[2][1] + L[2][2] * LT[2][2];
}

void main(void) {

  gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix *
                vec4(aVertexPosition, 1.0);
  vColor = vec3(
    dot(uPrecomputeLR, aPrecomputeLT),
    dot(uPrecomputeLG, aPrecomputeLT),
    dot(uPrecomputeLB, aPrecomputeLT)
  );
}