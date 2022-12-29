#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 50
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10
#define W_LIGHT 10
#define BIAS 1e-2

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

// 查看实例https://codepen.io/arkhamwjz/pen/MWbqJNG?editors=1010
void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

// 查看实例https://codepen.io/arkhamwjz/pen/MWbqJNG?editors=1010（注释下poissonDiskSamples）
void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  float textureSize = 2048.0;
  // 注意 block 的步长要比 PCSS 中的 PCF 步长长一些，这样生成的软阴影会更加柔和
  float filterStride = 20.0;
  float filterRange = 1.0 / textureSize * filterStride;

  float avgDepth = 0.0;
  int count = 0;
  for( int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i ++ ) {
    vec2 samplePos = uv + poissonDisk[i] * filterRange;
    float depth = unpack(texture2D(shadowMap, samplePos));
    if(depth + float(BIAS) < zReceiver) {
      avgDepth += depth;
      count++;
    }
  }
  if(count == 0) return 1.0;
  if(count == BLOCKER_SEARCH_NUM_SAMPLES) return 0.0;

  avgDepth = avgDepth / float(count);
	return avgDepth;
}

float PCF(sampler2D shadowMap, vec4 coords) {
  // 随机采样
  poissonDiskSamples(coords.xy);

  // shadow map 的大小, 越大滤波的范围越小
  float textureSize = 2048.0;
  // 滤波的步长
  float filterStride = 5.0;
  // 滤波窗口的范围
  float filterRange = 1.0 / textureSize * filterStride;

  float shadow = 0.0;
  for( int i = 0; i < PCF_NUM_SAMPLES; i ++ ) {
    vec2 samplePos = coords.xy + poissonDisk[i] * filterRange;
    shadow += (unpack(texture2D(shadowMap, samplePos)) + float(BIAS) > coords.z ? 1.0 : 0.0);
  }
  shadow /= float(PCF_NUM_SAMPLES);

  return shadow;
}

float PCSS(sampler2D shadowMap, vec4 coords){
  // 随机采样
  uniformDiskSamples(coords.xy);

  // STEP 1: avgblocker depth
  float avgblocker = findBlocker(shadowMap, coords.xy, coords.z);

  // STEP 2: penumbra size
  float penumbra = (coords.z - avgblocker) * float(W_LIGHT) / avgblocker ;

  // STEP 3: filtering
  // shadow map 的大小, 越大滤波的范围越小
  float textureSize = 2048.0;
  // 滤波的步长
  float filterStride = 2.0;
  // 滤波窗口的范围
  float filterRange = 1.0 / textureSize * filterStride * penumbra;

  float shadow = 0.0;
  for( int i = 0; i < PCF_NUM_SAMPLES; i ++ ) {
    vec2 samplePos = coords.xy + poissonDisk[i] * filterRange;
    shadow += (unpack(texture2D(shadowMap, samplePos)) + float(BIAS) > coords.z ? 1.0 : 0.0);
  }
  shadow /= float(PCF_NUM_SAMPLES);

  return shadow;
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  // 别忘了深度pack过
  float depth = unpack(texture2D(shadowMap, shadowCoord.xy));
  return depth + float(BIAS) > shadowCoord.z ? 1.0 : 0.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {
  // 透视分割，正常OpenGL会处理，但这里是直接计算后传递过来的值
  vec3 projCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  // 映射到NDC坐标
  projCoord = projCoord * 0.5 + 0.5;

  float visibility;
  // visibility = useShadowMap(uShadowMap, vec4(projCoord, 1.0));
  // visibility = PCF(uShadowMap, vec4(projCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(projCoord, 1.0));

  vec3 phongColor = blinnPhong();
  
  gl_FragColor = vec4(phongColor * visibility, 1.0);
  // gl_FragColor = vec4(visibility, 0.0, 0.0, 1.0);
  // gl_FragColor = vec4(phongColor, 1.0);
}