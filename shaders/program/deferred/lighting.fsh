/*
 * Program description:
 * Deferred lighting pass for solid objects
 */

#include "/include/main.glsl"

//--// Outputs //-------------------------------------------------------------//

/* RENDERTARGETS: 3 */
layout (location = 0) out vec3 radiance;

//--// Inputs //--------------------------------------------------------------//

in vec2 coord;

flat in vec3 ambientIrradiance;
flat in vec3 directIrradiance;
flat in vec3 skyIrradiance;

//--// Uniforms //------------------------------------------------------------//

uniform sampler2D noisetex;

uniform sampler2D colortex0;  // Translucent overlays
uniform usampler2D colortex1; // Scene data
uniform sampler2D colortex2; // clouds history
uniform sampler2D colortex3;  // Scene radiance
uniform sampler2D skyCapture;  // Sky capture
uniform sampler2D colortex5;  // Indirect lighting
uniform sampler2D colortex7;  // Clear sky
uniform sampler2D colortex8;  // Scene history
uniform sampler2D colortex15; // Cloud shadow map

#if defined INDIRECT_LIGHTING && defined PHOTONICS && defined PHOTONICS_ENABLED && (!defined RESTIR_COMBINED_GI || LIGHTING_MODE == 0)
uniform sampler2D radiosityIndirect;
#endif

uniform sampler2D lodDepthTex1;

uniform sampler2D shadowtex0;

#ifdef SHADOW
#ifdef SHADOW_COLOR
uniform sampler2D shadowcolor0;
uniform sampler2DShadow shadowtex0HW;
#endif
uniform sampler2DShadow shadowtex1HW;
#endif

//--// Includes //------------------------------------------------------------//

#define PROGRAM_DEFERRED_LIGHTING
#define TEMPORAL_REPROJECTION

#include "/block.properties"
#include "/entity.properties"

#include "/include/atmospherics/weather.glsl"

#include "/include/fragment/aces/matrices.glsl"
#include "/include/fragment/fog.glsl"
#include "/include/fragment/material.glsl"
#include "/include/fragment/textureFormat.glsl"

#include "/include/lighting/lighting.glsl"
#include "/include/lighting/reflections.glsl"

#include "/include/utility/color.glsl"
#include "/include/utility/fastMath.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/spaceConversion.glsl"
#include "/include/utility/sphericalHarmonics.glsl"

//--// Functions //-----------------------------------------------------------//

const float hbilRenderScale = 0.01 * INDIRECT_RENDER_SCALE;

// Spatial upsampling for HBIL

vec4 weighHbilSample(vec4 data, vec3 normal, float z0, float NoV, float weight) {
	const float depthStrictness = 10.0;
	const float depthTolerance  = 0.005;

	bool isSky = false;

	if (!isSky) {
		vec3 irradianceSample = decodeRgbe8(vec4(unpackUnorm2x8(data.x), unpackUnorm2x8(data.y)));
		vec3 normalSample = octDecode(unpackUnorm2x8(data.w));

		float z1 = data.z * renderDistance;

		weight *= exp2(-max0(abs(z0 - z1) - depthTolerance) * depthStrictness * NoV);
		weight *= pow16(abs(dot(normal, normalSample))) * 0.99 + 0.01;

		return vec4(irradianceSample, 1.0) * weight;
	} else {
		return vec4(0.0);
	}
}

vec3 upsampleHbil(vec3 normal, float linZ, float NoV) {
	vec4 result = vec4(0.0);

	vec2 pos = gl_FragCoord.xy * hbilRenderScale - 0.5;

	ivec2 i = ivec2(pos);
	vec2  f = fract(pos);

	vec4 s0 = texelFetch(colortex5, i + ivec2(0, 0), 0);
	vec4 s1 = texelFetch(colortex5, i + ivec2(1, 0), 0);
	vec4 s2 = texelFetch(colortex5, i + ivec2(0, 1), 0);
	vec4 s3 = texelFetch(colortex5, i + ivec2(1, 1), 0);

	result += weighHbilSample(s0, normal, linZ, NoV, (1.0 - f.x) * (1.0 - f.y)); // bottom left
	result += weighHbilSample(s1, normal, linZ, NoV, f.x - f.x * f.y);           // bottom right
	result += weighHbilSample(s2, normal, linZ, NoV, f.y - f.x * f.y);           // top left
	result += weighHbilSample(s3, normal, linZ, NoV, f.x * f.y);                 // top right

	return (result.w == 0.0) ? skyIrradiance * pow4(eyeSkylight) : result.xyz / result.w;
}

void main() {
	ivec2 texel = ivec2(gl_FragCoord.xy);

	/* -- texture fetches -- */

	float depth   = texelFetch(lodDepthTex1, texel, 0).x;
	vec4 overlays = texelFetch(colortex0, texel, 0);
	uvec4 encoded = texelFetch(colortex1, texel, 0);
	radiance      = texelFetch(colortex3, texel, 0).rgb;
	vec3 clearSky = texelFetch(colortex7, texel, 0).rgb;

	if (depth == 0.0) return;

	/* -- transformations -- */

	//if (depth < handDepth) depth += 0.38; // Hand lighting fix from Capt Tatsu

	vec3 viewPos  = screenToViewPos(coord, depth, true);
	vec3 scenePos = viewToSceneSpace(viewPos);
	vec3 worldPos = scenePos + cameraPosition;

	vec3 viewerDir = normalize(gbufferModelViewInverse[3].xyz - scenePos);

	/* -- unpack gbuffer -- */

	mat2x4 data = mat2x4(
		unpackUnorm4x8(encoded.x),
		unpackUnorm4x8(encoded.y)
	);

	vec3 albedo = data[0].xyz;
	uint blockId = uint(data[0].w * 255.0);
	vec3 geometryNormal = octDecode(data[1].xy);
	vec2 lmCoord = data[1].zw;

#ifdef NORMAL_MAP
	vec4 normalData = unpackUnormArb(encoded.z, uvec4(12, 12, 7, 1));
	vec3 normal = octDecode(normalData.xy);
#else
	#define normal geometryNormal
#endif

	uint overlayId = uint(overlays.a + 0.5);
	albedo = overlayId == 0 ? albedo + overlays.rgb : albedo; // enchantment glint
	albedo = overlayId == 1 ? 2.0 * albedo * overlays.rgb : albedo; // damage overlay
	albedo = srgbToLinear(clamp01(albedo)) * r709ToAp1Unlit;

	/* -- fetch hbil -- */

#ifdef INDIRECT_LIGHTING
	float linZ = linearizeDepth(depth);
	float NoV = abs(dot(normal, viewerDir));
	vec3 indirectIrradiance = upsampleHbil(normal, linZ, NoV);
#endif

	/* -- get material -- */

	Material material = getMaterial(albedo, blockId);

#ifdef SPECULAR_MAP
	vec4 specularTex = unpackUnorm4x8(encoded.w);
	decodeSpecularTex(specularTex, material);
#endif

	/* -- puddles -- */

	getRainPuddles(
		noisetex,
		material.porosity,
		lmCoord,
		worldPos,
		geometryNormal,
		normal,
		material.albedo,
		material.f0,
		material.roughness
	);
	material.n = f0ToIor(material.f0.x);

	/* -- lighting -- */

	float dither = getInterleavedGradientNoise(gl_FragCoord.xy, frameCounter);
	
	float sssDepth;
	radiance = getSceneLighting(
		material,
		scenePos,
		viewPos,
		normal,
		geometryNormal,
		viewerDir,
		directIrradiance,
#ifdef INDIRECT_LIGHTING
		indirectIrradiance,
#else
		ambientIrradiance,
		skyIrradiance,
#endif
		lmCoord,
		dither,
		1.0,
		blockId,
		sssDepth
	);

	/* -- reflections -- */

#ifdef SSR
	mat3 tbnMatrix = tbnNormal(geometryNormal);

	vec3 viewerDirTangent = viewerDir * tbnMatrix;

	radiance += getSpecularReflections(
		material,
		tbnMatrix,
		vec3(coord, depth),
		viewPos,
		normal,
		viewerDir,
		viewerDirTangent,
		lmCoord.y
	);
#endif

	/* -- fog -- */

	radiance = applyFog(radiance, scenePos, clearSky);
}