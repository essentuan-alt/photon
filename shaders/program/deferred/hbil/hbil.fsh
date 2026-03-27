/*
 * Program description:
 * Compute horizon-based indirect lighting
 *
 * references:
 * HBIL - https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
 * GTAO - https://github.com/Patapom/GodComplex/blob/master/Tests/TestHBIL/2018%20Mayaux%20-%20Horizon-Based%20Indirect%20Lighting%20(HBIL).pdf (new paper)
 */

#include "/include/main.glsl"

//--// Outputs //-------------------------------------------------------------//

/* RENDERTARGETS: 5 */
layout (location = 0) out vec4 data;

//--// Inputs //--------------------------------------------------------------//

flat in vec3[9] skySh;

flat in vec3 ambientIrradiance;
flat in vec3 directIrradiance;
flat in vec3 skyIrradiance;

//--// Uniforms //------------------------------------------------------------//

uniform sampler2D noisetex;

uniform usampler2D colortex1; // Scene data

uniform sampler2D shadowcolor0; // Shadow albedo
uniform sampler2D shadowcolor1; // Shadow normal & skylight

uniform sampler2D shadowtex1;

uniform sampler2D lodDepthTex1;

//--// Includes //------------------------------------------------------------//

#include "/include/lighting/shadowDistortion.glsl"
#include "/include/lighting/cloudShadows.glsl"

#include "/include/utility/color.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/fastMath.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/sampling.glsl"
#include "/include/utility/spaceConversion.glsl"
#include "/include/utility/sphericalHarmonics.glsl"

//--// Functions //-----------------------------------------------------------//

const float indirectRenderScale = 0.01 * INDIRECT_RENDER_SCALE;

float getMaxHorizonAngle (vec2 sliceDir, vec2 screenPos, vec3 viewPos, vec3 viewDir, vec2 stepSize, float dither) {
    vec2 stepDir = stepSize * sliceDir.xy;
    vec2 stepPos = screenPos + stepDir * dither;
    
    float maxTheta = -1.0;

    for (int i = 0; i < GTAO_HORIZON_STEPS; i++, stepPos += stepDir) {
        float sampleDepth = texelFetch(lodDepthTex1, ivec2(viewSize * stepPos), 0).r;

        vec3 sampleVec = screenToViewPos(stepPos.xy, sampleDepth, true) - viewPos;
        float lengthSqu = dot(sampleVec, sampleVec);

        float cosTheta = dot(sampleVec, viewDir) * inversesqrt(lengthSqu);
              cosTheta = mix(cosTheta, -1.0, clamp01(lengthSqu - 3.0 * GTAO_RADIUS));

        maxTheta = max(maxTheta, cosTheta);
    }

    return acos(clamp(maxTheta, -1.0, 1.0));
}

vec4 getAmbientOcclusion (vec3 screenPos, vec3 viewPos, vec3 viewNormal, vec2 dither) {
    #if GTAO_SLICES > 0
        vec3 viewDir = normalize(-viewPos);
        vec2 stepSize = vec2(lodProjMat_0.x, lodProjMat_1.y) * GTAO_RADIUS * rcp(GTAO_HORIZON_STEPS * max(0.25, -viewPos.z));

        vec3 sliceDir = vec3(cos(tau * dither.x), sin(tau * dither.x), 0.0);
        vec4 integratedData = vec4(0.0);

        for (int i = 0; i < GTAO_SLICES; i++) {
            float sliceAngle = pi * (i + dither.x) * rcp(GTAO_SLICES);
            vec3 sliceDir = vec3(cos(sliceAngle), sin(sliceAngle), 0.0);

            vec3 tangent = sliceDir - dot(sliceDir, viewDir) * viewDir;
            vec3 axis = cross(sliceDir, viewDir);
            vec3 projNormal = viewNormal - axis * dot(viewNormal, axis);

            float cosGamma = clamp01(dot(viewDir, projNormal) * inversesqrt(dot(projNormal, projNormal)));
            float gamma = sign(dot(tangent, projNormal)) * acos(cosGamma);

            vec2 horizonAngles = vec2(
                getMaxHorizonAngle(-sliceDir.xy, screenPos.xy, viewPos, viewDir, stepSize, dither.y),
                getMaxHorizonAngle( sliceDir.xy, screenPos.xy, viewPos, viewDir, stepSize, dither.y)
            );

            horizonAngles = gamma + clamp(vec2(-horizonAngles.x, horizonAngles.y) - gamma, -halfPi, halfPi);

            float bentAngle = 0.5 * (horizonAngles.x + horizonAngles.y);

            integratedData.xyz += viewDir * cos(bentAngle) + tangent * sin(bentAngle);
            integratedData.w   += dot(vec2(0.25), cosGamma + 2.0 * horizonAngles * sin(gamma) - cos(2.0 * horizonAngles - gamma));
        }

        return vec4(normalize(mat3(gbufferModelViewInverse) * (normalize(integratedData.xyz) - 0.5 * viewDir + 0.2 * viewNormal)), integratedData.w * rcp(float(GTAO_SLICES)));
    #else
        return vec4(mat3(gbufferModelViewInverse) * viewNormal, 1.0);
    #endif
}

vec3 getBouncedSunlight (vec3 shadowViewPos, vec3 bentNormal, vec2 dither, float skylight) {
    #if SUNLIGHT_GI_SAMPLES > 0
        vec3 shadowViewNormal = mat3(shadowModelView) * bentNormal;
        vec2 shadowClipPos = shadowProjScale.xy * shadowViewPos.xy;

        vec2 sampleState = shadowProjScale.x * SUNLIGHT_GI_RANGE * vec2(cos(dither.x * tau * rcp(SUNLIGHT_GI_SAMPLES)), sin(dither.x * tau * rcp(SUNLIGHT_GI_SAMPLES)));
        mat2 samplePhase = rotate(rcp(SUNLIGHT_GI_SAMPLES) * tau);

        vec3 integratedData = vec3(0.0);

        for (int i = 0; i < SUNLIGHT_GI_SAMPLES; i++) {
            float sampleDist = fract(0.4301597 * i + dither.y);
            sampleState *= samplePhase;

            vec2 sampleClipPos = shadowClipPos + sampleDist * sampleState;
            ivec2 sampleTexel = ivec2(float(shadowMapResolution) * (distortShadowPos(sampleClipPos) * 0.5 + 0.5));

            vec3 sampleViewVec = shadowProjScaleInv * vec3(sampleClipPos, texelFetch(shadowtex1, sampleTexel, 0).r * 2.0 - 1.0) - shadowViewPos;

            float sqrLength = dot(sampleViewVec, sampleViewVec);
            float invLength = inversesqrt(max(0.01, sqrLength));

            if (invLength > rcp(SUNLIGHT_GI_RANGE)) {
                vec4 data0 = texelFetch(shadowcolor0, sampleTexel, 0);
                vec3 data1 = texelFetch(shadowcolor1, sampleTexel, 0).rgb;

                vec3 radiance =  sampleDist * data0.rgb;
                     radiance *= smoothstep(-SUNLIGHT_GI_RANGE, -0.75 * SUNLIGHT_GI_RANGE, -sqrLength * invLength);
                     radiance *= max0(dot(shadowViewNormal, sampleViewVec));
                     radiance *= max0(-dot(octDecode(data1.rg), sampleViewVec));
                     radiance *= sqr(invLength * invLength);
                #ifdef SUNLIGHT_GI_LEAK_FIX
                     radiance *= exp(-6.0 * abs(data1.b - skylight));
                #endif

                integratedData += radiance;
            }
        }

        return 4.0 * SUNLIGHT_GI_RANGE * SUNLIGHT_GI_RANGE * rcp(SUNLIGHT_GI_SAMPLES) * integratedData;
    #else
        return vec3(0.0);
    #endif
}

float adjustBlocklight (float blocklight, float ao) {
	float falloff  = rcp(sqr(16.0 - 15.0 * blocklight));
	      falloff  = linearStep(rcp(sqr(16.0)), 1.0, falloff);
	      falloff *= mix(ao, 1.0, falloff);

	return falloff;
}

float adjustSkylight (float skylight) {
	return pow4(skylight);
}

void main() {
	ivec2 texel     = ivec2(gl_FragCoord.xy);
    ivec2 viewTexel = ivec2(gl_FragCoord.xy * rcp(indirectRenderScale));

	vec2 coord = gl_FragCoord.xy * viewTexelSize * rcp(indirectRenderScale);

	if (clamp01(coord) != coord) discard;

	/* -- texture fetches -- */

	float depth   = texelFetch(lodDepthTex1, viewTexel, 0).x;
	uvec3 encoded = texelFetch(colortex1, viewTexel, 0).xyz;
	vec2 dither   = vec2(texelFetch(noisetex, texel & 511, 0).b, texelFetch(noisetex, (texel + 249) & 511, 0).b);

	if (depth == 0.0) { data = vec4(0.0); return; }
	//if (depth < handDepth) depth += 0.38; // Hand lighting fix from Capt Tatsu

	/* -- unpack gbuffer  -- */

	vec2 lmCoord = unpackUnorm4x8(encoded.y).zw;

#ifdef NORMAL_MAP
	vec4 normalData = unpackUnormArb(encoded.z, uvec4(12, 12, 7, 1));
	vec2 encodedNormal = normalData.xy;
#else
	vec2 encodedNormal = unpackUnorm4x8(encoded.y).xy;
#endif

	vec3 worldNormal = octDecode(encodedNormal);
	vec3 viewNormal  = mat3(gbufferModelView) * worldNormal;

	/* -- transformations  -- */

	vec3 screenPos = vec3(coord, depth);
	vec3 viewPos = screenToViewPos(coord, depth, true);
	vec3 scenePos = transform(gbufferModelViewInverse, viewPos);

 #ifdef PHOTONICS_ENABLED && defined PHOTONICS
    bool is_lod = length(scenePos) > far;
#else
    const bool is_lod = true;
#endif

	/* -- indirect lighting -- */

	vec2 rng = R2(frameCounter, dither);
	vec3 irradiance = vec3(0.0);

	// Ambient occlusion

	vec4 ao = getAmbientOcclusion(screenPos, viewPos, viewNormal, rng);

	// Sunlight GI

    if (is_lod) {
        vec3 shadowViewPos = transform(shadowModelView, scenePos);
        vec3 sunlight = getBouncedSunlight(shadowViewPos, worldNormal, rng, lmCoord.y);

        irradiance += ao.w * directIrradiance * sunlight;
    }

	// Blocklight

    if (is_lod) {
        vec3 blocklightColor = blackbody(BLOCKLIGHT_TEMPERATURE);
        float blocklightFalloff = adjustBlocklight(lmCoord.x, ao.w);
        irradiance += 32.0 * blocklightColor * blocklightFalloff;
    }

	// Skylight

    if (is_lod) {
        vec3 skylight = evaluateSphericalHarmonicsIrradiance(skySh, ao.xyz, ao.w);

        irradiance += adjustSkylight(lmCoord.y) * skylight;
    }

	// Ambient light

	irradiance += ao.w * ambientIrradiance;

	/* -- pack irradiance and gbuffer data -- */

	vec4 irradianceRgbe8 = encodeRgbe8(irradiance);

	data.x = packUnorm2x8(irradianceRgbe8.xy);
	data.y = packUnorm2x8(irradianceRgbe8.zw);
	data.z = clamp01(linearizeDepth(depth) * rcp(renderDistance));
	data.w = packUnorm2x8(encodedNormal);
}
