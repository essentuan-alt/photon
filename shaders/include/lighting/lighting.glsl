#if !defined INCLUDE_LIGHTING_LIGHTING
#define INCLUDE_LIGHTING_LIGHTING

#include "/include/atmospherics/phaseFunctions.glsl"

#include "/include/lighting/bsdf.glsl"
#include "/include/lighting/cloudShadows.glsl"
#include "/include/lighting/shadowMapping.glsl"

#include "/include/fragment/raytracer.glsl"

#include "/include/utility/fastMath.glsl"
#include "/include/utility/spaceConversion.glsl"
#include "/photonics/ph_samplers.glsl"

const float skylightBoost       = 1.0;
const float blocklightIntensity = 64.0 * BLOCKLIGHT_INTENSITY;
const float emissionIntensity   = 16.0 * BLOCKLIGHT_INTENSITY;
const float sssIntensity        = 3.0;
const float sssDensity          = 12.0;

vec3 getSubsurfaceScattering (vec3 albedo, float sssAmount, float sssDepth, float LoV) {
	if (sssAmount < eps) return vec3(0.0);

	vec3 coeff = normalizeSafe(albedo) * sqrt(sqrt(length(albedo)));
	     coeff = (clamp01(coeff) * sssDensity - sssDensity) / sssAmount;

	vec3 sss1 = exp(3.0 * coeff * sssDepth) * henyeyGreensteinPhase(-LoV, 0.4);
	vec3 sss2 = exp(1.0 * coeff * sssDepth) * (0.6 * henyeyGreensteinPhase(-LoV, 0.33) + 0.4 * henyeyGreensteinPhase(-LoV, -0.2));

	return albedo * sssIntensity * sssAmount * (sss1 + sss2);
}

vec3 getScreenSpaceShadows (vec3 viewPos, float dither, out float distantSss) {
	mat3 t = tbnNormal(viewShadowDir);
	vec3 offsetPos = viewPos + mat2x3(t) * polar(0.5, tau * dither);

	vec3 rayEnd = viewToScreenSpace(offsetPos, true);
	vec3 rayPos = viewToScreenSpace(offsetPos + viewShadowDir * 15.0, true);

	bool hit = raymarchIntersection(
		rayPos,
		rayEnd - rayPos,
		dither,
		8u,
		2u,
		0.25
	);

	distantSss = max0(0.5 + dot(viewShadowDir, screenToViewPos(rayPos.xy, rayPos.z, true) - offsetPos));

	return vec3(!hit);
}

float getBlocklightFalloff(float blocklight, float ao) {
	float falloff  = rcp(sqr(16.0 - 15.0 * blocklight));
	      falloff  = linearStep(rcp(sqr(16.0)), 1.0, falloff);
	      falloff *= mix(ao, 1.0, falloff);

	return falloff;
}

float getSkylightFalloff(float skylight) {
	return pow4(skylight);
}

float getFakeBouncedLight(vec3 bentNormal, float sssDepth, float ao) {
	const float bounceAlbedo = 0.5;
	const float bounceBoost  = pi;
	const float bounceMul    = bounceAlbedo * bounceBoost * rcpPi;

	vec3 bounceDir = vec3(shadowDir.xz, -shadowDir.y).xzy;
	float bounce0 = clamp01(dot(bentNormal, bounceDir)) * (1.0 - exp2(-0.125 * sssDepth));
	float bounce1 = 0.33 * ao * clamp01(0.5 - 0.5 * bentNormal.y);

	return (bounceAlbedo * rcpPi) * (bounce0 + bounce1) * dampen(clamp01(shadowDir.y + 0.15));
}

vec3 getSceneLighting(
	Material material,
	vec3 scenePos,
	vec3 viewPos,
	vec3 normal,
	vec3 geometryNormal,
	vec3 viewerDir,
	vec3 directIrradiance,
#if defined PROGRAM_DEFERRED_LIGHTING && defined INDIRECT_LIGHTING
	vec3 indirectIrradiance,
#else
	vec3 ambientIrradiance,
	vec3 skyIrradiance,
#endif
	vec2 lmCoord,
	float dither,
	float ao,
	uint blockId,
	out float sssDepth
) {
	ao = 1.0;
	vec3 radiance = material.emission * emissionIntensity;
	bool is_lod = length(scenePos) > far;

	// Sunlight/moonlight

#if defined WORLD_OVERWORLD || defined WORLD_END
	float NoL = dot(normal, shadowDir) * step(0.0, dot(geometryNormal, shadowDir));

#if defined WORLD_OVERWORLD && defined CLOUD_SHADOWS
	float cloudShadow = getCloudShadows(colortex15, scenePos);
#else
	float cloudShadow = 1.0;
#endif

	vec3 shadowViewPos = transform(shadowModelView, scenePos);

	float blockerDepth = getBlockerDepth(shadowViewPos, dither);

	vec3 visibility = calculateShadows(
		shadowViewPos, 
		geometryNormal, 
		blockId, 
		cloudShadow, 
		lmCoord.y, 
		NoL,
		dither, 
		blockerDepth
	);

	float invDist = rcp(min(far, shadowDistance));

	float lodGradient = smoothstep(0.9, 1.0, length(scenePos) * invDist);
	float shadowGradient = dot(scenePos, shadowDir) > 0.0 ? lodGradient : smoothstep(0.9, 1.0, length(shadowViewPos.xy) * invDist);

	sssDepth = blockerDepth;
	
	if (lodGradient > 0.0) {
		float distantSss;

		visibility = (visibility * (1.0 - shadowGradient) + shadowGradient) * (getScreenSpaceShadows(viewPos, dither, distantSss) * lodGradient + (1.0 - lodGradient));
		sssDepth = mix(sssDepth, distantSss, lodGradient);
	}

	visibility *= NoL;

	if (maxOf(visibility) > eps || material.sssAmount > eps) {
		float NoV = clamp01(dot(normal, viewerDir));
		float LoV = dot(shadowDir, viewerDir);
		float halfwayNorm = inversesqrt(2.0 * LoV + 2.0);
		float NoH = (NoL + NoV) * halfwayNorm;
		float LoH = LoV * halfwayNorm + halfwayNorm;

		vec3 diffuse = diffuseHammon(material, NoL, NoV, NoH, LoV) * (1.0 - 0.75 * material.sssAmount);
		vec3 specular = getSpecularHighlight(material, NoL, NoV, NoH, LoV, LoH);
		vec3 subsurface = getSubsurfaceScattering(material.albedo, material.sssAmount, sssDepth, LoV);

		radiance += directIrradiance * ((diffuse + specular) * visibility + subsurface) * cloudShadow;
	}
#endif

	vec3 bsdf = material.albedo * rcpPi * float(!material.isMetal);

#if defined PROGRAM_DEFERRED_LIGHTING && defined PHOTONICS_ENABLED && defined PHOTONICS
	if (!is_lod) {

#if defined INDIRECT_LIGHTING && defined PHOTONICS_ENABLED && (!defined RESTIR_COMBINED_GI || LIGHTING_MODE == 0)
		radiance+= texture(radiosityIndirect, coord).rgb * bsdf;
#endif

		vec3 direct = sample_photonics_direct(coord);
		direct += sample_photonics_handheld(coord);
		direct *= 4.7f;
		direct *= bsdf;

		radiance+= direct;
	}
#endif

#if defined PROGRAM_DEFERRED_LIGHTING && defined INDIRECT_LIGHTING
	// Indirect lighting already computed alongside HBIL
	radiance += indirectIrradiance * ao * bsdf;
#else

	// Blocklight

	vec3 blocklightColor = blackbody(BLOCKLIGHT_TEMPERATURE);
	float blocklightFalloff = getBlocklightFalloff(lmCoord.x, ao);
	radiance += blocklightIntensity * blocklightColor * blocklightFalloff * bsdf;

	// Skylight

	float skylightFalloff = getSkylightFalloff(lmCoord.y);
	radiance += skyIrradiance * skylightFalloff * skylightBoost * bsdf;

#if defined WORLD_OVERWORLD && defined FAKE_BOUNCED_SUNLIGHT && SHADOW_QUALITY == SHADOW_QUALITY_FANCY
	radiance += getFakeBouncedLight(normal, blockerDepth, ao) * directIrradiance * bsdf * (skylightFalloff * skylightFalloff * cloudShadow);
#endif

	// Ambient light

	radiance += ambientIrradiance * ao * bsdf;
#endif

	return radiance;
}

#endif // INCLUDE_LIGHTING_LIGHTING
