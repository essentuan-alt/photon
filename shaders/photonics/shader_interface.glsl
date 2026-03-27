#include "/include/main.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/spaceConversion.glsl"

uniform usampler2D colortex1; // albedo, geometry normal, texture normal
uniform sampler2D lodDepthTex1;

uniform sampler2D skyCapture; // Sky capture, lighting color palette, dynamic weather properties

#include "/include/atmospherics/skyProjection.glsl"
#include "/include/utility/sphericalHarmonics.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/sampling.glsl"

ivec2 texel = ivec2(gl_FragCoord.xy);
vec3 indirect_light_color = vec3(1f);

vec3 load_world_position() {
    float depth = texelFetch(lodDepthTex1, texel, 0).x;

    vec3 viewPos = screenToViewPos(gl_FragCoord.xy * viewTexelSize, depth, true);
    vec3 scenePos = viewToSceneSpace(viewPos);

    return scenePos + cameraPosition;
}

void load_fragment_variables(out vec3 albedo, out vec3 world_pos, out vec3 geometry_normal, out vec3 texture_normal) {
    uvec4 encoded = texelFetch(colortex1, texel, 0);
    mat2x4 data = mat2x4(
        unpackUnorm4x8(encoded.x),
        unpackUnorm4x8(encoded.y)
    );

    albedo = data[0].rgb;
    geometry_normal = octDecode(data[1].xy);

#ifdef NORMAL_MAP
    vec4 normalData = unpackUnormArb(encoded.z, uvec4(12, 12, 7, 1));
    texture_normal = octDecode(normalData.xy);
#else
    texture_normal = geometry_normal;
#endif

#if defined PH_LIGHTING_PASS && defined INDIRECT_LIGHTING
    #if defined SH_SKYLIGHT && defined OVERWORLD
        vec3 skySh[9] = vec3[9](vec3(0f), vec3(0f), vec3(0f), vec3(0f), vec3(0f), vec3(0f), vec3(0f), vec3(0f), vec3(0f));

        // Sample into SH
        const uint sampleCount = 256;
        for (uint i = 0; i < sampleCount; ++i) {
            vec3 direction = uniformHemisphereSample(vec3(0.0, 1.0, 0.0), R2(int(i)));
            vec3 radiance  = texture(skyCapture, projectSky(direction)).rgb;
            float[9] coeff = getSphericalHarmonicsCoefficientsOrder2(direction);

            for (uint band = 0; band < 9; ++band) skySh[band] += radiance * coeff[band];
        }

        // Normalize SH
        const float sampleSolidAngle = tau / float(sampleCount);
        for (uint band = 0; band < 9; ++band) skySh[band] *= sampleSolidAngle;

        indirect_light_color = evaluateSphericalHarmonicsIrradiance(skySh, vec3(0f), 1f);
    #else
        indirect_light_color = texelFetch(skyCapture, ivec2(255, 2), 0).rgb;
    #endif
#endif

    world_pos = load_world_position() - 0.01f * geometry_normal;
}

vec2 get_taa_jitter() {
    return taa_offset;
}

vec3 sun_direction = sunAngle < 0.5 ? sunDir : moonDir;

bool is_in_world() {
    return texelFetch(depthtex0, texel, 0).x <= 0.99999f;
}