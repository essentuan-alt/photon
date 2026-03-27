writeonly uniform image2D imgIndirect;

void write_indirect(vec3 color) {
    imageStore(imgIndirect, ivec2(gl_FragCoord.xy), vec4(color, 1f));
}