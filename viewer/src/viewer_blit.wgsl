struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@group(0) @binding(0)
var image: texture_2d<f32>;
@group(0) @binding(1)
var samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    const positions = array(
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
    );

    let p = positions[idx];

    return VertexOutput(
        vec4(p, 0.0, 1.0),
        vec2(p.x, -p.y) / 2.0 + 0.5,
    );
}

@fragment
fn fs_main(inp: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image, samp, inp.texcoord);
}
