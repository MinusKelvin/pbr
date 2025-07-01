struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@group(0) @binding(0)
var image: texture_2d<f32>;
@group(0) @binding(1)
var samp: sampler;
@group(0) @binding(2)
var<uniform> rot: mat3x3f;

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
        vec2(p.x, p.y) / 2.0 + 0.5,
    );
}

fn equal_area_sphere_to_square(dir: vec3f) -> vec2f {
    const PI = 3.1415926535;

    let d = abs(dir);
    let r = sqrt(1.0 - d.y);

    let a = max(d.x, d.z);
    let b = min(d.x, d.z);
    var phi = atan2(b, a) * 2.0 / PI;

    if d.x < d.z {
        phi = 1.0 - phi;
    }

    var p = vec2(r - phi * r, phi * r);

    if dir.y < 0.0 {
        p = 1.0 - p.yx;
    }

    p *= select(sign(dir.xz), vec2(1.0), dir.xz == vec2(0.0));
    return (p + 1.0) * 0.5;
}

@fragment
fn fs_main(inp: VertexOutput) -> @location(0) vec4<f32> {
    let d = rot * normalize(vec3(inp.texcoord * 2.0 - 1.0, 1.0));
    let tc = equal_area_sphere_to_square(d);
    return textureSample(image, samp, tc);
}
