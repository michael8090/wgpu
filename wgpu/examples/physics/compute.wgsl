struct Particle {
  pos : vec2<f32>;
  vel : vec2<f32>;
  a : vec2<f32>;
  m : vec2<f32>;
};

[[block]]
struct SimParams {
  deltaT : f32;
  rule1Distance : f32;
  rule2Distance : f32;
  rule3Distance : f32;
  rule1Scale : f32;
  rule2Scale : f32;
  rule3Scale : f32;
};

[[block]]
struct Particles {
  particles : [[stride(32)]] array<Particle>;
};

[[group(0), binding(0)]] var<uniform> params : SimParams;
[[group(0), binding(1)]] var<storage, read> particlesSrc : Particles;
[[group(0), binding(2)]] var<storage, read_write> particlesDst : Particles;


[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
  let total = arrayLength(&particlesSrc.particles);
  let index = global_invocation_id.x;
  if (index >= total) {
    return;
  }

  var p : vec2<f32> = particlesSrc.particles[index].pos;
  var v : vec2<f32> = particlesSrc.particles[index].vel;
  var a : vec2<f32> = vec2<f32>(0.0, 0.0);
  var m : f32 = particlesSrc.particles[index].m[0];

  var i : u32 = 0u;
  loop {
    if (i >= total) {
      break;
    }
    if (i == index) {
      continue;
    }

    let pos_i = particlesSrc.particles[i].pos;
    let m_i = particlesSrc.particles[i].m[0];

    let d = pos_i - p;
    var r_2 = d.x * d.x + d.y * d.y;
    if (r_2 < 10000.0) {
      // r_2 = 0.001;
      continue;
    }
    let a_i = normalize(d) * m_i * 100000000.0 / r_2;
    a = a + a_i;

    continuing {
      i = i + 1u;
    }
  }

  let t = params.deltaT;

  v = v + a * t;

  p = p + v * t;

  // Write back
  particlesDst.particles[index].pos = p;
  particlesDst.particles[index].vel = v;
}
