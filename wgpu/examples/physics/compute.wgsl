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

fn wrap(v: vec2<f32>, bound: f32) -> vec2<f32> {
  var ret = vec2<f32>(v.x, v.y);
  if (ret.x < -bound) {
    ret.x = bound;
  }
  if (ret.x > bound) {
    ret.x = -bound;
  }
  if (ret.y < -bound) {
    ret.y = bound;
  }
  if (ret.y > bound) {
    ret.y = -bound;
  }
  return ret;
}

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
  let total = arrayLength(&particlesSrc.particles);
  let index = global_invocation_id.x;
  if (index >= total) {
    return;
  }

  // var vPos : vec2<f32> = particlesSrc.particles[index].pos;
  // var vVel : vec2<f32> = particlesSrc.particles[index].vel;

  var p : vec2<f32> = particlesSrc.particles[index].pos;
  var v : vec2<f32> = particlesSrc.particles[index].vel;
  var a : vec2<f32> = vec2<f32>(0.0, 0.0);
  var m : f32 = particlesSrc.particles[index].m[0];

  // var cMass : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var cVel : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var colVel : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var cMassCount : i32 = 0;
  // var cVelCount : i32 = 0;

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
    if (r_2 < 0.1) {
      r_2 = 0.1;
    }
    let a_i = normalize(d) * m_i * 0.001 / r_2;
    a = a + a_i;

    // if (distance(pos, vPos) < params.rule1Distance) {
    //   cMass = cMass + pos;
    //   cMassCount = cMassCount + 1;
    // }
    // if (distance(pos, vPos) < params.rule2Distance) {
    //   colVel = colVel - (pos - vPos);
    // }
    // if (distance(pos, vPos) < params.rule3Distance) {
    //   cVel = cVel + vel;
    //   cVelCount = cVelCount + 1;
    // }

    continuing {
      i = i + 1u;
    }
  }
  // if (cMassCount > 0) {
  //   cMass = cMass * (1.0 / f32(cMassCount)) - vPos;
  // }
  // if (cVelCount > 0) {
  //   cVel = cVel * (1.0 / f32(cVelCount));
  // }

  // vVel = vVel + (cMass * params.rule1Scale) +
  //     (colVel * params.rule2Scale) +
  //     (cVel * params.rule3Scale);

  // // clamp velocity for a more pleasing simulation
  // vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);

  let t = params.deltaT;

  v = 0.99 * v + a * t;
  v = wrap(v, 1.0);

  p = p + v * t;
  p = wrap(p, 1.0);

  // Write back
  particlesDst.particles[index].pos = p;
  particlesDst.particles[index].vel = v;
}
