/**
 * Focus Labs — hero scene
 * A silver "time vortex": a particle galaxy slowly winding around a bright
 * core, plus drifting dust. Subtle mouse parallax, scroll dolly, and pauses
 * when off-screen. Degrades gracefully if WebGL or the CDN is unavailable.
 */
import * as THREE from "three";

const canvas = document.getElementById("heroCanvas");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (canvas && !reducedMotion) {
  try {
    init(canvas);
  } catch (err) {
    console.warn("Hero scene disabled:", err);
  }
}

function init(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x07080a, 0.045);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
  camera.position.set(0, 1.6, 9);
  camera.lookAt(0, 0, 0);

  const galaxy = createGalaxy();
  scene.add(galaxy);

  const dust = createDust();
  scene.add(dust);

  const core = createCore();
  scene.add(core);

  const ring = createRing();
  scene.add(ring);

  // --- Interaction state -------------------------------------------------
  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };

  window.addEventListener("pointermove", (e) => {
    mouse.tx = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.ty = (e.clientY / window.innerHeight) * 2 - 1;
  });

  let scrollProgress = 0;
  window.addEventListener(
    "scroll",
    () => {
      scrollProgress = Math.min(window.scrollY / window.innerHeight, 1);
    },
    { passive: true }
  );

  function resize() {
    const { clientWidth: w, clientHeight: h } = canvas;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // Pause rendering while the hero is off-screen
  let visible = true;
  new IntersectionObserver(([entry]) => (visible = entry.isIntersecting), {
    threshold: 0,
  }).observe(canvas);

  const clock = new THREE.Clock();

  renderer.setAnimationLoop(() => {
    if (!visible) return;

    const t = clock.getElapsedTime();

    mouse.x += (mouse.tx - mouse.x) * 0.04;
    mouse.y += (mouse.ty - mouse.y) * 0.04;

    galaxy.rotation.y = t * 0.045;
    dust.rotation.y = -t * 0.012;
    ring.rotation.z = t * 0.08;

    core.material.opacity = 0.75 + Math.sin(t * 1.4) * 0.12;
    const s = 1 + Math.sin(t * 1.4) * 0.05;
    core.scale.setScalar(s);

    camera.position.x = mouse.x * 0.9;
    camera.position.y = 1.6 - mouse.y * 0.6 - scrollProgress * 1.2;
    camera.position.z = 9 - scrollProgress * 2.2;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
  });
}

/* ------------------------------------------------------------------------ */

function createGalaxy() {
  const COUNT = 14000;
  const ARMS = 3;
  const RADIUS = 7;

  const positions = new Float32Array(COUNT * 3);
  const colors = new Float32Array(COUNT * 3);

  const inner = new THREE.Color(0xffffff);
  const mid = new THREE.Color(0xc7ccd5);
  const outer = new THREE.Color(0x3e434c);

  for (let i = 0; i < COUNT; i++) {
    // Bias particles toward the center for a bright core
    const r = Math.pow(Math.random(), 1.6) * RADIUS;
    const armAngle = ((i % ARMS) / ARMS) * Math.PI * 2;
    const spin = r * 0.85;

    const spread = (1 - r / RADIUS) * 0.35 + 0.06;
    const rx = gaussian() * spread * r * 0.45;
    const ry = gaussian() * spread * r * 0.22;
    const rz = gaussian() * spread * r * 0.45;

    const angle = armAngle + spin;
    positions[i * 3] = Math.cos(angle) * r + rx;
    positions[i * 3 + 1] = ry;
    positions[i * 3 + 2] = Math.sin(angle) * r + rz;

    const c =
      r / RADIUS < 0.3
        ? inner.clone().lerp(mid, (r / RADIUS) / 0.3)
        : mid.clone().lerp(outer, (r / RADIUS - 0.3) / 0.7);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.022,
    sizeAttenuation: true,
    vertexColors: true,
    map: createParticleTexture(),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const points = new THREE.Points(geometry, material);
  points.rotation.x = 0.28;
  return points;
}

function createDust() {
  const COUNT = 900;
  const positions = new Float32Array(COUNT * 3);

  for (let i = 0; i < COUNT; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 24;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 14;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 24;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    size: 0.035,
    sizeAttenuation: true,
    color: 0x9aa0ab,
    map: createParticleTexture(),
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  return new THREE.Points(geometry, material);
}

function createCore() {
  const material = new THREE.SpriteMaterial({
    map: createGlowTexture(),
    color: 0xe8eaee,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.setScalar(3.2);
  return sprite;
}

function createRing() {
  const geometry = new THREE.TorusGeometry(4.6, 0.004, 8, 220);
  const material = new THREE.MeshBasicMaterial({
    color: 0xaab0ba,
    transparent: true,
    opacity: 0.22,
  });
  const ring = new THREE.Mesh(geometry, material);
  ring.rotation.x = Math.PI / 2 - 0.28;
  return ring;
}

/* --- texture helpers ----------------------------------------------------- */

function createParticleTexture() {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.55)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

function createGlowTexture() {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,0.9)");
  g.addColorStop(0.2, "rgba(232,234,238,0.4)");
  g.addColorStop(0.5, "rgba(170,176,186,0.12)");
  g.addColorStop(1, "rgba(170,176,186,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

function gaussian() {
  // Box–Muller transform for a softer, organic particle spread
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
