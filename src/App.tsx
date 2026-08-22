import React, { useEffect, useRef, useState } from "react";

// ==========================================
// 1. SHADERS (WebGL2 Kodlari)
// ==========================================
const VERT = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2  uRes;
uniform vec2  uCenter;
uniform vec2  uHalf;
uniform float uRadius;
uniform float uMag;
uniform float uRefract;
uniform float uChroma;
uniform vec2  uChromaDir;
uniform float uAlpha;
uniform float uLift;

out vec4 frag;

float sdRoundBox(vec2 p, vec2 b, float r) {
  r = min(r, min(b.x, b.y));
  vec2 q = abs(p) - b + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

float lensSDF(vec2 sp) {
  return sdRoundBox(sp - uCenter, uHalf, uRadius);
}

vec3 samplePx(vec2 sp) {
  vec2 uv = clamp(sp / uRes, vec2(0.0005), vec2(0.9995));
  return texture(uTex, uv).rgb;
}

void main() {
  vec2 sp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 base = samplePx(sp);

  if (uAlpha <= 0.001) { frag = vec4(base, 1.0); return; }

  float d = lensSDF(sp);
  float aa = 1.0 - smoothstep(-1.5, 1.5, d);
  if (aa <= 0.001) { frag = vec4(base, 1.0); return; }

  float e = 1.2;
  vec2 n = normalize(vec2(
    lensSDF(sp + vec2(e, 0.0)) - lensSDF(sp - vec2(e, 0.0)),
    lensSDF(sp + vec2(0.0, e)) - lensSDF(sp - vec2(0.0, e))
  ) + 1e-6);

  float minHalf = min(uHalf.x, uHalf.y);
  float thickness = minHalf * 0.62;
  float t = clamp(-d / thickness, 0.0, 1.0);

  float k = 1.0 - t;
  float bulge = 1.0 - sqrt(max(0.0, 1.0 - k * k));

  float band = smoothstep(0.30, 0.92, bulge);
  float profile = band * band;

  vec2 q = sp - uCenter;
  float rim = band;
  vec2 magPos = uCenter + q * (1.0 + uMag * rim);

  float disp = profile * uRefract * minHalf * 0.72;

  float ch = uChroma * profile * minHalf * 0.085 * (0.4 + uLift * 0.6);
  vec2 dir = normalize(uChromaDir + 1e-6);

  vec2 pR = magPos - n * (disp + ch) + dir * ch * 1.10;
  vec2 pG = magPos - n * disp;
  vec2 pB = magPos - n * (disp - ch) - dir * ch * 1.10;

  vec3 col = vec3(samplePx(pR).r, samplePx(pG).g, samplePx(pB).b);

  float rest = 1.0 - uLift;
  col = mix(col, vec3(0.145, 0.150, 0.165), rest * 0.10);
  col *= mix(1.03, 1.0, uLift);

  float fres = pow(band, 2.0);
  col += vec3(0.42, 0.60, 1.0) * fres * 0.10 * (0.20 + 0.80 * uLift);

  vec2 L = normalize(vec2(-0.42, -1.0));
  float lambert = max(dot(-n, L), 0.0);
  col += vec3(0.88, 0.94, 1.0) * pow(lambert, 2.2) * fres * 0.20 * (0.16 + 0.84 * uLift);

  float upper = max(dot(n, vec2(0.0, -1.0)), 0.0);
  float hl = pow(upper, 4.5) * smoothstep(0.68, 1.0, bulge);
  col += vec3(1.0) * hl * (0.06 + uLift * 0.45);

  float lower = max(dot(n, vec2(0.0, 1.0)), 0.0);
  col -= vec3(0.12, 0.13, 0.17) * pow(lower, 2.8) * band * 0.55;

  float inner = smoothstep(0.0, 2.5, -d);
  col *= 1.0 - (1.0 - inner) * 0.18;

  float ring = 1.0 - smoothstep(0.0, 2.2, -d);
  col += vec3(1.0) * ring * (0.12 + 0.26 * uLift);

  frag = vec4(mix(base, col, aa * uAlpha), 1.0);
}`;

// ==========================================
// 2. LAYOUT & ICONS
// ==========================================
type Rect = { x: number; y: number; w: number; h: number };
type NavLayout = {
  width: number;
  height: number;
  dpr: number;
  island: Rect;
  islandRadius: number;
  tabCenters: number[];
  tabWidth: number;
  centerY: number;
  profile: { cx: number; cy: number; r: number };
  lensRest: { w: number; h: number; r: number };
  lensLift: { w: number; h: number; r: number; dy: number };
};

const TABS = ["HOME", "SEARCH", "PRACTISE", "NEW"] as const;
type TabId = 0 | 1 | 2 | 3;
type Highlight = TabId | "profile" | null;

function computeLayout(width: number, height: number, dpr: number, safeBottom: number): NavLayout {
  const landscape = width > height;
  const margin = landscape ? Math.max(24, width * 0.06) : 12;
  const islandH = landscape ? 56 : 66;
  const profileD = islandH - 6;
  const gap = 10;
  const bottomInset = safeBottom + (landscape ? 10 : 18);
  const islandY = height - bottomInset - islandH;
  const islandW = width - margin * 2 - profileD - gap;
  const island: Rect = { x: margin, y: islandY, w: islandW, h: islandH };
  const tabWidth = islandW / TABS.length;
  const tabCenters = TABS.map((_, i) => island.x + tabWidth * (i + 0.5));
  const centerY = islandY + islandH / 2;
  const lensW = Math.min(tabWidth - 6, 96);
  const lensH = islandH - 10;

  return {
    width,
    height,
    dpr,
    island,
    islandRadius: islandH / 2,
    tabCenters,
    tabWidth,
    centerY,
    profile: { cx: island.x + islandW + gap + profileD / 2, cy: centerY, r: profileD / 2 },
    lensRest: { w: lensW, h: lensH, r: lensH / 2 },
    lensLift: { w: lensW * 1.42, h: lensH * 1.34, r: (lensH * 1.34) / 2, dy: 0 },
  };
}

function nearestTab(x: number, layout: NavLayout): TabId {
  let best: TabId = 0;
  let bestD = Infinity;
  layout.tabCenters.forEach((cx, i) => {
    const d = Math.abs(cx - x);
    if (d < bestD) {
      bestD = d;
      best = i as TabId;
    }
  });
  return best;
}

function hitTest(x: number, y: number, layout: NavLayout): { kind: "tab"; index: TabId } | { kind: "profile" } | null {
  const p = layout.profile;
  if (Math.hypot(x - p.cx, y - p.cy) <= p.r + 8) return { kind: "profile" };
  const i = layout.island;
  if (x >= i.x - 6 && x <= i.x + i.w + 6 && y >= i.y - 22 && y <= i.y + i.h + 12) {
    return { kind: "tab", index: nearestTab(x, layout) };
  }
  return null;
}

type Ctx = CanvasRenderingContext2D;
function stroke(ctx: Ctx, color: string, w: number) {
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

const TAB_ICONS = [
  (ctx: Ctx, cx: number, cy: number, s: number, color: string) => {
    const w = s * 0.86;
    const roofY = cy - s * 0.46;
    const baseY = cy + s * 0.42;
    const bodyTop = cy - s * 0.1;
    ctx.beginPath();
    ctx.moveTo(cx - w / 2, bodyTop);
    ctx.lineTo(cx, roofY);
    ctx.lineTo(cx + w / 2, bodyTop);
    stroke(ctx, color, s * 0.13);
    ctx.beginPath();
    ctx.moveTo(cx - w * 0.38, bodyTop + s * 0.04);
    ctx.lineTo(cx - w * 0.38, baseY);
    ctx.lineTo(cx + w * 0.38, baseY);
    ctx.lineTo(cx + w * 0.38, bodyTop + s * 0.04);
    stroke(ctx, color, s * 0.13);
  },
  (ctx: Ctx, cx: number, cy: number, s: number, color: string) => {
    const r = s * 0.3;
    const ox = cx - s * 0.05;
    const oy = cy - s * 0.08;
    ctx.beginPath();
    ctx.arc(ox, oy, r, 0, Math.PI * 2);
    stroke(ctx, color, s * 0.13);
    ctx.beginPath();
    ctx.moveTo(ox + r * 0.72, oy + r * 0.72);
    ctx.lineTo(cx + s * 0.4, cy + s * 0.42);
    stroke(ctx, color, s * 0.13);
  },
  (ctx: Ctx, cx: number, cy: number, s: number, color: string) => {
    ctx.beginPath();
    ctx.arc(cx, cy, s * 0.44, 0, Math.PI * 2);
    stroke(ctx, color, s * 0.12);
    ctx.beginPath();
    ctx.arc(cx, cy, s * 0.24, 0, Math.PI * 2);
    stroke(ctx, color, s * 0.12);
    ctx.beginPath();
    ctx.arc(cx, cy, s * 0.07, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  },
  (ctx: Ctx, cx: number, cy: number, s: number, color: string) => {
    const a = s * 0.42;
    ctx.beginPath();
    ctx.moveTo(cx - a, cy);
    ctx.lineTo(cx + a, cy);
    stroke(ctx, color, s * 0.13);
    ctx.beginPath();
    ctx.moveTo(cx, cy - a);
    ctx.lineTo(cx, cy + a);
    stroke(ctx, color, s * 0.13);
  },
];

// ==========================================
// 3. PHYSICS & SPRING
// ==========================================
class Spring {
  x: number;
  v = 0;
  target: number;
  k: number;
  c: number;
  constructor(x: number, k = 260, c = 26) {
    this.x = x;
    this.target = x;
    this.k = k;
    this.c = c;
  }
  set(x: number) {
    this.x = x;
    this.target = x;
    this.v = 0;
  }
  drive(x: number, dt: number) {
    if (dt > 0) {
      const raw = (x - this.x) / dt;
      this.v = Math.max(-4000, Math.min(4000, raw));
    }
    this.x = x;
    this.target = x;
  }
  tune(k: number, c: number) {
    this.k = k;
    this.c = c;
  }
  step(dt: number) {
    let remaining = Math.min(dt, 0.032);
    const h = 1 / 600;
    while (remaining > 0) {
      const s = remaining > h ? h : remaining;
      remaining -= s;
      const a = this.k * (this.target - this.x) - this.c * this.v;
      this.v += a * s;
      this.x += this.v * s;
    }
    if (Math.abs(this.target - this.x) < 0.05 && Math.abs(this.v) < 0.6) {
      this.x = this.target;
      this.v = 0;
    }
  }
  get settled() {
    return this.x === this.target && this.v === 0;
  }
}

function stiffnessForDistance(distancePx: number) {
  const t = Math.max(0, Math.min(1, distancePx / 320));
  return 900 - 240 * t;
}
function dampingForStiffness(k: number, zeta = 1) {
  return 2 * zeta * Math.sqrt(k);
}

// ==========================================
// 4. RENDERER & SCENE
// ==========================================
class NavScene {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private bg: HTMLCanvasElement;
  private bgCtx: CanvasRenderingContext2D;
  private key = "";

  constructor() {
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: false })!;
    this.bg = document.createElement("canvas");
    this.bgCtx = this.bg.getContext("2d", { alpha: false })!;
  }

  render(layout: NavLayout, highlight: Highlight): boolean {
    const pw = Math.round(layout.width * layout.dpr);
    const ph = Math.round(layout.height * layout.dpr);
    const key = `${pw}x${ph}|${highlight === null ? "none" : highlight}`;
    if (key === this.key) return false;
    this.key = key;

    for (const c of [this.canvas, this.bg]) {
      if (c.width !== pw || c.height !== ph) {
        c.width = pw;
        c.height = ph;
      }
    }

    const bgCtx = this.bgCtx;
    bgCtx.setTransform(layout.dpr, 0, 0, layout.dpr, 0, 0);
    bgCtx.fillStyle = "#0e0e11";
    bgCtx.fillRect(0, 0, layout.width, layout.height);
    bgCtx.setTransform(1, 0, 0, 1, 0, 0);

    const ctx = this.ctx;
    ctx.setTransform(layout.dpr, 0, 0, layout.dpr, 0, 0);
    ctx.clearRect(0, 0, layout.width, layout.height);
    ctx.drawImage(this.bg, 0, 0, layout.width, layout.height);

    const { island: r, islandRadius: rad } = layout;
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 28;
    ctx.shadowOffsetY = 12;
    this.roundRect(ctx, r.x, r.y, r.w, r.h, rad);
    ctx.fillStyle = "rgba(10,10,12,0.88)";
    ctx.fill();
    ctx.restore();

    ctx.save();
    this.roundRect(ctx, r.x, r.y, r.w, r.h, rad);
    ctx.clip();
    ctx.filter = "blur(20px) saturate(112%)";
    ctx.drawImage(this.bg, 0, 0, layout.width, layout.height);
    ctx.filter = "none";
    const ig = ctx.createLinearGradient(r.x, r.y, r.x + r.w, r.y + r.h);
    ig.addColorStop(0, "rgba(26,26,30,0.80)");
    ig.addColorStop(1, "rgba(20,20,23,0.84)");
    ctx.fillStyle = ig;
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.restore();

    ctx.save();
    this.roundRect(ctx, r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1, rad);
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    const iconSize = layout.island.h * 0.34;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    TABS.forEach((label, i) => {
      const active = highlight === i;
      const color = active ? "rgba(255,255,255,0.98)" : "rgba(178,186,220,0.72)";
      const cx = layout.tabCenters[i]!;
      const cy = r.y + r.h * 0.38;
      TAB_ICONS[i]!(ctx, cx, cy, iconSize, color);
      ctx.fillStyle = color;
      ctx.font = `${active ? 600 : 500} 10px -apple-system, system-ui, sans-serif`;
      ctx.fillText(label, cx, r.y + r.h * 0.79);
    });

    const { cx: pcx, cy: pcy, r: pr } = layout.profile;
    ctx.save();
    ctx.beginPath();
    ctx.arc(pcx, pcy, pr, 0, Math.PI * 2);
    ctx.clip();
    ctx.filter = "blur(20px) saturate(112%)";
    ctx.drawImage(this.bg, 0, 0, layout.width, layout.height);
    ctx.filter = "none";
    ctx.fillStyle = "rgba(22,22,26,0.78)";
    ctx.fillRect(pcx - pr, pcy - pr, pr * 2, pr * 2);
    const ag = ctx.createLinearGradient(pcx - pr, pcy - pr, pcx + pr, pcy + pr);
    ag.addColorStop(0, "rgba(94,132,255,0.85)");
    ag.addColorStop(1, "rgba(150,84,220,0.85)");
    ctx.fillStyle = ag;
    ctx.beginPath();
    ctx.arc(pcx, pcy, pr - 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = "rgba(255,255,255,0.96)";
    ctx.font = "600 16px -apple-system, system-ui, sans-serif";
    ctx.fillText("A", pcx, pcy + 1);

    ctx.beginPath();
    ctx.arc(pcx, pcy, pr - 0.75, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return true;
  }

  private roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
}

class LensRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private tex: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private texW = 0;
  private texH = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { alpha: false });
    if (!gl) throw new Error("WebGL2 mavjud emas");
    this.gl = gl;

    this.program = this.link(VERT, FRAG);
    gl.useProgram(this.program);
    for (const name of ["uTex", "uRes", "uCenter", "uHalf", "uRadius", "uMag", "uRefract", "uChroma", "uChromaDir", "uAlpha", "uLift"]) {
      this.u[name] = gl.getUniformLocation(this.program, name);
    }

    this.vao = gl.createVertexArray()!;
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.u["uTex"]!, 0);
  }

  private link(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    return p;
  }

  resize(pw: number, ph: number) {
    if (this.gl.canvas.width !== pw || this.gl.canvas.height !== ph) {
      this.gl.canvas.width = pw;
      this.gl.canvas.height = ph;
    }
    this.gl.viewport(0, 0, pw, ph);
  }

  upload(source: HTMLCanvasElement) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    if (source.width !== this.texW || source.height !== this.texH) {
      this.texW = source.width;
      this.texH = source.height;
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, source.width, source.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }
  }

  draw(p: any) {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform2f(this.u["uRes"]!, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(this.u["uCenter"]!, p.cx, p.cy);
    gl.uniform2f(this.u["uHalf"]!, p.hw, p.hh);
    gl.uniform1f(this.u["uRadius"]!, p.radius);
    gl.uniform1f(this.u["uMag"]!, p.mag);
    gl.uniform1f(this.u["uRefract"]!, p.refract);
    gl.uniform1f(this.u["uChroma"]!, p.chroma);
    gl.uniform2f(this.u["uChromaDir"]!, p.chromaDirX, p.chromaDirY);
    gl.uniform1f(this.u["uAlpha"]!, p.alpha);
    gl.uniform1f(this.u["uLift"]!, p.lift);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose() {
    this.gl.deleteTexture(this.tex);
    this.gl.deleteProgram(this.program);
    this.gl.deleteVertexArray(this.vao);
  }
}

// ==========================================
// 5. ASOSIY REACT KOMPONENTI (App)
// ==========================================
export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [selected, setSelected] = useState<TabId | "profile">(2);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let renderer: LensRenderer;
    try {
      renderer = new LensRenderer(canvas);
    } catch (e) {
      return;
    }

    const scene = new NavScene();
    let layout = computeLayout(window.innerWidth, window.innerHeight, Math.min(window.devicePixelRatio || 1, 3), 0);

    let committed: TabId | "profile" = 2;
    let pending: TabId | "profile" = 2;
    let pressing = false;
    let didDrag = false;
    let downX = 0;
    let pointerId = -1;

    const x = new Spring(layout.tabCenters[2]!);
    const lift = new Spring(0);
    const alpha = new Spring(1);
    lift.tune(900, dampingForStiffness(900, 1));
    alpha.tune(900, dampingForStiffness(900, 1));

    let lastT = performance.now();
    let dtRef = 1 / 60;
    let dirty = true;
    let idleFrames = 0;
    let chromaDirX = 0.28;
    let chromaDirY = -0.45;

    function wake() {
      idleFrames = 0;
      dirty = true;
    }

    function relayout() {
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      layout = computeLayout(window.innerWidth, window.innerHeight, dpr, 0);
      renderer.resize(Math.round(layout.width * dpr), Math.round(layout.height * dpr));
      canvas!.style.width = `${layout.width}px`;
      canvas!.style.height = `${layout.height}px`;
      const anchor = pending === "profile" ? committed : pending;
      if (anchor !== "profile") x.set(layout.tabCenters[anchor]!);
      wake();
    }

    function moveTo(toX: number) {
      const k = stiffnessForDistance(Math.abs(toX - x.x));
      x.tune(k, dampingForStiffness(k, 1));
      x.target = toX;
      wake();
    }

    function setPending(next: TabId | "profile") {
      if (next === pending) return;
      const wasProfile = pending === "profile";
      pending = next;
      if (next === "profile") {
        alpha.target = 0;
      } else {
        alpha.target = 1;
        if (wasProfile) x.set(layout.tabCenters[next]!);
        else moveTo(layout.tabCenters[next]!);
      }
      wake();
    }

    function clampToIsland(px: number) {
      const minX = layout.island.x + layout.lensRest.w * 0.5;
      const maxX = layout.island.x + layout.island.w - layout.lensRest.w * 0.5;
      return Math.max(minX, Math.min(maxX, px));
    }

    function onDown(ev: PointerEvent) {
      if (pressing) return;
      const r = canvas!.getBoundingClientRect();
      const px = ev.clientX - r.left;
      const py = ev.clientY - r.top;
      const hit = hitTest(px, py, layout);
      if (!hit) return;
      pointerId = ev.pointerId;
      pressing = true;
      didDrag = false;
      downX = px;
      lift.target = 1;
      setPending(hit.kind === "profile" ? "profile" : hit.index);
    }

    function onMove(ev: PointerEvent) {
      if (!pressing || ev.pointerId !== pointerId) return;
      const r = canvas!.getBoundingClientRect();
      const px = ev.clientX - r.left;
      const py = ev.clientY - r.top;
      if (!didDrag && Math.abs(px - downX) > 8) didDrag = true;
      if (!didDrag) return;

      const overProfile = Math.hypot(px - layout.profile.cx, py - layout.profile.cy) <= layout.profile.r + 10;
      if (overProfile) {
        setPending("profile");
        return;
      }
      if (pending === "profile") setPending(nearestTab(clampToIsland(px), layout));
      x.drive(clampToIsland(px), dtRef);
      pending = nearestTab(x.x, layout);
      wake();
    }

    function onUp(ev: PointerEvent) {
      if (!pressing || ev.pointerId !== pointerId) return;
      pointerId = -1;
      pressing = false;
      lift.target = 0;
      if (pending !== "profile") moveTo(layout.tabCenters[pending]!);
      committed = pending;
      setSelected(pending);
    }

    let raf = 0;
    function frame(now: number) {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(0.032, Math.max(0.001, (now - lastT) / 1000));
      lastT = now;
      dtRef = dt;

      const highlight: Highlight = pressing ? null : committed;
      if (scene.render(layout, highlight)) {
        renderer.upload(scene.canvas);
        dirty = true;
      }

      x.step(dt);
      lift.step(dt);
      alpha.step(dt);

      const moving = pressing || !x.settled || !lift.settled || !alpha.settled;
      if (!moving && !dirty) {
        if (idleFrames++ > 2) return;
      } else {
        idleFrames = 0;
      }
      dirty = false;

      const dpr = layout.dpr;
      const l = Math.max(0, Math.min(1, lift.x));
      const contract = 0.28 + 0.72 * alpha.x;
      const w = (layout.lensRest.w + (layout.lensLift.w - layout.lensRest.w) * l) * contract;
      const h = (layout.lensRest.h + (layout.lensLift.h - layout.lensRest.h) * l) * contract;
      const cy = layout.centerY + layout.lensLift.dy * l;
      const speed = Math.max(-1, Math.min(1, x.v / 2600));

      chromaDirX += (speed * 0.9 + 0.28 - chromaDirX) * Math.min(1, dt * 12);
      chromaDirY += (-0.45 - chromaDirY) * Math.min(1, dt * 12);

      renderer.draw({
        cx: x.x * dpr,
        cy: (cy + 2) * dpr,
        hw: (w / 2) * dpr,
        hh: (h / 2) * dpr,
        radius: (h / 2) * 0.94 * dpr,
        mag: 0.16 + 0.34 * l,
        refract: 0.1 + 0.22 * l,
        chroma: 0.1 + 0.46 * l + Math.abs(speed) * 0.16,
        chromaDirX,
        chromaDirY,
        alpha: alpha.x,
        lift: l,
      });
    }

    relayout();
    lastT = performance.now();
    raf = requestAnimationFrame(frame);

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    window.addEventListener("resize", relayout);

    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      window.removeEventListener("resize", relayout);
      renderer.dispose();
    };
  }, []);

  return (
    <div style={{ margin: 0, padding: 0, width: "100vw", height: "100vh", overflow: "hidden", background: "#0e0e11" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block", touchAction: "none" }} />
    </div>
  );
}