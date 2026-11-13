import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { easeInOutCubic, type GalaxyRendererProps } from './galaxyTypes';

const R = 10; // world scale: coords in [-1,1] → [-10,10]
const DIM_FACTOR = 0.1;

function spriteTexture(): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.25, 'rgba(255,255,255,0.9)');
  grd.addColorStop(0.5, 'rgba(255,255,255,0.25)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

interface SceneState {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  points: THREE.Points;
  hits: THREE.Points;
  marker: THREE.Points;
  baseColors: Float32Array;
  positions: Float32Array;
  ids: Int32Array;
  cl: Int32Array;
  indexOf: Map<number, number>;
  disposables: { dispose: () => void }[];
}

export default function Galaxy3D(props: GalaxyRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<SceneState | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const [glError, setGlError] = useState<string | null>(null);
  const tweenRef = useRef<{ fromT: THREE.Vector3; toT: THREE.Vector3; fromP: THREE.Vector3; toP: THREE.Vector3; t0: number; dur: number } | null>(null);

  // ---------- build scene once per dataset ----------
  useEffect(() => {
    const el = containerRef.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (e) {
      setGlError((e as Error)?.message || 'WebGL is not available');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x020617, 1);
    el.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.touchAction = 'none';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('aria-label', '3D galaxy of images. Drag to orbit, scroll to zoom.');

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020617, 0.012);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 500);
    camera.position.set(0, 4, 26);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.7;
    controls.minDistance = 0.5;
    controls.maxDistance = 120;
    controls.autoRotateSpeed = 0.6;

    const disposables: { dispose: () => void }[] = [controls];
    const tex = spriteTexture();
    disposables.push(tex);

    // data
    const pts = propsRef.current.data.points;
    const n = pts.length;
    const positions = new Float32Array(n * 3);
    const baseColors = new Float32Array(n * 3);
    const ids = new Int32Array(n);
    const cl = new Int32Array(n);
    const indexOf = new Map<number, number>();
    const tmp = new THREE.Color();
    pts.forEach((p, i) => {
      positions[i * 3] = p.x * R;
      positions[i * 3 + 1] = p.y * R;
      positions[i * 3 + 2] = (p.z ?? 0) * R;
      ids[i] = p.image_id;
      cl[i] = p.cluster_id ?? -1;
      indexOf.set(p.image_id, i);
      tmp.set(propsRef.current.clusterColors.get(cl[i]) ?? '#94a3b8');
      baseColors[i * 3] = tmp.r;
      baseColors[i * 3 + 1] = tmp.g;
      baseColors[i * 3 + 2] = tmp.b;
    });

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(baseColors.slice(), 3));
    geom.computeBoundingSphere();
    const mat = new THREE.PointsMaterial({
      size: 0.36,
      map: tex,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(geom, mat);
    scene.add(points);
    disposables.push(geom, mat);

    // highlighted hits layer
    const hitGeom = new THREE.BufferGeometry();
    hitGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    hitGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
    const hitMat = new THREE.PointsMaterial({ size: 0.95, map: tex, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    const hits = new THREE.Points(hitGeom, hitMat);
    scene.add(hits);
    disposables.push(hitGeom, hitMat);

    // hover / focus marker
    const markerGeom = new THREE.BufferGeometry();
    markerGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    const markerMat = new THREE.PointsMaterial({ size: 1.6, map: tex, color: 0xe0e7ff, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
    const marker = new THREE.Points(markerGeom, markerMat);
    marker.visible = false;
    scene.add(marker);
    disposables.push(markerGeom, markerMat);

    // background stars
    const starN = 1800;
    const starPos = new Float32Array(starN * 3);
    for (let i = 0; i < starN; i++) {
      const u = Math.random() * 2 - 1;
      const th = Math.random() * Math.PI * 2;
      const rad = 70 + Math.random() * 90;
      const s = Math.sqrt(1 - u * u);
      starPos[i * 3] = rad * s * Math.cos(th);
      starPos[i * 3 + 1] = rad * u;
      starPos[i * 3 + 2] = rad * s * Math.sin(th);
    }
    const starGeom = new THREE.BufferGeometry();
    starGeom.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ size: 1.4, sizeAttenuation: false, color: 0x94a3b8, transparent: true, opacity: 0.55, depthWrite: false, fog: false });
    scene.add(new THREE.Points(starGeom, starMat));
    disposables.push(starGeom, starMat);

    // centre target on the data's bounding sphere
    const bs = geom.boundingSphere;
    if (bs) {
      controls.target.copy(bs.center);
      camera.position.copy(bs.center).add(new THREE.Vector3(0, bs.radius * 0.35, Math.max(6, bs.radius * 2.3)));
    }
    controls.update();

    const st: SceneState = { renderer, scene, camera, controls, points, hits, marker, baseColors, positions, ids, cl, indexOf, disposables };
    stateRef.current = st;

    // resize
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    // picking
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let pointer: { x: number; y: number } | null = null;
    let hoverIdx = -1;
    let down: { x: number; y: number } | null = null;
    let lastS = { sx: -1, sy: -1, i: -1 };

    const pickAt = (x: number, y: number): number => {
      const w = el.clientWidth, h = el.clientHeight;
      ndc.set((x / w) * 2 - 1, -(y / h) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const dist = camera.position.distanceTo(controls.target);
      raycaster.params.Points = { threshold: Math.max(0.05, dist * 0.012) };
      const hitsI = raycaster.intersectObject(points, false);
      const { activeClusters } = propsRef.current;
      let best = -1;
      let bestD = Infinity;
      for (const h of hitsI.slice(0, 40)) {
        const i = h.index!;
        if (activeClusters.size && !activeClusters.has(cl[i])) continue;
        const d = (h.distanceToRay ?? 0) + h.distance * 0.002;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best;
    };

    const screenOf = (i: number) => {
      const v = new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).project(camera);
      return { sx: ((v.x + 1) / 2) * el.clientWidth, sy: ((1 - v.y) / 2) * el.clientHeight };
    };

    const local = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const onMove = (e: PointerEvent) => {
      pointer = down ? null : local(e);
    };
    const onDown = (e: PointerEvent) => {
      down = local(e);
      tweenRef.current = null;
    };
    const onUp = (e: PointerEvent) => {
      const p = local(e);
      if (down && Math.hypot(p.x - down.x, p.y - down.y) < 4) {
        const i = pickAt(p.x, p.y);
        if (i >= 0) propsRef.current.onSelect(ids[i]);
      }
      down = null;
    };
    const onLeave = () => {
      pointer = null;
      lastS = { sx: -1, sy: -1, i: -1 };
      if (hoverIdx >= 0) {
        hoverIdx = -1;
        propsRef.current.onHover(null);
      }
    };
    const cnv = renderer.domElement;
    cnv.addEventListener('pointermove', onMove);
    cnv.addEventListener('pointerdown', onDown);
    cnv.addEventListener('pointerup', onUp);
    cnv.addEventListener('pointerleave', onLeave);

    // loop
    let raf = 0;
    let lastPick = 0;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const tw = tweenRef.current;
      if (tw) {
        const t = Math.min(1, (now - tw.t0) / tw.dur);
        const e = easeInOutCubic(t);
        controls.target.lerpVectors(tw.fromT, tw.toT, e);
        camera.position.lerpVectors(tw.fromP, tw.toP, e);
        if (t >= 1) tweenRef.current = null;
      }
      controls.autoRotate = Boolean(propsRef.current.autoRotate) && !tweenRef.current;
      controls.update();

      if (pointer && now - lastPick > 30) {
        lastPick = now;
        const i = pickAt(pointer.x, pointer.y);
        if (i !== hoverIdx) {
          hoverIdx = i;
          if (i < 0) propsRef.current.onHover(null);
        }
        if (i >= 0) {
          const s = screenOf(i);
          if (Math.abs(s.sx - lastS.sx) > 0.5 || Math.abs(s.sy - lastS.sy) > 0.5 || i !== lastS.i) {
            lastS = { ...s, i };
            propsRef.current.onHover({ imageId: ids[i], clusterId: cl[i], ...s });
          }
        } else lastS = { sx: -1, sy: -1, i: -1 };
      } else if (!pointer && hoverIdx >= 0 && (controls.autoRotate || tw)) {
        hoverIdx = -1;
        propsRef.current.onHover(null);
      }

      // marker: hovered point, else focused point
      const fId = propsRef.current.focusId;
      const mi = hoverIdx >= 0 ? hoverIdx : fId != null ? indexOf.get(fId) ?? -1 : -1;
      if (mi >= 0) {
        const attr = marker.geometry.getAttribute('position') as THREE.BufferAttribute;
        attr.setXYZ(0, positions[mi * 3], positions[mi * 3 + 1], positions[mi * 3 + 2]);
        attr.needsUpdate = true;
        marker.visible = true;
        (marker.material as THREE.PointsMaterial).size = hoverIdx >= 0 ? 1.4 : 1.4 + Math.sin(now / 220) * 0.5;
      } else marker.visible = false;

      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      cnv.removeEventListener('pointermove', onMove);
      cnv.removeEventListener('pointerdown', onDown);
      cnv.removeEventListener('pointerup', onUp);
      cnv.removeEventListener('pointerleave', onLeave);
      disposables.forEach((d) => d.dispose());
      scene.clear();
      renderer.dispose();
      renderer.forceContextLoss();
      cnv.remove();
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.data]);

  // ---------- colours: highlight / cluster filter ----------
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const { highlight, activeClusters, clusterColors } = props;
    const n = st.ids.length;
    const tmp = new THREE.Color();
    for (let i = 0; i < n; i++) {
      tmp.set(clusterColors.get(st.cl[i]) ?? '#94a3b8');
      st.baseColors[i * 3] = tmp.r;
      st.baseColors[i * 3 + 1] = tmp.g;
      st.baseColors[i * 3 + 2] = tmp.b;
    }
    const attr = st.points.geometry.getAttribute('color') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const hitIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      let f = 1;
      if (activeClusters.size && !activeClusters.has(st.cl[i])) f = DIM_FACTOR * 0.6;
      else if (highlight) {
        if (highlight.has(st.ids[i])) hitIdx.push(i);
        f = DIM_FACTOR * 1.6;
      }
      arr[i * 3] = st.baseColors[i * 3] * f;
      arr[i * 3 + 1] = st.baseColors[i * 3 + 1] * f;
      arr[i * 3 + 2] = st.baseColors[i * 3 + 2] * f;
    }
    attr.needsUpdate = true;

    const hp = new Float32Array(hitIdx.length * 3);
    const hc = new Float32Array(hitIdx.length * 3);
    hitIdx.forEach((i, k) => {
      hp[k * 3] = st.positions[i * 3];
      hp[k * 3 + 1] = st.positions[i * 3 + 1];
      hp[k * 3 + 2] = st.positions[i * 3 + 2];
      // brighten towards white
      hc[k * 3] = 0.5 + st.baseColors[i * 3] * 0.6;
      hc[k * 3 + 1] = 0.5 + st.baseColors[i * 3 + 1] * 0.6;
      hc[k * 3 + 2] = 0.5 + st.baseColors[i * 3 + 2] * 0.6;
    });
    const g = st.hits.geometry;
    g.setAttribute('position', new THREE.BufferAttribute(hp, 3));
    g.setAttribute('color', new THREE.BufferAttribute(hc, 3));
    g.computeBoundingSphere();
  }, [props.highlight, props.activeClusters, props.clusterColors, props.data, glError]);

  // ---------- fly ----------
  useEffect(() => {
    const st = stateRef.current;
    const f = props.fly;
    if (!st || !f) return;
    const toT = new THREE.Vector3(f.x * R, f.y * R, (f.z ?? 0) * R);
    const dir = st.camera.position.clone().sub(st.controls.target).normalize();
    const dist = Math.max(5, f.spread * R * 3.2);
    tweenRef.current = {
      fromT: st.controls.target.clone(),
      toT,
      fromP: st.camera.position.clone(),
      toP: toT.clone().add(dir.multiplyScalar(dist)),
      t0: performance.now(),
      dur: 900,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.fly?.nonce]);

  return (
    <div ref={containerRef} className="absolute inset-0 overflow-hidden">
      {glError && (
        <div className="flex h-full items-center justify-center p-6 text-center text-sm text-slate-400">
          3D mode needs WebGL, which is unavailable here ({glError}). Switch to 2D.
        </div>
      )}
    </div>
  );
}
