"use client";

import { useEffect, useRef } from "react";
import type { Mesh, Material, MeshLambertMaterial } from "three";

interface Props {
  ifcUrl: string | null;
  highlightGuids: string[];
}

interface ViewerHandle {
  load: (url: string) => Promise<void>;
  highlight: (guids: string[]) => Promise<void>;
  dispose: () => void;
}

async function createViewer(container: HTMLDivElement): Promise<ViewerHandle> {
  const OBC    = await import("@thatopen/components");
  const OBCF   = await import("@thatopen/components-front");
  const THREE  = await import("three");
  const WEBIFC = await import("web-ifc");

  // ── Scene ────────────────────────────────────────────────────────────────
  const components = new OBC.Components();
  const worlds     = components.get(OBC.Worlds);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const world = worlds.create() as any;

  world.scene    = new OBC.SimpleScene(components);
  world.renderer = new OBCF.PostproductionRenderer(components, container);
  world.camera   = new OBC.OrthoPerspectiveCamera(components);

  components.init();
  world.scene.setup();
  world.camera.controls.setLookAt(12, 6, 8, 0, 0, -10);
  world.scene.three.background = new THREE.Color(0x0a0a0a);

  const grids = components.get(OBC.Grids);
  grids.create(world);

  // ── web-ifc (single-thread, no @thatopen/fragments) ──────────────────────
  const ifcApi = new WEBIFC.IfcAPI();
  ifcApi.SetWasmPath("/", true);
  await ifcApi.Init(undefined, true); // forceSingleThread=true — guaranteed ST WASM

  // ── State ────────────────────────────────────────────────────────────────
  const guidMap     = new Map<string, Mesh[]>();
  let   modelMeshes: Mesh[] = [];
  let   highlighted: Mesh[] = [];

  const clearScene = () => {
    for (const mesh of modelMeshes) {
      world.scene.three.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as Material).dispose();
    }
    modelMeshes = [];
    guidMap.clear();
    highlighted = [];
  };

  // ── Load ─────────────────────────────────────────────────────────────────
  const load = async (url: string) => {
    clearScene();

    const resp   = await fetch(url);
    const buffer = await resp.arrayBuffer();
    const modelID = ifcApi.OpenModel(new Uint8Array(buffer));
    ifcApi.CreateIfcGuidToExpressIdMapping(modelID);

    const sceneBBox = new THREE.Box3();

    ifcApi.StreamAllMeshes(modelID, (flatMesh) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const guid = (ifcApi as any).GetGuidFromExpressId?.(modelID, flatMesh.expressID) as string | undefined;
      const meshGroup: Mesh[] = [];

      for (let gi = 0; gi < flatMesh.geometries.size(); gi++) {
        const placed   = flatMesh.geometries.get(gi);
        const geomData = ifcApi.GetGeometry(modelID, placed.geometryExpressID);

        const rawVerts   = ifcApi.GetVertexArray(geomData.GetVertexData(), geomData.GetVertexDataSize());
        const rawIndices = ifcApi.GetIndexArray(geomData.GetIndexData(), geomData.GetIndexDataSize());
        geomData.delete?.();

        // web-ifc interleaves position + normal: [x,y,z,nx,ny,nz, ...]
        const vertCount = rawVerts.length / 6;
        const positions = new Float32Array(vertCount * 3);
        const normals   = new Float32Array(vertCount * 3);
        for (let v = 0, p = 0; v < rawVerts.length; v += 6, p += 3) {
          positions[p]     = rawVerts[v];
          positions[p + 1] = rawVerts[v + 1];
          positions[p + 2] = rawVerts[v + 2];
          normals[p]       = rawVerts[v + 3];
          normals[p + 1]   = rawVerts[v + 4];
          normals[p + 2]   = rawVerts[v + 5];
        }

        const bufGeom = new THREE.BufferGeometry();
        bufGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        bufGeom.setAttribute("normal",   new THREE.BufferAttribute(normals, 3));
        // copy indices out of WASM memory before CloseModel
        bufGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(rawIndices), 1));

        const { x, y, z, w } = placed.color;
        const mat = new THREE.MeshLambertMaterial({
          color:       new THREE.Color(x, y, z),
          transparent: w < 0.99,
          opacity:     w,
          side:        THREE.DoubleSide,
        });

        const mesh = new THREE.Mesh(bufGeom, mat);
        mesh.matrix.fromArray(placed.flatTransformation);
        mesh.matrixAutoUpdate = false;
        if (guid) mesh.userData.guid = guid;

        world.scene.three.add(mesh);
        modelMeshes.push(mesh);
        meshGroup.push(mesh);

        // expand scene bounding box for camera fitting
        const mb = new THREE.Box3().setFromObject(mesh);
        sceneBBox.union(mb);
      }

      flatMesh.delete?.();
      if (guid && meshGroup.length) guidMap.set(guid, meshGroup);
    });

    // geometry is now in JS heap — safe to close model
    ifcApi.CloseModel(modelID);

    // fit camera
    if (!sceneBBox.isEmpty()) {
      const center = sceneBBox.getCenter(new THREE.Vector3());
      const size   = sceneBBox.getSize(new THREE.Vector3());
      const dist   = Math.max(size.x, size.y, size.z) * 1.8;
      world.camera.controls.setLookAt(
        center.x + dist, center.y + dist * 0.6, center.z + dist,
        center.x, center.y, center.z,
        true,
      );
    }
  };

  // ── Highlight ─────────────────────────────────────────────────────────────
  const HIGHLIGHT = new THREE.Color(0xff6600);

  const highlight = async (guids: string[]) => {
    for (const mesh of highlighted) {
      (mesh.material as MeshLambertMaterial).emissive.set(0x000000);
    }
    highlighted = [];

    for (const guid of guids) {
      for (const mesh of guidMap.get(guid) ?? []) {
        (mesh.material as MeshLambertMaterial).emissive.copy(HIGHLIGHT);
        highlighted.push(mesh);
      }
    }
  };

  // ── Dispose ───────────────────────────────────────────────────────────────
  const dispose = () => {
    clearScene();
    components.dispose();
  };

  return { load, highlight, dispose };
}

export default function IfcViewer({ ifcUrl, highlightGuids }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef    = useRef<ViewerHandle | null>(null);
  const currentUrl   = useRef<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    createViewer(containerRef.current).then(handle => {
      viewerRef.current = handle;
      if (ifcUrl) {
        handle.load(ifcUrl).catch(console.error);
        currentUrl.current = ifcUrl;
      }
    });
    return () => {
      viewerRef.current?.dispose();
      viewerRef.current = null;
      currentUrl.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!viewerRef.current || !ifcUrl) return;
    if (ifcUrl === currentUrl.current) return;
    currentUrl.current = ifcUrl;
    viewerRef.current.load(ifcUrl).catch(console.error);
  }, [ifcUrl]);

  useEffect(() => {
    viewerRef.current?.highlight(highlightGuids);
  }, [highlightGuids]);

  return <div ref={containerRef} className="w-full h-full" style={{ background: "#0a0a0a" }} />;
}
