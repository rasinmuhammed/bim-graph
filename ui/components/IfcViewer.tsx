"use client";

import { useEffect, useRef } from "react";

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
  const OBC   = await import("@thatopen/components");
  const OBCF  = await import("@thatopen/components-front");
  const THREE  = await import("three");

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

  const ifcLoader   = components.get(OBC.IfcLoader);
  const fragManager = components.get(OBC.FragmentsManager);
  const highlighter = components.get(OBCF.Highlighter);

  // FragmentsManager must be initialized before IfcLoader or Highlighter
  fragManager.init("/fragments-worker.mjs");

  // Point IfcLoader at the wasm files we serve from /public
  ifcLoader.settings.wasm.path    = "/";
  ifcLoader.settings.wasm.absolute = true;
  await ifcLoader.setup();

  highlighter.setup({ world });

  let currentModel: Awaited<ReturnType<typeof ifcLoader.load>> | null = null;

  const load = async (url: string) => {
    if (currentModel) {
      world.scene.three.remove(currentModel.object);
      fragManager.dispose();
      currentModel = null;
    }
    const resp   = await fetch(url);
    const buffer = await resp.arrayBuffer();
    const model  = await ifcLoader.load(new Uint8Array(buffer), true, url.split("/").pop() ?? "model");
    world.scene.three.add(model.object);
    currentModel = model;

    // fit camera around model bounding box
    const bbox = new THREE.Box3().setFromObject(model.object);
    const center = bbox.getCenter(new THREE.Vector3());
    const size   = bbox.getSize(new THREE.Vector3());
    const dist   = Math.max(size.x, size.y, size.z) * 1.8;
    world.camera.controls.setLookAt(
      center.x + dist, center.y + dist * 0.6, center.z + dist,
      center.x, center.y, center.z,
      true,
    );
  };

  const highlight = async (guids: string[]) => {
    highlighter.clear("select");
    if (!guids.length) return;
    try {
      const modelIdMap = await fragManager.guidsToModelIdMap(guids);
      await highlighter.highlightByID("select", modelIdMap, true, true);
    } catch {
      // model not yet loaded or guids not found
    }
  };

  const dispose = () => components.dispose();

  return { load, highlight, dispose };
}

export default function IfcViewer({ ifcUrl, highlightGuids }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef    = useRef<ViewerHandle | null>(null);
  const currentUrl   = useRef<string | null>(null);

  // initialise viewer once
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

  // reload when file changes
  useEffect(() => {
    if (!viewerRef.current || !ifcUrl) return;
    if (ifcUrl === currentUrl.current) return;
    currentUrl.current = ifcUrl;
    viewerRef.current.load(ifcUrl).catch(console.error);
  }, [ifcUrl]);

  // highlight when guids change
  useEffect(() => {
    viewerRef.current?.highlight(highlightGuids);
  }, [highlightGuids]);

  return <div ref={containerRef} className="w-full h-full" style={{ background: "#0a0a0a" }} />;
}
