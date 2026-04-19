"""
download_ifc_samples.py
────────────────────────
Downloads publicly available IFC sample files for benchmark testing.

All URLs verified against GitHub API — raw.githubusercontent.com links
for files confirmed to be actual IFC content (not Git LFS pointers).

Run:
  python scripts/download_ifc_samples.py

Files are saved to data/. Re-running skips already-downloaded files.
After downloading, run the indexer to add them to ChromaDB + Neo4j.
"""
import pathlib
import urllib.request
import sys

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

# Official buildingSMART PCERT sample scene — IFC4, real file content (not LFS stubs).
# Includes architecture + HVAC + structural disciplines — ideal for MEP + cross-discipline tests.
_BASE = "https://raw.githubusercontent.com/buildingSMART/Sample-Test-Files/main/IFC%204.0.2.1%20(IFC%204)/PCERT-Sample-Scene"

_FILES = [
    {
        "name":        "PCERT-Building-Architecture.ifc",
        "url":         f"{_BASE}/Building-Architecture.ifc",
        "description": "PCERT office building — architectural discipline, IFC4, ~225 KB",
    },
    {
        "name":        "PCERT-Building-Hvac.ifc",
        "url":         f"{_BASE}/Building-Hvac.ifc",
        "description": "PCERT office building — HVAC/MEP discipline, IFC4, ~180 KB. "
                       "Contains IfcFan, IfcDuctSegment, IfcValve etc. — "
                       "first IFC in our benchmark with real MEP elements.",
    },
    {
        "name":        "PCERT-Building-Structural.ifc",
        "url":         f"{_BASE}/Building-Structural.ifc",
        "description": "PCERT office building — structural discipline, IFC4, ~297 KB. "
                       "Rich beam/column/slab data with Pset_BeamCommon properties.",
    },
    {
        "name":        "PCERT-Infra-Plumbing.ifc",
        "url":         f"{_BASE}/Infra-Plumbing.ifc",
        "description": "PCERT infrastructure — plumbing discipline, IFC4, ~512 KB. "
                       "Tests IfcPipeSegment, IfcPipeFitting retrieval at scale.",
    },
]


def download(entry: dict) -> bool:
    dest = _DATA_DIR / entry["name"]
    if dest.exists():
        size_kb = dest.stat().st_size // 1024
        print(f"  ✓ {entry['name']} already exists ({size_kb} KB) — skipping.")
        return True

    print(f"  ↓  {entry['name']}")
    print(f"     {entry['description']}")
    try:
        req = urllib.request.Request(
            entry["url"],
            headers={"User-Agent": "bim-graph-benchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            data = resp.read()
            f.write(data)
        size_kb = len(data) // 1024
        print(f"     → {size_kb} KB saved to {dest.name}")
        return True
    except Exception as e:
        print(f"     ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def inspect_ifc(path: pathlib.Path) -> None:
    try:
        import ifcopenshell
        model   = ifcopenshell.open(str(path))
        schema  = model.schema
        storeys = model.by_type("IfcBuildingStorey")
        if not storeys:
            print(f"    Schema: {schema} | No IfcBuildingStorey found")
            # Count top-level products
            total = len(list(model.by_type("IfcProduct")))
            print(f"    Total IfcProduct elements: {total}")
            return
        print(f"    Schema: {schema}")
        for s in storeys:
            count = sum(
                len(r.RelatedElements)
                for r in model.by_type("IfcRelContainedInSpatialStructure")
                if r.RelatingStructure == s
            )
            types = {}
            for r in model.by_type("IfcRelContainedInSpatialStructure"):
                if r.RelatingStructure == s:
                    for el in r.RelatedElements:
                        types[el.is_a()] = types.get(el.is_a(), 0) + 1
            top_types = sorted(types.items(), key=lambda x: -x[1])[:4]
            type_str  = ", ".join(f"{t}×{n}" for t, n in top_types)
            print(f"    Storey: {s.Name!r:<28} {count:>4} elements  [{type_str}]")
    except Exception as e:
        print(f"    Could not inspect: {e}")


def main():
    print(f"IFC sample downloader — saving to {_DATA_DIR}\n")
    ok = 0
    for entry in _FILES:
        if download(entry):
            ok += 1
        print()

    print(f"{'─'*60}")
    print(f"{ok}/{len(_FILES)} files downloaded.\n")
    print("Storey structure inspection:")
    print("─" * 60)
    for entry in _FILES:
        path = _DATA_DIR / entry["name"]
        if path.exists():
            print(f"\n{entry['name']}  ({path.stat().st_size // 1024} KB)")
            inspect_ifc(path)

    if ok == len(_FILES):
        print(f"\n✓ All files ready.")
        print("Next steps:")
        print("  1. Load into Neo4j:  python -m graph_db.loader  (from src/)")
        print("  2. Re-index Chroma:  python -m indexer.spatial_indexer  (from src/)")
    else:
        print("\nSome downloads failed — check network or update URLs in this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
