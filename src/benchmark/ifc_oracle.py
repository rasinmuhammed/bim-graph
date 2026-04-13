"""
ifc_oracle.py
─────────────
Deterministic ground truth engine using IfcOpenShell.
Returns the exact set of elements on a given floor — used to score
both the baseline and the agentic pipeline.
"""
import pathlib
import ifcopenshell

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_IFC_PATH     = str(_PROJECT_ROOT / "data" / "Duplex_A_20110907.ifc")


def get_floor_elements(target_floor: str, ifc_path: str = _IFC_PATH) -> dict:
    """
    Pull every element directly contained in the given IfcBuildingStorey.

    Returns
    -------
    {
        "matched_storey": str | None,
        "total_elements": int,
        "elements": [{"name": str, "type": str, "guid": str}, ...],
        "element_types": {type: count},
    }
    """
    model         = ifcopenshell.open(ifc_path)
    target_storey = None

    for storey in model.by_type("IfcBuildingStorey"):
        if target_floor.lower() in storey.Name.lower():
            target_storey = storey
            break

    if target_storey is None:
        return {
            "matched_storey": None,
            "total_elements": 0,
            "elements":       [],
            "element_types":  {},
        }

    elements     = []
    element_types: dict[str, int] = {}

    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == target_storey:
            for el in rel.RelatedElements:
                ifc_type = el.is_a()
                elements.append({
                    "name": el.Name,
                    "type": ifc_type,
                    "guid": el.GlobalId,
                })
                element_types[ifc_type] = element_types.get(ifc_type, 0) + 1

    return {
        "matched_storey": target_storey.Name,
        "total_elements": len(elements),
        "elements":       elements,
        "element_types":  element_types,
    }


def list_all_floors(ifc_path: str = _IFC_PATH) -> list[dict]:
    """Return every IfcBuildingStorey with name, elevation and element count."""
    model  = ifcopenshell.open(ifc_path)
    floors = []
    for storey in model.by_type("IfcBuildingStorey"):
        gt = get_floor_elements(storey.Name, ifc_path)
        floors.append({
            "name":           storey.Name,
            "elevation_m":    storey.Elevation,
            "element_count":  gt["total_elements"],
        })
    return floors


if __name__ == "__main__":
    import json
    print("\n=== IFC Floor Summary ===")
    for f in list_all_floors():
        print(f"  {f['name']:10} | elevation: {f['elevation_m']:>7.2f}m | elements: {f['element_count']}")
