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

def _find_storey(model, floor_name: str):
    """Case-insensitive exact storey lookup."""
    for storey in model.by_type("IfcBuildingStorey"):
        if storey.Name and storey.Name.lower().strip() == floor_name.lower().strip():
            return storey
    return None


def get_ground_truth_guids(ifc_path: str, floor_name: str) -> set[str]:
    """
    Returns the set of GUIDs for every element on the given floor.
    This is the oracle — deterministic, not LLM-dependent.
    """
    model  = ifcopenshell.open(ifc_path)
    target = _find_storey(model, floor_name)
    if target is None:
        return set()

    guids: set[str] = set()
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == target:
            for el in rel.RelatedElements:
                guids.add(el.GlobalId)
    return guids


def get_ground_truth_guids_by_types(
    ifc_path: str, floor_name: str, ifc_types: list[str]
) -> set[str]:
    """
    Returns the set of GUIDs for elements of the specified IFC types on the given floor.
    Used for type-specific queries (walls, doors, MEP) so the oracle only grades
    against elements the query actually asked about — not all floor elements.
    """
    model  = ifcopenshell.open(ifc_path)
    target = _find_storey(model, floor_name)
    if target is None:
        return set()

    type_set = {t.lower() for t in ifc_types}
    guids: set[str] = set()
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == target:
            for el in rel.RelatedElements:
                if el.is_a().lower() in type_set:
                    guids.add(el.GlobalId)
    return guids


def score_answer(answer: str, ground_truth_guids: set[str]) -> dict:
    """
    Extract GUIDs from the generated answer and compute P/R/F1 vs the oracle.

    Extraction strategy (ordered by precision):
      1. Labeled GUIDs: match 'GUID: <22chars>' — what the generate node outputs.
      2. Bare fallback: word-boundary-anchored 22-char base64 strings, only used
         if no labeled GUIDs exist (backward compatibility with old answer formats).

    The original regex '[A-Za-z0-9$_]{22}' was too broad — it matched any 22-char
    run in the text, including parts of URLs, long words run together, and formatted
    output. This inflated 'found' counts and produced misleadingly high precision.
    """
    import re

    # Primary: labeled GUIDs as the generate node always produces them
    found = set(re.findall(r"GUID:\s*([A-Za-z0-9$_]{22})", answer))

    # Fallback: bare 22-char base64url strings anchored at word boundaries
    if not found:
        found = set(re.findall(r"(?<![A-Za-z0-9$_])([A-Za-z0-9$_]{22})(?![A-Za-z0-9$_])", answer))

    if not ground_truth_guids:
        if not found:
            # System correctly said nothing exists — perfect score.
            return {
                "scoring_method":     "correctly_empty",
                "precision":          1.0,
                "recall":             1.0,
                "f1":                 1.0,
                "found":              0,
                "expected":           0,
                "hallucinated_guids": 0,
            }
        # System hallucinated GUIDs that don't exist in the oracle.
        return {
            "scoring_method":     "guid",
            "precision":          0.0,
            "recall":             0.0,
            "f1":                 0.0,
            "found":              len(found),
            "expected":           0,
            "hallucinated_guids": len(found),
        }

    tp = len(found & ground_truth_guids)
    fp = len(found - ground_truth_guids)
    fn = len(ground_truth_guids - found)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "scoring_method":    "guid",
        "precision":         round(precision, 3),
        "recall":            round(recall, 3),
        "f1":                round(f1, 3),
        "found":             len(found),
        "expected":          len(ground_truth_guids),
        "hallucinated_guids": fp,
    }


def score_cross_floor_answer(answer: str, ifc_path: str) -> dict:
    """
    Score a cross-floor query (target_floor=null) by checking whether the
    answer correctly identifies which floors exist and roughly how many
    elements each has. Uses keyword matching against the oracle floor list.

    Returns a 'coverage' score (0-1): fraction of floor names mentioned.
    This is less rigorous than GUID-level scoring but honest — cross-floor
    aggregation queries cannot be scored with GUID P/R/F1.
    """
    floors  = list_all_floors(ifc_path)
    total   = len(floors)
    if total == 0:
        return {"scoring_method": "cross_floor_coverage", "coverage": 0.0, "floors_found": 0, "floors_total": 0}

    answer_lower = answer.lower()
    found = sum(1 for f in floors if f["name"].lower() in answer_lower)

    return {
        "scoring_method": "cross_floor_coverage",
        "coverage":       round(found / total, 3),
        "f1":             round(found / total, 3),   # treat coverage as F1 for summary averaging
        "precision":      round(found / total, 3),
        "recall":         round(found / total, 3),
        "floors_found":   found,
        "floors_total":   total,
    }

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
    print("\n=== IFC Floor Summary ===")
    for f in list_all_floors():
        print(f"  {f['name']:10} | elevation: {f['elevation_m']:>7.2f}m | elements: {f['element_count']}")
