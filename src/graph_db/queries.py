"""
graph_db/queries.py
───────────────────
Neo4j Cypher query library for BIM-Graph.

GRAPH DATABASE PRIMER (read this before the code):
────────────────────────────────────────────────────
A graph database stores data as:
  • Nodes      — entities,   e.g. (:Storey {name: "Level 2"})
  • Labels     — categories, e.g. :Storey, :Element, :Building
  • Properties — key/value,  e.g. {guid: "3Ax...", ifc_type: "IfcPump"}
  • Relationships — directed edges, e.g. (:Storey)-[:CONTAINS]->(:Element)

Our IFC schema in Neo4j:
  (:Project {guid, name, file})
    └─[:HAS_SITE]─>
  (:Site {guid, name})
    └─[:HAS_BUILDING]─>
  (:Building {guid, name})
    └─[:HAS_STOREY]─>
  (:Storey {guid, name, elevation, file})
    └─[:CONTAINS]─>
  (:Element {guid, name, ifc_type, file})

"""

import logging
from functools import lru_cache
from neo4j import GraphDatabase, Driver
from config import settings

logger = logging.getLogger("bim_graph.graph_db")

# ── MEP types (same set as nodes.py — kept here to avoid circular imports) ─────
_MEP_TYPES: list[str] = [
    "IfcFlowTerminal", "IfcFlowFitting", "IfcFlowSegment",
    "IfcFlowController", "IfcDistributionFlowElement",
    "IfcEnergyConversionDevice", "IfcFlowMovingDevice",
    "IfcFlowStorageDevice", "IfcAirTerminal", "IfcFlowInstrument",
    "IfcValve", "IfcDuctFitting", "IfcDuctSegment",
    "IfcPipeFitting", "IfcPipeSegment",
    "IfcPump", "IfcFan", "IfcCompressor",
    "IfcBoiler", "IfcChiller", "IfcHeatExchanger", "IfcUnitaryEquipment",
    "IfcDistributionElement", "IfcDistributionControlElement",
    "IfcSanitaryTerminal", "IfcElectricAppliance",
    "IfcLightFixture", "IfcOutlet", "IfcSensor", "IfcActuator", "IfcController",
]


# ── Driver singleton ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_driver() -> Driver:
    """
    Return a cached Neo4j driver.
    The driver manages a connection pool internally — creating it once
    and reusing it is the correct pattern.
    """
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=10,
    )


def is_graph_available() -> bool:
    try:
        _get_driver().verify_connectivity()
        return True
    except Exception:
        return False


# ── Schema setup ────────────────────────────────────────────────────────────────
def ensure_schema() -> None:
    """
    Create uniqueness constraints and indexes.
    CONSTRAINT enforces that no two nodes share the same guid.
    INDEX speeds up MATCH queries on name and ifc_type fields.
    """
    constraints = [
        "CREATE CONSTRAINT storey_guid IF NOT EXISTS FOR (s:Storey)  REQUIRE s.guid IS UNIQUE",
        "CREATE CONSTRAINT element_guid IF NOT EXISTS FOR (e:Element) REQUIRE e.guid IS UNIQUE",
        "CREATE CONSTRAINT project_guid IF NOT EXISTS FOR (p:Project) REQUIRE p.guid IS UNIQUE",
        "CREATE CONSTRAINT building_guid IF NOT EXISTS FOR (b:Building) REQUIRE b.guid IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX element_ifc_type IF NOT EXISTS FOR (e:Element) ON (e.ifc_type)",
        "CREATE INDEX storey_name      IF NOT EXISTS FOR (s:Storey)  ON (s.name)",
        "CREATE INDEX element_file     IF NOT EXISTS FOR (e:Element) ON (e.file)",
        "CREATE INDEX storey_file      IF NOT EXISTS FOR (s:Storey)  ON (s.file)",
    ]
    with _get_driver().session() as session:
        for stmt in constraints + indexes:
            session.run(stmt)
    logger.info("Neo4j schema constraints and indexes ensured.")


def clear_graph() -> None:
    """Delete all BIM graph data. Used by the deterministic admin reindex job."""
    with _get_driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Neo4j graph cleared.")


# ── Read queries ────────────────────────────────────────────────────────────────

def get_all_elements_on_floor(floor: str, ifc_file: str) -> list[dict]:
    """
    Filter by e.file (not s.file) so federated IFC models work correctly —
    discipline files (MEP, Structural) share storey GUIDs with the Architecture
    file, so the storey's file tag is unreliable. Element file tags are authoritative.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND e.file = $file
    RETURN e.ifc_type AS ifc_type, e.name AS name, e.guid AS guid
    ORDER BY e.ifc_type, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floor=floor, file=ifc_file)
        return [r.data() for r in result]


def get_storey_details(floor: str, ifc_file: str) -> dict | None:
    """Return the canonical storey name/GUID/elevation for a resolved floor."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND e.file = $file
    RETURN s.name AS name, s.guid AS guid, s.elevation AS elevation_m
    LIMIT 1
    """
    with _get_driver().session() as session:
        record = session.run(cypher, floor=floor, file=ifc_file).single()
        return record.data() if record else None


def get_elements_on_floors(floors: list[str], ifc_file: str, ifc_types: list[str] | None = None) -> list[dict]:
    """Return elements on any of the supplied storeys, optionally filtered by IFC type."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name IN $floors AND e.file = $file
      AND ($ifc_types IS NULL OR e.ifc_type IN $ifc_types)
    RETURN s.name AS floor, e.ifc_type AS ifc_type, e.name AS name, e.guid AS guid
    ORDER BY s.elevation, e.ifc_type, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floors=floors, file=ifc_file, ifc_types=ifc_types)
        return [r.data() for r in result]


def get_elements_not_on_floor(floor: str, ifc_file: str, ifc_types: list[str] | None = None) -> list[dict]:
    """Return elements contained by storeys other than the excluded floor."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name <> $floor AND e.file = $file
      AND ($ifc_types IS NULL OR e.ifc_type IN $ifc_types)
    RETURN s.name AS floor, e.ifc_type AS ifc_type, e.name AS name, e.guid AS guid
    ORDER BY s.elevation, e.ifc_type, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floor=floor, file=ifc_file, ifc_types=ifc_types)
        return [r.data() for r in result]


def count_elements_by_storey(ifc_file: str, ifc_types: list[str] | None = None) -> list[dict]:
    """Return per-storey counts, optionally restricted to specific IFC types."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.file = $file
      AND ($ifc_types IS NULL OR e.ifc_type IN $ifc_types)
    RETURN s.name AS floor, s.guid AS storey_guid, s.elevation AS elevation_m, count(e) AS count
    ORDER BY count DESC, s.elevation
    """
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file, ifc_types=ifc_types)
        return [r.data() for r in result]


def count_elements_total(ifc_file: str, ifc_types: list[str] | None = None) -> list[dict]:
    """Return counts by IFC type across the whole model."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.file = $file AND ($ifc_types IS NULL OR e.ifc_type IN $ifc_types)
    RETURN e.ifc_type AS ifc_type, count(e) AS count
    ORDER BY count DESC, e.ifc_type
    """
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file, ifc_types=ifc_types)
        return [r.data() for r in result]


def get_mep_elements_on_floor(floor: str, ifc_file: str) -> list[dict]:
    """
    WHERE e.ifc_type IN $mep_types
      → Equivalent to SQL's IN clause.
      → Filters elements to MEP/equipment types only.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND e.file = $file AND e.ifc_type IN $mep_types
    RETURN e.ifc_type AS ifc_type, e.name AS name, e.guid AS guid
    ORDER BY e.ifc_type, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floor=floor, file=ifc_file, mep_types=_MEP_TYPES)
        return [r.data() for r in result]


def count_elements_by_type_on_floor(floor: str, ifc_file: str) -> list[dict]:
    """
    count(e)         → count nodes in the group (like SQL COUNT(*))
    ORDER BY count DESC  → most common types first

    This query is what makes Neo4j shine for your paper —
    it answers "What's on Level 2?" with a structured breakdown,
    something chunked RAG cannot produce deterministically.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND e.file = $file
    RETURN e.ifc_type AS ifc_type, count(e) AS count
    ORDER BY count DESC
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floor=floor, file=ifc_file)
        return [r.data() for r in result]


def get_elements_by_type_across_floors(ifc_type: str, ifc_file: str) -> list[dict]:
    """
    This finds all elements of a specific IFC type across every floor.
    The graph traversal handles the hierarchy naturally — no joins, no subqueries.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.ifc_type = $ifc_type AND e.file = $file
    RETURN s.name AS floor, e.name AS name, e.guid AS guid
    ORDER BY s.name, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, ifc_type=ifc_type, file=ifc_file)
        return [r.data() for r in result]


def get_floor_summary(ifc_file: str) -> list[dict]:
    """
    Return all floors with element counts — used by the /floors endpoint
    as a faster alternative to re-parsing the IFC file each time.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.file = $file
    RETURN s.name AS name, s.elevation AS elevation_m, count(e) AS element_count
    ORDER BY s.elevation
    """
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file)
        return [r.data() for r in result]


def get_loaded_file_stats() -> list[dict]:
    """Return graph diagnostics grouped by IFC file."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    RETURN e.file AS file, count(DISTINCT s) AS storey_count, count(DISTINCT e) AS element_count
    ORDER BY file
    """
    with _get_driver().session() as session:
        result = session.run(cypher)
        return [r.data() for r in result if r.get("file")]


def get_model_summary(ifc_file: str) -> dict:
    """Return floors and top IFC types for a model summary endpoint."""
    floors = get_floor_summary(ifc_file)
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.file = $file
    RETURN e.ifc_type AS ifc_type, count(e) AS count
    ORDER BY count DESC, e.ifc_type
    LIMIT 12
    """
    with _get_driver().session() as session:
        top_types = [r.data() for r in session.run(cypher, file=ifc_file)]
    return {
        "ifc_file": ifc_file,
        "storey_count": len(floors),
        "element_count": sum(int(f.get("element_count", 0)) for f in floors),
        "floors": floors,
        "top_ifc_types": top_types,
    }


def is_file_loaded(ifc_file: str) -> bool:
    """Check if this IFC file has already been ingested into Neo4j."""
    cypher = "MATCH (e:Element {file: $file}) RETURN count(e) AS n LIMIT 1"
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file)
        return result.single()["n"] > 0


def get_all_storey_names(ifc_file: str) -> list[str]:
    """Return storey names accessible from this file's elements (handles federated models)."""
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE e.file = $file
    RETURN DISTINCT s.name AS name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file)
        return [r["name"] for r in result]





def format_results_as_context(results: list[dict], floor: str) -> list[str]:
    """
    Convert Neo4j query results into the same string format used by
    the AST retrieval node, so the generate node receives consistent input.

    Output format matches spatial_ast_retrieval:
      "Entity: IfcPump | Name: Main Pump | GUID: 3Ax..."
    """
    if not results:
        return [f"--- [SOURCE: NEO4J GRAPH DB | FLOOR: {floor}] ---",
                "No elements found on this floor in the graph database."]

    lines = [f"--- [SOURCE: NEO4J GRAPH DB | CONFIRMED FLOOR: {floor}] ---"]
    for r in results:
        prefix = f"Floor: {r.get('floor')} | " if r.get("floor") else ""
        lines.append(
            prefix +
            f"Entity: {r.get('ifc_type', 'Unknown')} | "
            f"Name: {r.get('name', 'Unnamed')} | "
            f"GUID: {r.get('guid', 'N/A')}"
        )
    return lines


def format_count_context(rows: list[dict], label: str) -> list[str]:
    """Format count/aggregation rows as generation context."""
    lines = [f"--- [SOURCE: NEO4J GRAPH DB | STRUCTURAL COUNT: {label}] ---"]
    if not rows:
        lines.append("No matching elements found in the graph database.")
        return lines
    for row in rows:
        if "floor" in row:
            lines.append(
                f"Floor: {row.get('floor')} | Count: {row.get('count', 0)} | "
                f"Storey GUID: {row.get('storey_guid', 'N/A')}"
            )
        else:
            lines.append(f"Entity: {row.get('ifc_type', 'Unknown')} | Count: {row.get('count', 0)}")
    return lines
