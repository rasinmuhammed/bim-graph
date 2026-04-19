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


# ── Read queries ────────────────────────────────────────────────────────────────

def get_all_elements_on_floor(floor: str, ifc_file: str) -> list[dict]:
    """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
      → Find every Storey node that has a CONTAINS relationship to an Element node.

    WHERE s.name = $floor AND s.file = $file
      → Filter: only the storey matching our floor name in our IFC file.
      → $floor and $file are parameters — Neo4j escapes them, preventing injection.

    RETURN ...
      → Project only the fields we need, not the whole node.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND s.file = $file
    RETURN e.ifc_type AS ifc_type, e.name AS name, e.guid AS guid
    ORDER BY e.ifc_type, e.name
    """
    with _get_driver().session() as session:
        result = session.run(cypher, floor=floor, file=ifc_file)
        return [r.data() for r in result]


def get_mep_elements_on_floor(floor: str, ifc_file: str) -> list[dict]:
    """
    WHERE e.ifc_type IN $mep_types
      → Equivalent to SQL's IN clause.
      → Filters elements to MEP/equipment types only.
    """
    cypher = """
    MATCH (s:Storey)-[:CONTAINS]->(e:Element)
    WHERE s.name = $floor AND s.file = $file AND e.ifc_type IN $mep_types
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
    WHERE s.name = $floor AND s.file = $file
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
    WHERE e.ifc_type = $ifc_type AND s.file = $file
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
    MATCH (s:Storey)
    WHERE s.file = $file
    OPTIONAL MATCH (s)-[:CONTAINS]->(e:Element)
    RETURN s.name AS name, s.elevation AS elevation_m, count(e) AS element_count
    ORDER BY s.elevation
    """
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file)
        return [r.data() for r in result]


def is_file_loaded(ifc_file: str) -> bool:
    """Check if this IFC file has already been ingested into Neo4j."""
    cypher = "MATCH (s:Storey {file: $file}) RETURN count(s) AS n LIMIT 1"
    with _get_driver().session() as session:
        result = session.run(cypher, file=ifc_file)
        return result.single()["n"] > 0


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
        lines.append(
            f"Entity: {r.get('ifc_type', 'Unknown')} | "
            f"Name: {r.get('name', 'Unnamed')} | "
            f"GUID: {r.get('guid', 'N/A')}"
        )
    return lines
