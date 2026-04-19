"""
graph_db/loader.py
──────────────────
Ingests an IFC file into Neo4j by traversing the IfcOpenShell AST
and writing nodes + relationships using Cypher MERGE statements.

MERGE vs CREATE:
  CREATE always inserts — running twice causes duplicate nodes.
  MERGE checks if the node exists first (by the property in the pattern).
  If it exists → update it. If not → create it.
  This makes the loader idempotent: safe to re-run on the same file.

Run this script directly to ingest all IFC files in data/:
  python -m graph_db.loader
"""

import pathlib
import logging
import ifcopenshell
from neo4j import Driver
from graph_db.queries import _get_driver, ensure_schema, is_file_loaded

logger = logging.getLogger("bim_graph.loader")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def load_ifc_to_graph(ifc_path: str | pathlib.Path, driver: Driver | None = None) -> dict:
    """
    Parse an IFC file and write its spatial hierarchy into Neo4j.

    Returns a stats dict so you can verify the load:
      {"storeys": 3, "elements": 412, "file": "Duplex_A_20110907.ifc"}

    The graph structure written:
      (:Project)-[:HAS_SITE]->(:Site)-[:HAS_BUILDING]->(:Building)
      (:Building)-[:HAS_STOREY]->(:Storey)
      (:Storey)-[:CONTAINS]->(:Element)
    """
    ifc_path = pathlib.Path(ifc_path)
    filename = ifc_path.name
    drv      = driver or _get_driver()

    logger.info("Loading IFC file into Neo4j: %s", filename)
    model = ifcopenshell.open(str(ifc_path))

    stats = {"storeys": 0, "elements": 0, "file": filename}

    with drv.session() as session:
        # ── 1. Project node ────────────────────────────────────────────────────
        # IfcProject is the root of every IFC file.
        # MERGE ON GUID so re-running doesn't create duplicates.
        projects = model.by_type("IfcProject")
        project  = projects[0] if projects else None
        if project:
            session.run(
                """
                MERGE (p:Project {guid: $guid})
                SET p.name = $name, p.file = $file
                """,
                guid=project.GlobalId,
                name=project.Name or "Unnamed Project",
                file=filename,
            )

        # ── 2. Site node ───────────────────────────────────────────────────────
        sites = model.by_type("IfcSite")
        site  = sites[0] if sites else None
        if site:
            session.run(
                """
                MERGE (si:Site {guid: $guid})
                SET si.name = $name, si.file = $file
                """,
                guid=site.GlobalId,
                name=site.Name or "Unnamed Site",
                file=filename,
            )
            if project and site:
                session.run(
                    """
                    MATCH (p:Project {guid: $pguid})
                    MATCH (si:Site   {guid: $sguid})
                    MERGE (p)-[:HAS_SITE]->(si)
                    """,
                    pguid=project.GlobalId,
                    sguid=site.GlobalId,
                )

        # ── 3. Building node ───────────────────────────────────────────────────
        buildings = model.by_type("IfcBuilding")
        building  = buildings[0] if buildings else None
        if building:
            session.run(
                """
                MERGE (b:Building {guid: $guid})
                SET b.name = $name, b.file = $file
                """,
                guid=building.GlobalId,
                name=building.Name or "Unnamed Building",
                file=filename,
            )
            if site and building:
                session.run(
                    """
                    MATCH (si:Site     {guid: $sguid})
                    MATCH (b:Building  {guid: $bguid})
                    MERGE (si)-[:HAS_BUILDING]->(b)
                    """,
                    sguid=site.GlobalId,
                    bguid=building.GlobalId,
                )

        # ── 4. Storeys ─────────────────────────────────────────────────────────
        # IfcBuildingStorey = a floor/level in the building.
        # Elevation is the height in metres from the building datum.
        for storey in model.by_type("IfcBuildingStorey"):
            session.run(
                """
                MERGE (s:Storey {guid: $guid})
                SET s.name      = $name,
                    s.elevation = $elevation,
                    s.file      = $file
                """,
                guid=storey.GlobalId,
                name=storey.Name or "Unnamed Storey",
                elevation=float(storey.Elevation) if storey.Elevation else 0.0,
                file=filename,
            )
            if building:
                session.run(
                    """
                    MATCH (b:Building {guid: $bguid})
                    MATCH (s:Storey   {guid: $sguid})
                    MERGE (b)-[:HAS_STOREY]->(s)
                    """,
                    bguid=building.GlobalId,
                    sguid=storey.GlobalId,
                )
            stats["storeys"] += 1

        # ── 5. Elements ────────────────────────────────────────────────────────
        # IfcRelContainedInSpatialStructure links elements to storeys.
        # We batch elements per storey for efficiency.
        for rel in model.by_type("IfcRelContainedInSpatialStructure"):
            storey = rel.RelatingStructure
            if not storey.is_a("IfcBuildingStorey"):
                continue

            # Batch write: build a list of element dicts, write in one query.
            # unwind $elements means Neo4j iterates the list server-side,
            # which is far faster than one session.run() call per element.
            element_batch = []
            for element in rel.RelatedElements:
                element_batch.append({
                    "guid":     element.GlobalId,
                    "name":     element.Name or "Unnamed",
                    "ifc_type": element.is_a(),
                    "file":     filename,
                })

            if element_batch:
                session.run(
                    """
                    UNWIND $elements AS el
                    MERGE (e:Element {guid: el.guid})
                    SET e.name     = el.name,
                        e.ifc_type = el.ifc_type,
                        e.file     = el.file
                    WITH e, el
                    MATCH (s:Storey {guid: $storey_guid})
                    MERGE (s)-[:CONTAINS]->(e)
                    """,
                    elements=element_batch,
                    storey_guid=storey.GlobalId,
                )
                stats["elements"] += len(element_batch)

    logger.info(
        "✓ Loaded %s → %d storeys, %d elements",
        filename, stats["storeys"], stats["elements"],
    )
    return stats


if __name__ == "__main__":
    """
    Run this to load all IFC files in data/ into Neo4j.
    Requires Neo4j to be running: docker compose up neo4j -d
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from graph_db.queries import ensure_schema

    logger.info("Setting up Neo4j schema...")
    ensure_schema()

    data_dir = _PROJECT_ROOT / "data"
    ifc_files = list(data_dir.glob("*.ifc"))

    if not ifc_files:
        logger.error("No IFC files found in %s", data_dir)
    else:
        for ifc_file in ifc_files:
            if is_file_loaded(ifc_file.name):
                logger.info("Skipping %s — already in graph.", ifc_file.name)
                continue
            load_ifc_to_graph(ifc_file)

    logger.info("All IFC files ingested into Neo4j.")
