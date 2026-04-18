import pytest
import ifcopenshell


def test_duplex_has_storeys(duplex_ifc_path):
    model   = ifcopenshell.open(duplex_ifc_path)
    storeys = model.by_type("IfcBuildingStorey")
    assert len(storeys) > 0, "Duplex IFC must contain at least one storey"


def test_duplex_elements_on_storeys(duplex_ifc_path):
    model   = ifcopenshell.open(duplex_ifc_path)
    rels    = model.by_type("IfcRelContainedInSpatialStructure")
    assert len(rels) > 0, "Must have spatial containment relations"

    total_elements = sum(len(r.RelatedElements) for r in rels)
    assert total_elements > 10, "Must have meaningful element count"