import pathlib
import ifcopenshell
import pydantic

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_IFC_PATH     = str(_PROJECT_ROOT / "data" / "Duplex_A_20110907.ifc")

class StoreyNode(pydantic.BaseModel):
    name: str
    guid: str
    elevation: float | None

model = ifcopenshell.open(_IFC_PATH)

storeys_list = []

for storey in model.by_type("IfcBuildingStorey"):
    node = StoreyNode(
        name = storey.Name,
        guid = storey.GlobalId,
        elevation = storey.Elevation
    )
    storeys_list.append(node)

for node in storeys_list:
    print(node.model_dump())