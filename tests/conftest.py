import pathlib
import pytest

IFC_DUPLEX = str(pathlib.Path(__file__).parent.parent / "data" / "Duplex_A_20110907.ifc")

@pytest.fixture(scope="session")
def duplex_ifc_path():
    """Share IFC path - loaded once per test session."""
    return IFC_DUPLEX
    