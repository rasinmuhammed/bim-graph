import pathlib
import ifcopenshell

_PROJECT_ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
_DEFAULT_IFC     = str(_PROJECT_ROOT / "data" / "Duplex_A_20110907.ifc")

def chunk_ifc_to_text( ifc_path: str = _DEFAULT_IFC, chunk_size: int = 500 ) -> list[str]:
    model = ifcopenshell.open(ifc_path)
    lines = []

    for product in model.by_type("IfcProduct"):
        line = f"Entity: {product.is_a()} | Name: {product.Name} | GUID: {product.GlobalId}"
        lines.append(line)

    giant_string = "\n".join(lines)
    
    # Split into chunks
    chunks = [giant_string[i: i + chunk_size] for i in range(0, len(giant_string), chunk_size)]
    return chunks

if __name__ == "__main__":
    chunks = chunk_ifc_to_text()
    print(f"Total chunks: {len(chunks)}")
    print("\n--- CHUNK 0 ---\n", chunks[0])
    print("\n--- CHUNK 1 ---\n", chunks[1])
    