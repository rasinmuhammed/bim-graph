from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ROOT / ".env", extra="ignore")

    #LLM
    groq_api_key:     str = Field("", validation_alias="GROQ_API_KEY")
    cerebras_api_key: str = Field(..., validation_alias="CEREBRAS_API_KEY")
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    llm_model:     str = "llama3.1-8b"                       # fast node: constraint extraction
    llm_model_big: str = "qwen-3-235b-a22b-instruct-2507"    # generate + evaluate: strongest available

    #Paths
    chroma_path: str = str(_ROOT / "data" / "chroma_db")
    bm25_path: str = str(_ROOT / "data" / "bm25_index.pkl")
    ifc_data_dir: str = str(_ROOT / "data")
    logs_dir: str = str(_ROOT / "logs")

    #Pipeline tuning
    max_ast_elements: int = 2000        # raised from 1000 — Llama-3.3-70b handles large context
    max_context_chars: int = 60_000     # raised from 30_000 for large-floor inventory queries
    rrf_k: int = 60
    retrieval_top_k: int = 10           # final fused docs returned to generate (was hardcoded 5)
    chroma_group_size: int = 50         # elements per (storey × type) chunk in ChromaDB
    cache_ttl_seconds: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Neo4j
    neo4j_uri:      str = "bolt://localhost:7687"
    neo4j_user:     str = "neo4j"
    neo4j_password: str = "bimgraph123"

settings = Settings()
    