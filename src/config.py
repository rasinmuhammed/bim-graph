from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    #LLM 
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    llm_model_big: str = "llama-3.3-70b-versatile"

    #Paths
    chroma_path: str = str(_ROOT / "data" / "chroma_db")
    bm25_path: str = str(_ROOT / "data" / "bm25_index.pkl")
    ifc_data_dir: str = str(_ROOT / "data")
    logs_dir: str = str(_ROOT / "logs")

    #Pipeline tuning
    max_ast_elements: int = 1000
    max_context_chars: int = 30_000
    rrf_k: int = 60
    cache_ttl_seconds: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    
settings = Settings()
    