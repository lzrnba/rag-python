from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """系统配置"""

    _project_root: Path = Path(__file__).resolve().parent.parent
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # LLM 配置
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "EMPTY"
    
    # Embedding 配置
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    
    # Reranker 配置
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_ENABLED: bool = True
    RERANK_CANDIDATES: int = 20
    
    # 向量库 / 文档目录配置
    VECTOR_STORE_PATH: str = "data/vectorstore"
    DOCUMENTS_DIR: str = "data/documents"

    @property
    def VECTOR_STORE_PATH_RESOLVED(self) -> str:
        p = Path(self.VECTOR_STORE_PATH)
        return str(p if p.is_absolute() else (self._project_root / p).resolve())

    @property
    def DOCUMENTS_DIR_RESOLVED(self) -> str:
        p = Path(self.DOCUMENTS_DIR)
        return str(p if p.is_absolute() else (self._project_root / p).resolve())
    
    # RAG 配置
    MAX_ITERATIONS: int = 3
    TOP_K: int = 5
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag.log"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
