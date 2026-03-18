from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """系统配置"""
    
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
    
    # 向量库配置
    VECTOR_STORE_PATH: str = "data/vectorstore"
    
    # RAG 配置
    MAX_ITERATIONS: int = 3
    TOP_K: int = 5
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
