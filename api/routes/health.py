from fastapi import APIRouter
from pydantic import BaseModel
import requests
from core.config import settings

router = APIRouter(tags=["health"])

class HealthResponse(BaseModel):
    status: str
    components: dict
    llm_model: str
    llm_base_url: str

def check_ollama() -> dict:
    """检查 Ollama 服务状态"""
    try:
        # Ollama 原生 API
        ollama_url = settings.LLM_BASE_URL.replace("/v1", "")
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            model_loaded = settings.LLM_MODEL in models
            return {
                "status": "healthy" if model_loaded else "model_not_found",
                "available_models": models,
                "target_model": settings.LLM_MODEL,
                "model_loaded": model_loaded
            }
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    ollama_status = check_ollama()
    llm_ok = ollama_status.get("status") == "healthy"

    return HealthResponse(
        status="healthy" if llm_ok else "degraded",
        components={
            "llm_service": ollama_status,
            "vector_store": "healthy",
            "database": "healthy"
        },
        llm_model=settings.LLM_MODEL,
        llm_base_url=settings.LLM_BASE_URL
    )
