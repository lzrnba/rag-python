from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import time
from loguru import logger
from agents.graph import create_agent_graph
from agents.state import AgentState

router = APIRouter(prefix="/v1", tags=["chat"])

# 数据模型
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    user_id: str = Field(..., description="用户 ID")
    conversation_id: Optional[str] = Field(None, description="会话 ID")
    options: Optional[dict] = Field(default_factory=dict, description="选项")

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    reasoning: Optional[str] = None
    sources: List[str] = []
    retrieval_process: dict
    confidence: float
    response_time_ms: int

# 全局 RAG 图实例
rag_graph = None

def initialize_graph():
    """初始化 RAG 图"""
    global rag_graph
    rag_graph = create_agent_graph()
    logger.info("RAG graph initialized")

@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    RAG 对话接口
    
    流程：
    1. 构建初始状态
    2. 调用 RAG 状态图
    3. 返回响应
    """
    if rag_graph is None:
        raise HTTPException(status_code=500, detail="RAG graph not initialized")
    
    start_time = time.time()
    
    try:
        # 生成会话 ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 获取选项
        options = request.options or {}
        max_iterations = options.get("max_iterations", 3)
        top_k = options.get("top_k", 5)
        
        # 构建初始状态
        initial_state: AgentState = {
            "query": request.query,
            "original_query": request.query,
            "user_id": request.user_id,
            "conversation_id": conversation_id,
            "documents": [],
            "document_scores": [],
            "is_sufficient": False,
            "missing_info": "",
            "sufficiency_score": 0.0,
            "iterations": 0,
            "max_iterations": max_iterations,
            "prev_missing_info": "",
            "answer": "",
            "reasoning": None,
            "sources": [],
            "confidence": 0.0,
            "start_time": start_time,
            "retrieval_times": [],
            "error": None,
            "top_k": top_k
        }
        
        # 调用 RAG 图
        logger.info(f"Starting RAG pipeline: user={request.user_id}, query='{request.query}'")
        result = rag_graph.invoke(initial_state)
        
        # 检查错误
        if result.get("error"):
            logger.error(f"RAG pipeline error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 计算响应时间
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # 构建响应
        response = ChatResponse(
            conversation_id=result["conversation_id"],
            answer=result["answer"],
            reasoning=result.get("reasoning"),
            sources=result.get("sources", []),
            retrieval_process={
                "iterations": result["iterations"],
                "sufficiency_score": result["sufficiency_score"],
                "total_documents_retrieved": len(result["documents"]),
                "retrieval_times": result["retrieval_times"]
            },
            confidence=result["confidence"],
            response_time_ms=response_time_ms
        )
        
        logger.info(
            f"RAG pipeline completed: iterations={result['iterations']}, "
            f"confidence={result['confidence']:.2f}, time={response_time_ms}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat completion failed: {}", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
