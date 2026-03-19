from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import time
from loguru import logger
from agents.graph import create_agent_graph
from agents.state import AgentState
from memory.conversation import memory

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

@router.delete("/chat/history/{conversation_id}")
async def clear_history(conversation_id: str):
    """清除指定会话的对话历史"""
    memory.clear(conversation_id)
    return {"success": True, "message": f"会话 {conversation_id} 的历史已清除"}


@router.get("/chat/history/{conversation_id}")
async def get_history(conversation_id: str):
    """获取指定会话的对话历史"""
    messages = memory.get_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "total_turns": len(messages) // 2
    }


@router.get("/chat/memory/stats")
async def memory_stats():
    """获取对话记忆统计信息"""
    return memory.stats()

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
        # 生成会话 ID（必须在获取历史之前）
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 获取选项
        options = request.options or {}
        max_iterations = options.get("max_iterations", 3)
        top_k = options.get("top_k", 5)
        
        # 获取对话历史
        chat_history = memory.get_history_text(conversation_id, max_turns=3)
        logger.info(f"Memory lookup: conv={conversation_id}, history_len={len(chat_history)}, history='{chat_history[:100] if chat_history else 'EMPTY'}'")        

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
            "top_k": top_k,
            "chat_history": chat_history,
            "skip_retrieval": False,
            "top_doc_score": 0.0
        }
        
        # 调用 RAG 图
        logger.info(f"Starting RAG pipeline: user={request.user_id}, query='{request.query}'")
        result = rag_graph.invoke(initial_state)
        
        # 检查错误
        if result.get("error"):
            logger.error(f"RAG pipeline error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 保存本轮对话到记忆
        memory.add_turn(
            conversation_id=conversation_id,
            query=request.query,
            answer=result["answer"]
        )

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
