from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
import json
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
    retrieved_docs: List[Dict[str, Any]] = []
    retrieval_process: dict
    confidence: float
    response_time_ms: int
    is_doc_grounded: bool = True
    doc_notice: Optional[str] = None
    confidence_mode: str = "doc_grounded"

# 全局 RAG 图实例
rag_graph = None


def _format_retrieved_docs(documents: List[dict]) -> List[Dict[str, Any]]:
    """抽取可用于前端展示的检索/重排打分信息"""
    formatted: List[Dict[str, Any]] = []
    for doc in documents:
        meta = doc.get("metadata", {})
        formatted.append({
            "doc_id": doc.get("doc_id", ""),
            "section_path": meta.get("section_path", ""),
            "final_score": float(doc.get("final_score", 0.0)),
            "bm25_norm": float(doc.get("bm25_norm", 0.0)),
            "vector_norm": float(doc.get("vector_norm", 0.0)),
            "rerank_score": float(doc.get("rerank_score", 0.0)) if doc.get("rerank_score") is not None else None,
            "neighbor_of": doc.get("neighbor_of"),
            "content_preview": (doc.get("content", "")[:160] + "...") if len(doc.get("content", "")) > 160 else doc.get("content", "")
        })
    return formatted

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式 RAG 对话接口（SSE）

    流程：
    1. RAG 图执行检索、评估（与普通接口相同）
    2. 最后生成阶段改为流式输出 token
    3. 通过 Server-Sent Events 逐 token 推送到前端
    """
    if rag_graph is None:
        raise HTTPException(status_code=500, detail="RAG graph not initialized")

    async def event_generator():
        start_time = time.time()
        try:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            options = request.options or {}
            max_iterations = options.get("max_iterations", 3)
            top_k = options.get("top_k", 5)

            chat_history = memory.get_history_text(conversation_id, max_turns=3)

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
                "top_doc_score": 0.0,
                "retrieval_confidence": 0.0,
                "generation_confidence": 0.0,
                "is_doc_grounded": True,
                "doc_notice": None,
                "confidence_mode": "doc_grounded"
            }

            # 先发送「检索中」状态
            yield f"data: {json.dumps({'type': 'status', 'content': 'retrieving'}, ensure_ascii=False)}\n\n"

            # 运行 RAG 图（检索+评估，不含最终生成）
            # 通过 stream_mode 获取中间状态
            from agents.nodes import _llm_client
            from llm.prompts import GENERATOR_PROMPT, HISTORY_SECTION_TEMPLATE
            import re

            # 执行完整图但跳过最终 LLM 生成（在此处流式生成）
            # 方案：先跑到 generate 节点之前，拿到 context，再流式生成
            result = rag_graph.invoke(initial_state)

            if result.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'content': result['error']}, ensure_ascii=False)}\n\n"
                return

            # 发送检索完成状态
            yield f"data: {json.dumps({'type': 'status', 'content': 'generating'}, ensure_ascii=False)}\n\n"

            # 流式生成答案
            if _llm_client is None:
                # LLM 不可用，直接发送已有答案
                answer = result.get("answer", "未找到相关信息")
                yield f"data: {json.dumps({'type': 'token', 'content': answer}, ensure_ascii=False)}\n\n"
            else:
                # 重新构建 prompt 并流式输出
                context = "\n\n".join([doc.get("content", "") for doc in result["documents"]])
                if not context.strip():
                    context = "（无相关文档，请根据对话历史回答）"

                history_section = ""
                if chat_history.strip():
                    history_section = HISTORY_SECTION_TEMPLATE.format(history=chat_history)

                prompt = GENERATOR_PROMPT.format(
                    context=context,
                    query=request.query,
                    history_section=history_section
                )

                full_answer = ""
                in_thought = False
                thought_buf = ""

                for token in _llm_client.stream_generate(prompt, temperature=0.7, max_tokens=1024):
                    full_answer += token

                    # 过滤 <thought>...</thought> 标签，不推送给前端
                    if "<thought>" in token:
                        in_thought = True
                    if in_thought:
                        thought_buf += token
                        if "</thought>" in thought_buf:
                            in_thought = False
                            # 发送 thought 关闭后的剩余内容
                            after = thought_buf.split("</thought>", 1)[1]
                            thought_buf = ""
                            if after.strip():
                                yield f"data: {json.dumps({'type': 'token', 'content': after}, ensure_ascii=False)}\n\n"
                        continue

                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                # 提取 reasoning
                reasoning = None
                thought_match = re.search(r'<thought>(.*?)</thought>', full_answer, re.DOTALL)
                if thought_match:
                    reasoning = thought_match.group(1).strip()
                    clean_answer = full_answer.split("</thought>")[-1].strip()
                else:
                    clean_answer = full_answer.strip()

                # 保存记忆
                memory.add_turn(
                    conversation_id=conversation_id,
                    query=request.query,
                    answer=clean_answer
                )

                # 发送结束元数据
                response_time_ms = int((time.time() - start_time) * 1000)
                meta = {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "reasoning": reasoning,
                    "sources": result.get("sources", []),
                    "retrieved_docs": _format_retrieved_docs(result.get("documents", [])),
                    "confidence": result.get("confidence", 0.0),
                    "is_doc_grounded": result.get("is_doc_grounded", True),
                    "doc_notice": result.get("doc_notice"),
                    "confidence_mode": result.get("confidence_mode", "doc_grounded"),
                    "retrieval_process": {
                        "iterations": result["iterations"],
                        "sufficiency_score": result["sufficiency_score"],
                        "total_documents_retrieved": len(result["documents"]),
                        "retrieval_times": result["retrieval_times"]
                    },
                    "response_time_ms": response_time_ms
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("Stream chat failed: {}", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
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
            retrieved_docs=_format_retrieved_docs(result.get("documents", [])),
            retrieval_process={
                "iterations": result["iterations"],
                "sufficiency_score": result["sufficiency_score"],
                "total_documents_retrieved": len(result["documents"]),
                "retrieval_times": result["retrieval_times"]
            },
            confidence=result["confidence"],
            response_time_ms=response_time_ms,
            is_doc_grounded=result.get("is_doc_grounded", True),
            doc_notice=result.get("doc_notice"),
            confidence_mode=result.get("confidence_mode", "doc_grounded")
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
