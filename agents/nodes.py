import time
import json
import re
from loguru import logger
from agents.state import AgentState
from retrieval.hybrid import HybridRetriever
from llm.vllm_client import QwenClient
from llm.prompts import GRADER_PROMPT, REWRITER_PROMPT, GENERATOR_PROMPT

# 全局检索器实例
_retriever = None
_llm_client = None

def set_retriever(retriever: HybridRetriever):
    """设置全局检索器"""
    global _retriever
    _retriever = retriever

def set_llm_client(client: QwenClient):
    """设置全局 LLM 客户端"""
    global _llm_client
    _llm_client = client


def retrieve_and_rerank_node(state: AgentState) -> AgentState:
    """
    检索与重排节点
    
    执行步骤：
    1. 混合检索（向量 + BM25）
    2. Cross-Encoder 重排（暂未实现）
    3. 返回 Top-K 文档
    """
    query = state["query"]
    start_time = time.time()
    
    try:
        if _retriever is None:
            logger.error("Retriever not initialized")
            state["error"] = "Retriever not initialized"
            return state
        
        # 执行混合检索
        documents = _retriever.search(query, k=state.get("top_k", 5))
        
        # 更新状态
        state["documents"] = documents
        state["document_scores"] = [doc.get("final_score", 0.0) for doc in documents]
        state["iterations"] += 1
        
        # 记录检索耗时
        elapsed = time.time() - start_time
        state["retrieval_times"].append(elapsed)
        
        logger.info(
            f"Retrieval completed: query='{query}', "
            f"documents={len(documents)}, time={elapsed:.2f}s"
        )
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["error"] = str(e)
    
    return state


def grade_evidence_node(state: AgentState) -> AgentState:
    """
    评估智能体节点
    
    功能：
    - 判别当前文档是否足以回答问题
    - 输出布尔值及缺失信息描述
    """
    try:
        if _llm_client is None:
            logger.warning("LLM client not initialized, forcing sufficient to avoid infinite loop")
            state["is_sufficient"] = True
            state["sufficiency_score"] = 1.0
            state["missing_info"] = ""
            return state
        
        # 构建上下文
        context = "\n\n".join([
            doc.get("content", "") for doc in state["documents"]
        ])
        
        if not context.strip():
            logger.warning("No documents to grade")
            state["is_sufficient"] = False
            state["missing_info"] = "未检索到相关文档"
            state["sufficiency_score"] = 0.0
            return state
        
        # 构建 Prompt
        prompt = GRADER_PROMPT.format(
            query=state["original_query"],
            context=context
        )
        
        # 调用 LLM
        response = _llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=256
        )
        
        # 解析 JSON 响应
        try:
            result = json.loads(response.strip())
            state["is_sufficient"] = result.get("is_sufficient", False)
            state["missing_info"] = result.get("missing_info", "")
            state["sufficiency_score"] = result.get("sufficiency_score", 0.0)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse grader response: {response}")
            state["is_sufficient"] = False
            state["missing_info"] = "无法解析评估结果"
            state["sufficiency_score"] = 0.0
        
        logger.info(
            f"Grading completed: sufficient={state['is_sufficient']}, "
            f"score={state['sufficiency_score']:.2f}"
        )
        
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        state["error"] = str(e)
        state["is_sufficient"] = False
        state["sufficiency_score"] = 0.0
    
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    """
    重写智能体节点
    
    功能：
    - 基于原始问题和缺失信息生成新查询
    - 指向性更强，用于检索缺失信息
    """
    try:
        if _llm_client is None:
            logger.warning("LLM client not initialized, keeping original query")
            # LLM 不可用时保持原始查询不变，避免死循环
            return state
        
        prompt = REWRITER_PROMPT.format(
            original_query=state["original_query"],
            missing_info=state["missing_info"]
        )
        
        response = _llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=128
        )
        
        # 更新查询
        new_query = response.strip()
        state["prev_missing_info"] = state["missing_info"]
        state["query"] = new_query
        
        logger.info(f"Query rewritten: '{state['original_query']}' -> '{new_query}'")
        
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        state["error"] = str(e)
    
    return state


def generate_cot_answer_node(state: AgentState) -> AgentState:
    """
    生成智能体节点
    
    功能：
    - 基于确认的上下文生成回答
    - 使用思维链（CoT）显式输出推理步骤
    - 结合对话历史实现多轮对话
    """
    try:
        if _llm_client is None:
            logger.warning("LLM client not initialized, returning retrieved documents as answer")
            docs = state.get("documents", [])
            if docs:
                content = "\n\n".join([
                    f"[{doc.get('metadata', {}).get('section_path', doc.get('doc_id', ''))}]\n{doc.get('content', '')}"
                    for doc in docs
                ])
                state["answer"] = f"根据文档检索到以下相关内容：\n\n{content}"
            else:
                state["answer"] = "未找到相关文档内容。"
            state["reasoning"] = None
            state["sources"] = [doc.get("metadata", {}).get("section_path", "") for doc in docs]
            state["confidence"] = min(0.5 + len(docs) * 0.1, 0.8) if docs else 0.0
            return state
        
        # 构建上下文
        context = "\n\n".join([
            doc.get("content", "") for doc in state["documents"]
        ])
        
        if not context.strip():
            # 文档为空，但如果有历史对话，仍然尝试用历史回答
            chat_history = state.get("chat_history", "")
            if not chat_history.strip():
                state["answer"] = "未找到相关信息"
                state["reasoning"] = None
                state["sources"] = []
                state["confidence"] = 0.0
                return state
            # 有历史，用空文档+历史继续生成
            context = "（无相关文档，请根据对话历史回答）"
        
        # 构建历史段落
        from llm.prompts import HISTORY_SECTION_TEMPLATE
        chat_history = state.get("chat_history", "")
        history_section = ""
        if chat_history.strip():
            history_section = HISTORY_SECTION_TEMPLATE.format(history=chat_history)
        
        # 构建 Prompt
        prompt = GENERATOR_PROMPT.format(
            context=context,
            query=state["original_query"],
            history_section=history_section
        )
        
        # 调用 LLM
        response = _llm_client.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024
        )
        
        # 解析 CoT 和最终答案
        content = response
        
        # 尝试提取 <thought> 标签
        thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL)
        
        if thought_match:
            state["reasoning"] = thought_match.group(1).strip()
            state["answer"] = content.split('</thought>')[-1].strip()
        else:
            state["answer"] = content.strip()
            state["reasoning"] = None
        
        # 提取引用来源
        state["sources"] = [
            doc.get("metadata", {}).get("section_path", doc.get("doc_name", ""))
            for doc in state["documents"]
        ]
        
        # 计算置信度
        doc_count_factor = min(len(state["documents"]) / 3, 1.0)
        base = 0.6
        sufficiency_bonus = state["sufficiency_score"] * 0.2
        doc_bonus = doc_count_factor * 0.2
        state["confidence"] = min(base + sufficiency_bonus + doc_bonus, 1.0)
        
        logger.info(
            f"Answer generated: confidence={state['confidence']:.2f}, "
            f"sources={len(state['sources'])}"
        )
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        state["error"] = str(e)
        state["answer"] = "生成回答时出错"
        state["confidence"] = 0.0
    
    return state
