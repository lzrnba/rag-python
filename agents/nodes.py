import time
import json
import re
import math
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
            state["top_doc_score"] = 0.0
            return state

        # 执行混合检索
        documents = _retriever.search(query, k=state.get("top_k", 5))

        # 更新状态
        state["documents"] = documents

        # 统一分数尺度：若有 Cross-Encoder 原始分，转为 0~1 后用于路由与置信度
        route_scores = []
        for doc in documents:
            score = None

            rerank_raw = doc.get("rerank_score")
            if rerank_raw is not None:
                try:
                    raw = float(rerank_raw)
                    score = 1.0 / (1.0 + math.exp(-raw))
                    doc["rerank_score_norm"] = score
                except Exception:
                    score = None

            if score is None:
                try:
                    final_raw = doc.get("final_score", 0.0)
                    score = float(final_raw) if final_raw is not None else 0.0
                except Exception:
                    score = 0.0

            if math.isnan(score) or math.isinf(score):
                score = 0.0

            route_scores.append(max(0.0, min(score, 1.0)))

        state["document_scores"] = route_scores
        state["iterations"] += 1

        # 记录最高文档分数（用于后续评分路由）
        if documents:
            current_top = max(route_scores) if route_scores else 0.0
            state["top_doc_score"] = current_top

            prev_best = state.get("best_doc_score", 0.0)
            prev_best = float(prev_best) if prev_best is not None else 0.0
            state["best_doc_score"] = max(prev_best, current_top)

            sorted_scores = sorted(route_scores, reverse=True)
            score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            avg_topk = sum(sorted_scores[: min(3, len(sorted_scores))]) / min(3, len(sorted_scores))

            # Retrieval Confidence: 使用历史最佳分 + 当前分布，避免多轮重写后分数塌陷
            best_score = float(state.get("best_doc_score", current_top) or 0.0)
            retrieval_conf = 0.5 * best_score + 0.3 * avg_topk + 0.2 * max(score_gap, 0.0)
            state["retrieval_confidence"] = max(0.0, min(retrieval_conf, 1.0))
        else:
            state["top_doc_score"] = 0.0
            prev_best = state.get("best_doc_score", 0.0)
            prev_best = float(prev_best) if prev_best is not None else 0.0
            state["best_doc_score"] = prev_best
            state["retrieval_confidence"] = 0.0

        # 记录检索耗时
        elapsed = time.time() - start_time
        state["retrieval_times"].append(elapsed)

        logger.info(
            f"Retrieval completed: query='{query}', "
            f"documents={len(documents)}, top_score={state['top_doc_score']:.3f}, time={elapsed:.2f}s"
        )

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["error"] = str(e)
        state["top_doc_score"] = 0.0

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


def generate_fallback_answer_node(state: AgentState) -> AgentState:
    """
    非文档命中场景：直接返回简洁的"知识库未收录"提示，不调用 LLM。
    """
    query = state.get("original_query", "")
    state["answer"] = f"知识库中未收录与\"{query}\"相关的内容，请尝试换个问法或上传相关文档。"
    state["reasoning"] = None
    state["sources"] = []
    state["is_doc_grounded"] = False
    state["doc_notice"] = "知识库中未收录相关内容"
    state["confidence_mode"] = "fallback"
    state["retrieval_confidence"] = 0.0
    state["generation_confidence"] = 0.0
    state["confidence"] = 0.0
    logger.info(f"[Fallback] No relevant docs for query: '{query}'")
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

        # 文档命中模式标记
        state["is_doc_grounded"] = True
        state["doc_notice"] = None
        state["confidence_mode"] = "doc_grounded"

        # Generation Confidence: 用 sufficiency 近似生成可信度，并考虑回答稳定性代理
        suff = max(0.0, min(state.get("sufficiency_score", 0.0), 1.0))
        answer_len_factor = 0.7 if len(state["answer"]) > 40 else 0.5
        state["generation_confidence"] = max(0.0, min(0.7 * suff + 0.3 * answer_len_factor, 1.0))

        # Final Confidence: 检索置信度 + 生成置信度加权
        retrieval_conf = state.get("retrieval_confidence", 0.0)
        state["confidence"] = max(0.0, min(0.55 * retrieval_conf + 0.45 * state["generation_confidence"], 0.98))
        if retrieval_conf > 0.6 and state["confidence"] < 0.55:
            state["confidence"] = 0.55

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
