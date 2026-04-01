from typing import Literal
from langgraph.graph import StateGraph, END
from loguru import logger
from agents.state import AgentState
from agents.nodes import (
    retrieve_and_rerank_node,
    grade_evidence_node,
    rewrite_query_node,
    generate_cot_answer_node,
    generate_fallback_answer_node
)

# 记忆命中关键词
_MEMORY_PATTERNS = [
    "我叫", "我的名字", "你知道我", "我是谁", "记得我",
    "我说过", "刚才", "之前", "上面", "前面说",
    "我告诉你", "我提到", "你还记得", "我们聊", "我们说",
    "上一个", "上一次", "前一个", "前一次", "继续",
    "接着", "然后呢", "还有呢", "那么"
]

# 检索分数阈值：低于此值认为文档不相关，直接走 fallback 回答
SCORE_THRESHOLD = 0.35

# 高分命中直接生成阈值：避免 grader/rewrite 把已命中的查询改坏
HIGH_SCORE_DIRECT_GENERATE = 0.75


def history_check_node(state: AgentState) -> AgentState:
    """
    第一层：记忆前置检查
    对明显的记忆性问题（关键词命中）直接跳过检索
    """
    query = state.get("original_query", "")
    has_history = bool(state.get("chat_history", "").strip())

    keyword_hit = has_history and any(p in query for p in _MEMORY_PATTERNS)
    # 关闭“短问题直接记忆命中”，避免跳过文档检索
    short_query_hit = False

    is_memory_query = keyword_hit or short_query_hit
    state["skip_retrieval"] = is_memory_query
    state["top_doc_score"] = 0.0

    if is_memory_query:
        state["documents"] = []
        state["document_scores"] = []
        state["sufficiency_score"] = 1.0
        state["is_sufficient"] = True
        logger.info(f"[Layer1] Memory hit: '{query}' (keyword={keyword_hit}, short={short_query_hit})")
    else:
        logger.info(f"[Layer1] No memory hit, proceed to retrieval: '{query}'")

    return state


def route_history_check(state: AgentState) -> Literal["use_history", "retrieve"]:
    """第一层路由：记忆命中 → 直接生成；否则 → 检索"""
    return "use_history" if state.get("skip_retrieval", False) else "retrieve"


def route_after_retrieval(state: AgentState) -> Literal["grade", "generate", "fallback"]:
    """
    第二层路由（检索后评分路由）

    逻辑：
    - 检索到文档且最高分 >= HIGH_SCORE_DIRECT_GENERATE → 直接生成（避免 rewrite 改坏）
    - 检索到文档且最高分 >= SCORE_THRESHOLD → 走 Grader 精细评估
    - 否则 → 直接 fallback 回答，不再进入多轮检索评分
    """
    top_score = max(state.get("top_doc_score", 0.0), state.get("best_doc_score", 0.0))
    has_docs = bool(state.get("documents"))

    if has_docs and top_score >= HIGH_SCORE_DIRECT_GENERATE:
        logger.info(
            f"[Layer2] Score {top_score:.3f} >= {HIGH_SCORE_DIRECT_GENERATE}, direct generate"
        )
        return "generate"

    if has_docs and top_score >= SCORE_THRESHOLD:
        logger.info(f"[Layer2] Score {top_score:.3f} >= {SCORE_THRESHOLD}, proceed to grade")
        return "grade"

    logger.info(f"[Layer2] Score {top_score:.3f} < {SCORE_THRESHOLD}, direct fallback")
    state["documents"] = []
    state["document_scores"] = []
    state["sufficiency_score"] = 0.0
    state["is_sufficient"] = False
    return "fallback"


def create_agent_graph():
    """
    双层路由 RAG 状态图

    第一层（history_check）：关键词记忆命中 → 直接生成
    第二层（after_retrieval）：
      - 高分命中直接生成
      - 中等分数进入 Grader
      - 低分 fallback
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("history_check", history_check_node)
    workflow.add_node("retrieve_rerank", retrieve_and_rerank_node)
    workflow.add_node("grade", grade_evidence_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_cot_answer_node)
    workflow.add_node("fallback_generate", generate_fallback_answer_node)

    workflow.set_entry_point("history_check")

    # 第一层路由
    workflow.add_conditional_edges(
        "history_check",
        route_history_check,
        {
            "use_history": "generate",
            "retrieve": "retrieve_rerank"
        }
    )

    # 第二层路由（检索后评分）
    workflow.add_conditional_edges(
        "retrieve_rerank",
        route_after_retrieval,
        {
            "grade": "grade",
            "generate": "generate",
            "fallback": "fallback_generate"
        }
    )

    # 固定边
    workflow.add_edge("rewrite", "retrieve_rerank")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback_generate", END)

    # Grader 路由
    workflow.add_conditional_edges(
        "grade",
        route_based_on_sufficiency,
        {
            "sufficient": "generate",
            "insufficient": "rewrite"
        }
    )

    return workflow.compile()


def route_based_on_sufficiency(state: AgentState) -> Literal["sufficient", "insufficient"]:
    """
    Grader 后路由：基于证据充分性决定生成还是重写
    """
    max_iter = state.get("max_iterations", 3)

    if state["sufficiency_score"] > 0.85:
        logger.info("[Grader] Sufficient (high score)")
        return "sufficient"

    if state["iterations"] >= max_iter:
        logger.info(f"[Grader] Max iterations ({max_iter}) reached, force generate")
        return "sufficient"

    prev = state.get("prev_missing_info", "")
    curr = state.get("missing_info", "")
    if prev and curr and prev == curr:
        logger.info("[Grader] Loop detected, force generate")
        return "sufficient"

    logger.info(f"[Grader] Insufficient (score={state['sufficiency_score']:.2f}), rewrite")
    return "insufficient"
