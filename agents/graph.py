from typing import Literal
from langgraph.graph import StateGraph, END
from loguru import logger
from agents.state import AgentState
from agents.nodes import (
    retrieve_and_rerank_node,
    grade_evidence_node,
    rewrite_query_node,
    generate_cot_answer_node
)

# 记忆命中关键词：这类问题直接从历史回答，无需检索文档
_MEMORY_PATTERNS = [
    "我叫", "我的名字", "你知道我", "我是谁", "记得我",
    "我说过", "刚才", "之前", "上面", "前面说",
    "我告诉你", "我提到", "你还记得", "我们聊", "我们说",
    "上一个", "上一次", "前一个", "前一次", "继续",
    "接着", "然后呢", "还有呢", "那么"
]


def history_check_node(state: AgentState) -> AgentState:
    """
    记忆检查节点：判断问题是否可以直接由历史对话回答
    命中条件（满足任意一个）：
    1. 问题包含记忆关键词 且 有历史
    2. 问题极短（<=8字）且有历史（通常是追问）
    """
    query = state.get("original_query", "")
    has_history = bool(state.get("chat_history", "").strip())

    keyword_hit = has_history and any(p in query for p in _MEMORY_PATTERNS)
    short_query_hit = has_history and len(query.strip()) <= 8

    is_memory_query = keyword_hit or short_query_hit

    state["skip_retrieval"] = is_memory_query

    if is_memory_query:
        state["documents"] = []
        state["document_scores"] = []
        state["sufficiency_score"] = 1.0
        state["is_sufficient"] = True
        logger.info(f"Memory hit: query='{query}' (keyword={keyword_hit}, short={short_query_hit})")
    else:
        logger.info(f"No memory hit, proceeding to retrieval: query='{query}'")

    return state


def route_history_check(state: AgentState) -> Literal["use_history", "retrieve"]:
    """路由：记忆命中则直接生成，否则走检索流程"""
    if state.get("skip_retrieval", False):
        return "use_history"
    return "retrieve"


def create_agent_graph():
    """
    创建四智能体协作状态图
    
    流程：
    retrieve_rerank -> grade -> [sufficient] -> generate -> END
                               [insufficient] -> rewrite -> retrieve_rerank
    
    优化：history_check 节点前置，记忆命中时直接生成，跳过检索评估循环
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("history_check", history_check_node)
    workflow.add_node("retrieve_rerank", retrieve_and_rerank_node)
    workflow.add_node("grade", grade_evidence_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_cot_answer_node)

    # 入口：先做记忆检查
    workflow.set_entry_point("history_check")

    # history_check 路由：命中记忆直接生成，否则进入检索
    workflow.add_conditional_edges(
        "history_check",
        route_history_check,
        {
            "use_history": "generate",
            "retrieve": "retrieve_rerank"
        }
    )

    # 固定边
    workflow.add_edge("retrieve_rerank", "grade")
    workflow.add_edge("rewrite", "retrieve_rerank")
    workflow.add_edge("generate", END)

    # 条件路由
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
    基于证据充分性的路由逻辑
    """
    max_iter = state.get("max_iterations", 3)

    # 规则 1：充分性分数高
    if state["sufficiency_score"] > 0.85:
        logger.info("Evidence is sufficient (high score)")
        return "sufficient"

    # 规则 2：达到最大迭代次数
    if state["iterations"] >= max_iter:
        logger.info(f"Max iterations reached ({max_iter}), forcing generation")
        return "sufficient"

    # 规则 3：检查是否陷入循环
    prev = state.get("prev_missing_info", "")
    curr = state.get("missing_info", "")
    if prev and curr and prev == curr:
        logger.info("Loop detected (same missing info), forcing generation")
        return "sufficient"

    # 规则 4：继续迭代
    logger.info(f"Evidence insufficient (score={state['sufficiency_score']:.2f}), rewriting")
    return "insufficient"
