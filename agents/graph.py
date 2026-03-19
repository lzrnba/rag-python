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

def create_agent_graph():
    """
    创建四智能体协作状态图
    
    流程：
    retrieve_rerank -> grade -> [sufficient] -> generate -> END
                               [insufficient] -> rewrite -> retrieve_rerank
    """
    # LangGraph 0.0.20: 传入 TypedDict class 作为 schema
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve_rerank", retrieve_and_rerank_node)
    workflow.add_node("grade", grade_evidence_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_cot_answer_node)

    # 设置入口节点
    workflow.set_entry_point("retrieve_rerank")

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
