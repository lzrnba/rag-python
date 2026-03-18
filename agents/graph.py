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
    retrieve_rerank → grade → [sufficient] → generate → END
                              ↓
                         [insufficient] → rewrite → retrieve_rerank (循环)
    """
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("retrieve_rerank", retrieve_and_rerank_node)
    workflow.add_node("grade", grade_evidence_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_cot_answer_node)
    
    # 设置入口
    workflow.set_entry_point("retrieve_rerank")
    
    # 添加边
    workflow.add_edge("retrieve_rerank", "grade")
    
    # 条件路由：核心创新点
    workflow.add_conditional_edges(
        "grade",
        route_based_on_sufficiency,
        {
            "sufficient": "generate",
            "insufficient": "rewrite"
        }
    )
    
    workflow.add_edge("rewrite", "retrieve_rerank")  # 形成闭环
    workflow.add_edge("generate", END)
    
    # 编译应用
    return workflow.compile()


def route_based_on_sufficiency(state: AgentState) -> Literal["sufficient", "insufficient"]:
    """
    基于证据充分性的路由逻辑
    
    决策规则：
    1. 如果充分性分数 > 0.85，认为充分
    2. 如果达到最大迭代次数，强制生成
    3. 如果缺失信息重复（陷入循环），强制生成
    4. 否则继续迭代
    """
    MAX_ITER = state.get("max_iterations", 3)
    
    # 规则 1：充分性分数高
    if state["sufficiency_score"] > 0.85:
        logger.info("Evidence is sufficient (high score)")
        return "sufficient"
    
    # 规则 2：达到最大迭代次数
    if state["iterations"] >= MAX_ITER:
        logger.info(f"Max iterations reached ({MAX_ITER}), forcing generation")
        return "sufficient"
    
    # 规则 3：检查是否陷入循环
    if state["prev_missing_info"] and state["missing_info"]:
        # 简单的相似度检查：如果缺失信息相同，说明陷入循环
        if state["prev_missing_info"] == state["missing_info"]:
            logger.info("Detected loop (same missing info), forcing generation")
            return "sufficient"
    
    # 规则 4：继续迭代
    logger.info(f"Evidence insufficient (score={state['sufficiency_score']:.2f}), rewriting query")
    return "insufficient"
