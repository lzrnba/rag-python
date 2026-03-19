from typing import TypedDict, List, Optional, Annotated
import operator

class AgentState(TypedDict):
    """
    RAG Agent 运行时状态
    """
    # 输入
    query: str
    original_query: str
    user_id: str
    conversation_id: str

    # 检索结果
    documents: List[dict]
    document_scores: List[float]

    # 评估结果
    is_sufficient: bool
    missing_info: str
    sufficiency_score: float

    # 迭代控制
    iterations: int
    max_iterations: int
    prev_missing_info: str

    # 输出
    answer: str
    reasoning: Optional[str]
    sources: List[str]
    confidence: float

    # 检索评分路由
    top_doc_score: float     # 检索到的最高文档分数，用于评分路由
    skip_retrieval: bool     # 记忆命中时跳过检索直接生成

    # 对话历史
    chat_history: str        # 格式化的历史文本，注入 Prompt 用

    # 元数据
    start_time: float
    retrieval_times: List[float]
    error: Optional[str]
    top_k: int
