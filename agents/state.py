from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
    RAG Agent 运行时状态
    """
    # 输入
    query: str                          # 当前查询（可能被重写）
    original_query: str                 # 原始用户问题
    user_id: str                        # 用户 ID
    conversation_id: str                # 会话 ID
    
    # 检索结果
    documents: List[dict]               # 已检索文档列表
    document_scores: List[float]        # 文档相关性分数
    
    # 评估结果
    is_sufficient: bool                 # 证据是否充分
    missing_info: str                   # 缺失信息描述
    sufficiency_score: float            # 充分性分数（0-1）
    
    # 迭代控制
    iterations: int                     # 当前迭代次数
    max_iterations: int                 # 最大迭代次数
    prev_missing_info: str              # 上一次的缺失信息
    
    # 输出
    answer: str                         # 最终回答
    reasoning: Optional[str]            # 推理过程（CoT）
    sources: List[str]                  # 引用来源列表
    confidence: float                   # 置信度（0-1）
    
    # 元数据
    start_time: float                   # 开始时间
    retrieval_times: List[float]        # 每次检索耗时
    error: Optional[str]                # 错误信息
