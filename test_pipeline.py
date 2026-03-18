"""
RAG 系统测试脚本
用于本地测试 RAG 管道
"""

import asyncio
import time
from agents.state import AgentState
from agents.graph import create_agent_graph
from agents.nodes import set_retriever, set_llm_client
from retrieval.hybrid import HybridRetriever
from llm.vllm_client import QwenClient
from core.config import settings
from core.logging import setup_logging

logger = setup_logging()

async def test_rag_pipeline():
    """测试 RAG 管道"""
    
    logger.info("=" * 60)
    logger.info("RAG Pipeline Test")
    logger.info("=" * 60)
    
    try:
        # 初始化检索器
        logger.info("\n[1/3] Initializing Retriever...")
        retriever = HybridRetriever(
            vector_weight=settings.VECTOR_WEIGHT,
            bm25_weight=settings.BM25_WEIGHT,
            k=settings.TOP_K
        )
        
        # 示例文档
        sample_docs = [
            {
                "doc_id": "doc_001",
                "content": "UserController 是用户管理模块的主要控制器。login 方法用于处理用户登录请求。",
                "metadata": {"section_path": "第 3 章 → 用户模块 → UserController"}
            },
            {
                "doc_id": "doc_002",
                "content": "login 方法的签名为：public AuthResult login(HttpServletRequest request, String username, String password)。",
                "metadata": {"section_path": "第 3 章 → 用户模块 → UserController → login"}
            },
            {
                "doc_id": "doc_003",
                "content": "login 方法返回 AuthResult 对象，包含 accessToken 和 userInfo 字段。",
                "metadata": {"section_path": "第 3 章 → 用户模块 → AuthResult"}
            },
            {
                "doc_id": "doc_004",
                "content": "HttpServletRequest 是 Servlet API 中的请求对象，包含 HTTP 请求的所有信息。",
                "metadata": {"section_path": "第 2 章 → Servlet 基础 → HttpServletRequest"}
            }
        ]
        
        retriever.initialize(sample_docs)
        logger.info(f"✓ Retriever initialized with {len(sample_docs)} documents")
        
        # 初始化 LLM 客户端（模拟）
        logger.info("\n[2/3] Initializing LLM Client...")
        logger.warning("⚠ LLM client not available (vLLM service not running)")
        logger.info("  Using mock responses for testing")
        
        # 设置全局实例
        set_retriever(retriever)
        
        # 初始化 RAG 图
        logger.info("\n[3/3] Creating RAG Graph...")
        rag_graph = create_agent_graph()
        logger.info("✓ RAG graph created")
        
        # 测试查询
        logger.info("\n" + "=" * 60)
        logger.info("Testing RAG Pipeline")
        logger.info("=" * 60)
        
        test_query = "UserController.login() 方法如何调用？"
        logger.info(f"\nQuery: {test_query}")
        
        # 构建初始状态
        initial_state: AgentState = {
            "query": test_query,
            "original_query": test_query,
            "user_id": "test_user",
            "conversation_id": "test_conv",
            "documents": [],
            "document_scores": [],
            "is_sufficient": False,
            "missing_info": "",
            "sufficiency_score": 0.0,
            "iterations": 0,
            "max_iterations": 3,
            "prev_missing_info": "",
            "answer": "",
            "reasoning": None,
            "sources": [],
            "confidence": 0.0,
            "start_time": time.time(),
            "retrieval_times": [],
            "error": None,
            "top_k": 5
        }
        
        # 执行检索（不需要 LLM）
        logger.info("\nExecuting retrieval...")
        from agents.nodes import retrieve_and_rerank_node
        result = retrieve_and_rerank_node(initial_state)
        
        logger.info(f"\n✓ Retrieval completed:")
        logger.info(f"  - Documents retrieved: {len(result['documents'])}")
        logger.info(f"  - Iterations: {result['iterations']}")
        
        if result["documents"]:
            logger.info(f"\n  Retrieved documents:")
            for i, doc in enumerate(result["documents"], 1):
                logger.info(f"    {i}. {doc.get('doc_id')} (score: {doc.get('final_score', 0):.2f})")
                logger.info(f"       Path: {doc.get('metadata', {}).get('section_path', 'N/A')}")
                logger.info(f"       Content: {doc.get('content', '')[:100]}...")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())
