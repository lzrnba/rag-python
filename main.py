from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from core.config import settings
from core.logging import setup_logging
from api.routes import chat, health
from agents.graph import create_agent_graph
from agents.nodes import set_retriever, set_llm_client
from retrieval.hybrid import HybridRetriever
from llm.vllm_client import QwenClient

# 初始化日志
logger = setup_logging()

# 全局变量
rag_graph = None
retriever = None
llm_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    logger.info("🚀 RAG Service Starting...")
    
    try:
        # 初始化 LLM 客户端
        global llm_client
        llm_client = QwenClient(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY
        )
        logger.info(f"✓ LLM client initialized: {settings.LLM_MODEL}")
        
        # 初始化检索器
        global retriever
        retriever = HybridRetriever(
            vector_weight=settings.VECTOR_WEIGHT,
            bm25_weight=settings.BM25_WEIGHT,
            k=settings.TOP_K
        )
        
        # 初始化示例文档（开发用）
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
            }
        ]
        retriever.initialize(sample_docs)
        logger.info(f"✓ Retriever initialized with {len(sample_docs)} documents")
        
        # 设置全局实例
        set_retriever(retriever)
        set_llm_client(llm_client)
        
        # 初始化 RAG 图
        global rag_graph
        rag_graph = create_agent_graph()
        chat.rag_graph = rag_graph
        logger.info("✓ RAG graph initialized")
        
        logger.info(f"✓ RAG Service started on {settings.HOST}:{settings.PORT}")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize RAG service: {e}", exc_info=True)
        raise
    
    yield
    
    # 关闭事件
    logger.info("🛑 RAG Service Shutting Down...")

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 智能对话系统",
    description="基于 Qwen2.5-7B 的技术文档问答 API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(health.router)

# 根路由
@app.get("/")
async def root():
    """根路由"""
    return {
        "name": "RAG 智能对话系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# 运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
