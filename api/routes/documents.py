import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger

router = APIRouter(prefix="/v1", tags=["documents"])

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "documents")


class ReloadResponse(BaseModel):
    success: bool
    message: str
    loaded_files: List[str]
    total_documents: int


class DocumentListResponse(BaseModel):
    docs_dir: str
    files: List[dict]
    total: int


def load_documents_from_dir() -> List[dict]:
    """从 data/documents/ 目录加载所有文档"""
    docs = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        return docs

    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith((".txt", ".md")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    docs.append({
                        "doc_id": fname,
                        "content": content,
                        "metadata": {
                            "section_path": fname,
                            "source": fpath,
                            "size": os.path.getsize(fpath)
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
    return docs


@router.post("/documents/reload", response_model=ReloadResponse)
async def reload_documents():
    """
    重新加载文档接口
    
    从 data/documents/ 目录重新读取所有 .txt 和 .md 文件，
    并更新检索器索引，无需重启服务。
    """
    from agents.nodes import _retriever

    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        docs = load_documents_from_dir()

        if not docs:
            # 没有用户文档时加载示例文档
            docs = [
                {
                    "doc_id": "sample_001",
                    "content": "UserController 是用户管理模块的主要控制器。login 方法用于处理用户登录请求。",
                    "metadata": {"section_path": "示例文档 → UserController"}
                }
            ]
            logger.info("No user docs found, loaded sample docs")

        _retriever.initialize(docs)
        loaded_files = [d["doc_id"] for d in docs]

        logger.info(f"Documents reloaded: {len(docs)} files from {DOCS_DIR}")

        return ReloadResponse(
            success=True,
            message=f"成功加载 {len(docs)} 篇文档",
            loaded_files=loaded_files,
            total_documents=len(docs)
        )

    except Exception as e:
        logger.error(f"Document reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/list", response_model=DocumentListResponse)
async def list_documents():
    """
    列出文档目录中的所有文件
    """
    if not os.path.exists(DOCS_DIR):
        return DocumentListResponse(docs_dir=DOCS_DIR, files=[], total=0)

    files = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith((".txt", ".md")):
            files.append({
                "name": fname,
                "size": os.path.getsize(fpath),
                "path": fpath
            })

    return DocumentListResponse(
        docs_dir=DOCS_DIR,
        files=files,
        total=len(files)
    )
