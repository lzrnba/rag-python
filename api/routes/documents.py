import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger

router = APIRouter(prefix="/v1", tags=["documents"])

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "documents"
)

_embedder = None


def set_embedder(embedder):
    global _embedder
    _embedder = embedder


class ReloadResponse(BaseModel):
    success: bool
    message: str
    loaded_files: List[str]
    total_chunks: int
    vector_index_rebuilt: bool


class DocumentListResponse(BaseModel):
    docs_dir: str
    files: List[dict]
    total: int


def load_documents_from_dir() -> List[dict]:
    from retrieval.loader import DocumentLoader
    loader = DocumentLoader()
    chunks = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        return chunks
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith((".txt", ".md", ".pdf", ".docx")):
            try:
                file_chunks = loader.load_file(fpath)
                chunks.extend(file_chunks)
                logger.info(f"Loaded: {fname} ({len(file_chunks)} chunks)")
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
    return chunks


@router.post("/documents/reload", response_model=ReloadResponse)
async def reload_documents():
    """重新加载文档并重建索引（BM25 + FAISS）"""
    from agents.nodes import _retriever
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        chunks = load_documents_from_dir()
        if not chunks:
            chunks = [{
                "doc_id": "sample_001",
                "content": "UserController 是用户管理模块的主要控制器。login 方法用于处理用户登录请求。",
                "metadata": {"section_path": "示例文档 → UserController"}
            }]
            logger.info("No user docs found, loaded sample docs")

        _retriever.rebuild_index(chunks, embedder=_embedder)

        loaded_files = list(set(
            c["metadata"].get("filename", c["doc_id"]) for c in chunks
        ))
        vector_rebuilt = _retriever._use_vector

        return ReloadResponse(
            success=True,
            message=f"成功加载 {len(chunks)} 个文档块，来自 {len(loaded_files)} 个文件",
            loaded_files=loaded_files,
            total_chunks=len(chunks),
            vector_index_rebuilt=vector_rebuilt
        )
    except Exception as e:
        logger.error(f"Document reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/list", response_model=DocumentListResponse)
async def list_documents():
    """列出文档目录中的所有文件"""
    if not os.path.exists(DOCS_DIR):
        return DocumentListResponse(docs_dir=DOCS_DIR, files=[], total=0)
    supported = (".txt", ".md", ".pdf", ".docx")
    files = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith(supported):
            files.append({"name": fname, "size": os.path.getsize(fpath), "path": fpath})
    return DocumentListResponse(docs_dir=DOCS_DIR, files=files, total=len(files))
