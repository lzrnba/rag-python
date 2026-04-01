import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger
from core.config import settings

router = APIRouter(prefix="/v1", tags=["documents"])

DOCS_DIR = settings.DOCUMENTS_DIR_RESOLVED

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


class ChunkListResponse(BaseModel):
    total: int
    chunks: List[Dict[str, Any]]


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


@router.get("/documents/chunks", response_model=ChunkListResponse)
async def list_chunks(
    filename: Optional[str] = Query(default=None, description="按文件名过滤，例如 Redeme.md"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000)
):
    """输出当前文档切分后的 chunk 列表（调试接口）"""
    from agents.nodes import _retriever

    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    chunks = _retriever.chunks or []
    if filename:
        chunks = [
            c for c in chunks
            if c.get("metadata", {}).get("filename") == filename
            or c.get("doc_id", "").startswith(filename)
        ]

    total = len(chunks)
    sliced = chunks[offset: offset + limit]

    result: List[Dict[str, Any]] = []
    for c in sliced:
        meta = c.get("metadata", {})
        result.append({
            "doc_id": c.get("doc_id", ""),
            "filename": meta.get("filename"),
            "section_path": meta.get("section_path"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "content": c.get("content", ""),
            "content_preview": (c.get("content", "")[:180] + "...") if len(c.get("content", "")) > 180 else c.get("content", ""),
            "final_score": c.get("final_score"),
            "bm25_norm": c.get("bm25_norm"),
            "vector_norm": c.get("vector_norm"),
            "rerank_score": c.get("rerank_score"),
            "neighbor_of": c.get("neighbor_of")
        })

    return ChunkListResponse(total=total, chunks=result)
