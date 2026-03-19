# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
from core.config import settings


class HybridRetriever:
    """
    混合检索器：FAISS 向量检索 + BM25 关键词检索
    支持 jieba 中文分词
    """

    def __init__(
        self,
        vector_weight: float = settings.VECTOR_WEIGHT,
        bm25_weight: float = settings.BM25_WEIGHT,
        k: int = settings.TOP_K,
        store_path: str = "data/vectorstore"
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        self.store_path = store_path

        self.chunks: List[Dict] = []
        self.bm25 = None
        self.vector_store = None
        self.embedder = None
        self._use_vector = False
        self._use_jieba = False

        # 尝试导入可选依赖
        try:
            import jieba
            self._use_jieba = True
            jieba.setLogLevel('WARNING')
            logger.info("jieba loaded: Chinese tokenization enabled")
        except ImportError:
            logger.warning("jieba not installed, using whitespace tokenization")

    def initialize(self, documents: List[Dict], embedder=None) -> None:
        """
        初始化检索器

        Args:
            documents: 原始文档列表（未分块）
            embedder: OllamaEmbedder 实例（可选，不传则跳过向量检索）
        """
        from rank_bm25 import BM25Okapi
        from retrieval.loader import DocumentLoader
        from retrieval.vector_store import FAISSVectorStore

        # 文档分块
        loader = DocumentLoader()
        chunks = []
        for doc in documents:
            content = doc.get("content", "").strip()
            if not content:
                continue
            # 如果内容较短，直接作为一个 chunk
            if len(content) <= 600:
                chunks.append({
                    "doc_id": doc.get("doc_id", ""),
                    "content": content,
                    "metadata": doc.get("metadata", {})
                })
            else:
                # 使用 loader 分块
                sub_chunks = loader._chunk(content)
                fname = doc.get("doc_id", "doc")
                for i, chunk in enumerate(sub_chunks):
                    meta = dict(doc.get("metadata", {}))
                    meta["chunk_index"] = i
                    meta["total_chunks"] = len(sub_chunks)
                    meta["section_path"] = meta.get("section_path", fname) + f" [chunk {i+1}/{len(sub_chunks)}]"
                    chunks.append({
                        "doc_id": f"{fname}_chunk_{i}",
                        "content": chunk,
                        "metadata": meta
                    })

        self.chunks = chunks
        logger.info(f"Total chunks after splitting: {len(chunks)}")

        # 初始化 BM25
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

        # 初始化向量检索
        if embedder is not None:
            self.embedder = embedder
            self.vector_store = FAISSVectorStore(self.store_path)

            # 尝试加载已有索引
            if not self.vector_store.load():
                # 重新构建
                self.vector_store.build(chunks, embedder)
                self.vector_store.save()

            self._use_vector = self.vector_store.is_ready
            logger.info(f"Vector store ready: {self._use_vector}")
        else:
            logger.info("No embedder provided, using BM25 only")

    def rebuild_index(self, documents: List[Dict], embedder=None) -> None:
        """
        强制重建索引（文档更新时调用）
        """
        from retrieval.vector_store import FAISSVectorStore
        from retrieval.loader import DocumentLoader
        from rank_bm25 import BM25Okapi

        loader = DocumentLoader()
        chunks = []
        for doc in documents:
            content = doc.get("content", "").strip()
            if not content:
                continue
            if len(content) <= 600:
                chunks.append({
                    "doc_id": doc.get("doc_id", ""),
                    "content": content,
                    "metadata": doc.get("metadata", {})
                })
            else:
                sub_chunks = loader._chunk(content)
                fname = doc.get("doc_id", "doc")
                for i, chunk in enumerate(sub_chunks):
                    meta = dict(doc.get("metadata", {}))
                    meta["chunk_index"] = i
                    meta["section_path"] = meta.get("section_path", fname) + f" [chunk {i+1}/{len(sub_chunks)}]"
                    chunks.append({
                        "doc_id": f"{fname}_chunk_{i}",
                        "content": chunk,
                        "metadata": meta
                    })

        self.chunks = chunks
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        if embedder is not None:
            self.embedder = embedder
            self.vector_store = FAISSVectorStore(self.store_path)
            self.vector_store.build(chunks, embedder)
            self.vector_store.save()
            self._use_vector = self.vector_store.is_ready

        logger.info(f"Index rebuilt: {len(chunks)} chunks, vector={self._use_vector}")

    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        混合检索
        """
        if k is None:
            k = self.k

        if not self.chunks:
            logger.warning("No documents available")
            return []

        candidate_k = min(k * 3, len(self.chunks))

        # BM25 检索
        bm25_results = self._bm25_search(query, candidate_k)

        # 向量检索
        if self._use_vector and self.embedder:
            vector_results = self._vector_search(query, candidate_k)
            results = self._fuse(bm25_results, vector_results, k)
        else:
            # 只有 BM25
            results = bm25_results[:k]
            for r in results:
                r["final_score"] = r.get("normalized_score", 0.0)

        return results

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        # 归一化
        return self._normalize(results, "bm25_score", "bm25_norm")

    def _vector_search(self, query: str, k: int) -> List[Dict]:
        query_vec = self.embedder.embed_one(query)
        if len(query_vec) == 0:
            return []
        results = self.vector_store.search(query_vec, k)
        return self._normalize(results, "vector_score", "vector_norm")

    def _fuse(self, bm25_results: List[Dict], vector_results: List[Dict], k: int) -> List[Dict]:
        """
        加权融合 BM25 和向量检索结果
        """
        scores: Dict[str, dict] = {}

        for r in bm25_results:
            doc_id = r["doc_id"]
            scores[doc_id] = dict(r)
            scores[doc_id]["bm25_norm"] = r.get("bm25_norm", 0.0)
            scores[doc_id]["vector_norm"] = 0.0

        for r in vector_results:
            doc_id = r["doc_id"]
            if doc_id in scores:
                scores[doc_id]["vector_norm"] = r.get("vector_norm", 0.0)
            else:
                scores[doc_id] = dict(r)
                scores[doc_id]["bm25_norm"] = 0.0
                scores[doc_id]["vector_norm"] = r.get("vector_norm", 0.0)

        # 计算融合分数
        for doc_id, item in scores.items():
            item["final_score"] = (
                self.vector_weight * item["vector_norm"] +
                self.bm25_weight * item["bm25_norm"]
            )

        sorted_results = sorted(scores.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:k]

    def _tokenize(self, text: str) -> List[str]:
        if self._use_jieba:
            import jieba
            return list(jieba.cut(text))
        return text.split()

    def _normalize(self, results: List[Dict], score_key: str, norm_key: str) -> List[Dict]:
        if not results:
            return results
        scores = [r.get(score_key, 0.0) for r in results]
        min_s, max_s = min(scores), max(scores)
        for r, s in zip(results, scores):
            r[norm_key] = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        return results
