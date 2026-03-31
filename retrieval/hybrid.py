# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
from core.config import settings

try:
    from sentence_transformers import CrossEncoder
except Exception:  # 可选依赖，避免启动失败
    CrossEncoder = None


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
        self.chunk_map: Dict[str, Dict] = {}
        self.bm25 = None
        self.vector_store = None
        self.embedder = None
        self._use_vector = False
        self._use_jieba = False

        # Cross-Encoder 重排配置
        self.rerank_enabled = settings.RERANK_ENABLED
        self.rerank_candidates = settings.RERANK_CANDIDATES
        self.reranker = None

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
        self.chunk_map = {chunk["doc_id"]: chunk for chunk in chunks}
        logger.info(f"Total chunks after splitting: {len(chunks)}")

        # 初始化 BM25
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

        # 初始化 Cross-Encoder 重排器（可选）
        self._init_reranker()

        # 初始化向量检索
        if embedder is not None:
            self.embedder = embedder
            self.vector_store = FAISSVectorStore(self.store_path)

            # 用文档内容 hash 判断是否需要重建，避免每次启动都重建
            current_hash = self._compute_chunks_hash(chunks)
            cached_hash = self._load_chunks_hash()

            if cached_hash == current_hash and self.vector_store.load():
                logger.info("FAISS index cache hit, skipping rebuild")
            else:
                logger.info("Documents changed or no cache, rebuilding FAISS index...")
                self.vector_store.build(chunks, embedder)
                self.vector_store.save()
                self._save_chunks_hash(current_hash)

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
        self.chunk_map = {chunk["doc_id"]: chunk for chunk in chunks}
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._init_reranker()

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
            results = self._fuse(bm25_results, vector_results, candidate_k)
        else:
            # 只有 BM25
            results = bm25_results[:candidate_k]
            for r in results:
                r["final_score"] = r.get("bm25_norm", 0.0)

        # Cross-Encoder 重排（可选）
        rerank_top_n = min(max(k, self.rerank_candidates), len(results))
        results = self._rerank(query, results, top_n=rerank_top_n)

        # 邻接补偿
        return self._expand_with_neighbors(results, k)

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

    def _init_reranker(self) -> None:
        """初始化 Cross-Encoder 重排模型（可选）"""
        if not self.rerank_enabled:
            self.reranker = None
            logger.info("Reranker disabled by config")
            return

        if self.reranker is not None:
            return

        if CrossEncoder is None:
            logger.warning("sentence-transformers unavailable, reranker disabled")
            self.reranker = None
            return

        try:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
            logger.info(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load reranker '{settings.RERANKER_MODEL}': {e}")
            self.reranker = None

    def _rerank(self, query: str, results: List[Dict], top_n: int) -> List[Dict]:
        """
        Cross-Encoder 重排：对候选文档做 query-doc 交互打分。
        无模型时自动降级为原排序。
        """
        if not results:
            return results

        if self.reranker is None:
            return results[:top_n]

        candidates = results[: min(self.rerank_candidates, len(results))]
        pairs = [(query, item.get("content", "")) for item in candidates]

        try:
            scores = self.reranker.predict(pairs)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)
                item["final_score"] = float(score)

            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", -1e9), reverse=True)
            return reranked[:top_n]
        except Exception as e:
            logger.warning(f"Rerank failed, fallback to fused ranking: {e}")
            return results[:top_n]

    def _tokenize(self, text: str) -> List[str]:
        if self._use_jieba:
            import jieba
            return list(jieba.cut(text))
        return text.split()

    def _compute_chunks_hash(self, chunks: List[Dict]) -> str:
        """计算当前文档块的内容 hash，用于判断是否需要重建索引"""
        import hashlib
        content = "|".join(sorted(c["doc_id"] + c["content"][:64] for c in chunks))
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _hash_path(self) -> str:
        import os
        return os.path.join(self.store_path, "chunks.hash")

    def _load_chunks_hash(self) -> str:
        import os
        path = self._hash_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _save_chunks_hash(self, hash_val: str) -> None:
        import os
        os.makedirs(self.store_path, exist_ok=True)
        with open(self._hash_path(), "w", encoding="utf-8") as f:
            f.write(hash_val)

    def _normalize(self, results: List[Dict], score_key: str, norm_key: str) -> List[Dict]:
        if not results:
            return results
        scores = [r.get(score_key, 0.0) for r in results]
        min_s, max_s = min(scores), max(scores)
        for r, s in zip(results, scores):
            r[norm_key] = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        return results

    def _expand_with_neighbors(self, results: List[Dict], k: int) -> List[Dict]:
        """
        命中 chunk 后补充相邻 chunk，避免上下文被截断。
        只在相邻块存在且当前结果数不足时补充。
        """
        if not results:
            return results

        expanded: List[Dict] = []
        seen = set()

        for item in results:
            doc_id = item.get("doc_id")
            if doc_id and doc_id not in seen:
                expanded.append(item)
                seen.add(doc_id)

            meta = item.get("metadata", {})
            for neighbor_key in ("prev_chunk_id", "next_chunk_id"):
                if len(expanded) >= k:
                    break
                neighbor_id = meta.get(neighbor_key)
                if not neighbor_id or neighbor_id in seen:
                    continue
                neighbor = self.chunk_map.get(neighbor_id)
                if not neighbor:
                    continue
                neighbor_item = dict(neighbor)
                neighbor_item["bm25_norm"] = item.get("bm25_norm", 0.0)
                neighbor_item["vector_norm"] = item.get("vector_norm", 0.0)
                neighbor_item["final_score"] = item.get("final_score", 0.0) * 0.85
                neighbor_item["neighbor_of"] = doc_id
                expanded.append(neighbor_item)
                seen.add(neighbor_id)

            if len(expanded) >= k:
                break

        return expanded[:k]
