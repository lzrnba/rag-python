# -*- coding: utf-8 -*-
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
from loguru import logger


class FAISSVectorStore:
    """
    基于 FAISS 的向量检索库
    使用 IndexFlatIP（内积相似度，向量归一化后等价于余弦相似度）
    """

    def __init__(self, store_path: str = "data/vectorstore"):
        self.store_path = store_path
        self.index = None
        self.chunks: List[Dict] = []  # 与向量一一对应的文档块
        self.dim: Optional[int] = None

    def build(self, chunks: List[Dict], embedder) -> None:
        """
        构建 FAISS 索引

        Args:
            chunks: 文档块列表，每个含 content 字段
            embedder: OllamaEmbedder 实例
        """
        import faiss

        if not chunks:
            logger.warning("No chunks to build index")
            return

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")

        # 提取文本内容
        texts = [c["content"] for c in chunks]

        # 获取向量
        embeddings = embedder.embed(texts)  # shape: (n, dim)

        if len(embeddings) == 0:
            logger.error("Failed to get embeddings")
            return

        self.dim = embeddings.shape[1]

        # L2 归一化（余弦相似度）
        faiss.normalize_L2(embeddings)

        # 构建索引
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Dict]:
        """
        向量相似度检索

        Args:
            query_vec: 查询向量 shape (dim,)
            k: 返回数量

        Returns:
            文档块列表，含 vector_score 字段
        """
        import faiss

        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        # 归一化查询向量
        q = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)

        # 检索
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["vector_score"] = float(score)
            results.append(chunk)

        return results

    def save(self) -> None:
        """
        持久化索引到磁盘
        """
        import faiss

        if self.index is None:
            logger.warning("No index to save")
            return

        os.makedirs(self.store_path, exist_ok=True)
        index_path = os.path.join(self.store_path, "index.faiss")
        meta_path = os.path.join(self.store_path, "index.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "dim": self.dim}, f)

        logger.info(f"FAISS index saved to {self.store_path}")

    def load(self) -> bool:
        """
        从磁盘加载索引

        Returns:
            是否加载成功
        """
        import faiss

        index_path = os.path.join(self.store_path, "index.faiss")
        meta_path = os.path.join(self.store_path, "index.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.info("No existing FAISS index found")
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.chunks = meta["chunks"]
            self.dim = meta["dim"]
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, dim={self.dim}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
