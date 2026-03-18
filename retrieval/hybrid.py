from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger
from core.config import settings

class HybridRetriever:
    """
    混合检索器：融合向量检索与 BM25 关键词检索
    """
    
    def __init__(
        self,
        vector_weight: float = settings.VECTOR_WEIGHT,
        bm25_weight: float = settings.BM25_WEIGHT,
        k: int = settings.TOP_K
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        
        # 这里暂时使用模拟数据，后续集成真实向量库
        self.documents = []
        self.bm25_retriever = None
        self.vector_store = None
    
    def initialize(self, documents: List[Dict]):
        """
        初始化检索器
        
        Args:
            documents: 文档列表，每个文档包含 content 和 metadata
        """
        self.documents = documents
        
        # 初始化 BM25
        doc_contents = [doc.get("content", "") for doc in documents]
        tokenized_docs = [doc.split() for doc in doc_contents]
        self.bm25_retriever = BM25Okapi(tokenized_docs)
        
        logger.info(f"HybridRetriever initialized with {len(documents)} documents")
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            k: 返回的文档数
        
        Returns:
            排序后的文档列表
        """
        if k is None:
            k = self.k
        
        if not self.documents:
            logger.warning("No documents available for retrieval")
            return []
        
        # BM25 检索
        tokenized_query = query.split()
        bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
        
        # 获取 Top-K 索引
        top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.get("doc_id", f"doc_{idx}"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "bm25_score": float(bm25_scores[idx])
            })
        
        # 归一化分数
        results = self._normalize_scores(results)
        
        # 返回 Top-K
        return sorted(results, key=lambda x: x["final_score"], reverse=True)[:k]
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        归一化分数
        """
        if not results:
            return results
        
        scores = [r["bm25_score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [
                (s - min_score) / (max_score - min_score) 
                for s in scores
            ]
        
        for result, norm_score in zip(results, normalized_scores):
            result["normalized_score"] = norm_score
            # 暂时只使用 BM25，后续集成向量检索
            result["final_score"] = norm_score
        
        return results
