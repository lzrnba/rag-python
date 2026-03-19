from typing import List
import numpy as np
import requests
from loguru import logger
from core.config import settings


class OllamaEmbedder:
    """
    基于 Ollama bge-m3 的文本向量化模块
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434"
    ):
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._dim = None  # 向量维度，首次请求后自动设置

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        批量获取文本向量

        Args:
            texts: 文本列表

        Returns:
            shape (n, dim) 的 numpy 数组
        """
        if not texts:
            return np.array([])

        # 分批处理，每批最多 32 条
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Embedded batch {i//batch_size + 1}, size={len(batch)}")

        result = np.array(all_embeddings, dtype=np.float32)
        if self._dim is None and len(result) > 0:
            self._dim = result.shape[1]
            logger.info(f"Embedding dim detected: {self._dim}")
        return result

    def embed_one(self, text: str) -> np.ndarray:
        """
        获取单个文本向量
        """
        result = self.embed([text])
        return result[0] if len(result) > 0 else np.array([])

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        调用 Ollama Embedding 接口
        """
        payload = {
            "model": self.model,
            "input": texts
        }
        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    @property
    def dim(self) -> int:
        """
        返回向量维度（需要先调用 embed 一次）
        """
        if self._dim is None:
            # 自动检测维度
            test = self.embed(["test"])
            self._dim = test.shape[1] if len(test) > 0 else 1024
        return self._dim

    def health_check(self) -> bool:
        """
        检查 Embedding 服务是否可用
        """
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            pass
        return False
