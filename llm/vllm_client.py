from typing import Optional, List
import requests
from loguru import logger
from core.config import settings


class QwenClient:
    """
    LLM 客户端 - 兼容 Ollama / vLLM / OpenAI 格式
    Ollama 默认地址：http://localhost:11434/v1
    """

    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.LLM_BASE_URL,
        api_key: str = settings.LLM_API_KEY
    ):
        self.model = model
        # 去掉末尾斜杠，避免双斜杠问题
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成文本（兼容 Ollama OpenAI 格式）
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }

        # Ollama 不支持空 stop 列表，只在有值时传入
        if stop:
            payload["stop"] = stop

        try:
            logger.debug(f"Calling LLM: model={self.model}, url={self.base_url}")
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120  # Ollama 首次加载模型可能较慢
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"LLM response length: {len(content)} chars")
            return content
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LLM service at {self.base_url}")
            logger.error("Please make sure Ollama is running: ollama serve")
            raise
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out (120s)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入向量（使用 Ollama bge-m3 模型）
        """
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "input": texts
        }

        try:
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        检查 LLM 服务是否可用
        """
        try:
            response = self.session.get(
                f"{self.base_url.replace('/v1', '')}/api/tags",
                timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False
