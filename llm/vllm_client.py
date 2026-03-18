from typing import Optional, List
import requests
from loguru import logger
from core.config import settings

class QwenClient:
    """
    Qwen2.5-7B-Instruct 客户端
    通过 vLLM 部署，提供 OpenAI 兼容的 API 接口
    """
    
    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.LLM_BASE_URL,
        api_key: str = settings.LLM_API_KEY
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数（0.0 确定性，1.0 创造性）
            max_tokens: 最大生成长度
            top_p: 核采样参数
            stop: 停止词列表
        
        Returns:
            生成的文本
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop or []
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入向量
        """
        payload = {
            "model": settings.EMBEDDING_MODEL,
            "input": texts
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            raise
