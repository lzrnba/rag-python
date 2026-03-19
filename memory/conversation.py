# -*- coding: utf-8 -*-
import json
from typing import List, Dict
from loguru import logger


class ConversationMemory:
    """
    对话历史管理 - 优先使用 Redis，降级到内存存储
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_age_seconds: int = 86400,
        redis_host: str = "lzrwebsite.icu",
        redis_port: int = 6379,
        redis_password: str = "1911lzrnb",
        redis_db: int = 0,
        key_prefix: str = "rag:conv:"
    ):
        self.max_turns = max_turns
        self.max_age_seconds = max_age_seconds
        self.key_prefix = key_prefix
        self._redis = None
        self._fallback: Dict[str, list] = {}

        try:
            import redis
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            client.ping()
            self._redis = client
            logger.info(f"Redis connected: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory fallback")
            self._redis = None

    def _key(self, cid: str) -> str:
        return f"{self.key_prefix}{cid}"

    def _get(self, cid: str) -> List[Dict]:
        if self._redis:
            try:
                raw = self._redis.get(self._key(cid))
                return json.loads(raw) if raw else []
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        return list(self._fallback.get(cid, []))

    def _set(self, cid: str, messages: List[Dict]) -> None:
        if self._redis:
            try:
                self._redis.setex(
                    self._key(cid),
                    self.max_age_seconds,
                    json.dumps(messages, ensure_ascii=False)
                )
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        self._fallback[cid] = messages

    def add_turn(self, conversation_id: str, query: str, answer: str) -> None:
        """添加一轮对话"""
        msgs = self._get(conversation_id)
        msgs.append({"role": "user", "content": query})
        msgs.append({"role": "assistant", "content": answer})
        while len(msgs) > self.max_turns * 2:
            msgs.pop(0)
            msgs.pop(0)
        self._set(conversation_id, msgs)
        logger.debug(f"Memory saved: conv={conversation_id}, turns={len(msgs)//2}")

    def get_history(self, conversation_id: str) -> List[Dict]:
        """获取完整历史"""
        return self._get(conversation_id)

    def get_history_text(self, conversation_id: str, max_turns: int = 3) -> str:
        """获取格式化历史文本注入 Prompt"""
        msgs = self._get(conversation_id)
        if not msgs:
            return ""
        recent = msgs[-(max_turns * 2):]
        lines = []
        for m in recent:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role}：{m['content']}")
        return "\n".join(lines)

    def clear(self, conversation_id: str) -> None:
        """清除会话历史"""
        if self._redis:
            try:
                self._redis.delete(self._key(conversation_id))
                logger.info(f"Redis cleared: conv={conversation_id}")
                return
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        self._fallback.pop(conversation_id, None)

    def stats(self) -> dict:
        """统计信息"""
        backend = "redis" if self._redis else "memory"
        if self._redis:
            try:
                keys = self._redis.keys(f"{self.key_prefix}*")
                total_msg = sum(
                    len(json.loads(self._redis.get(k) or "[]"))
                    for k in keys
                )
                return {"backend": backend, "total_conversations": len(keys), "total_messages": total_msg}
            except Exception:
                pass
        return {
            "backend": backend,
            "total_conversations": len(self._fallback),
            "total_messages": sum(len(v) for v in self._fallback.values())
        }


# 全局单例
memory = ConversationMemory(
    max_turns=10,
    max_age_seconds=86400,
    redis_host="lzrwebsite.icu",
    redis_port=6379,
    redis_password="1911lzrnb",
    redis_db=0
)
