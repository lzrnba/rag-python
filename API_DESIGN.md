# RAG 系统 API 设计规范

## 一、通用规范

### 响应格式
```json
{
  "code": 0,
  "message": "success",
  "request_id": "req_xxx",
  "timestamp": "2026-03-16T10:30:00Z",
  "data": {}
}
```

### 错误码
| 错误码 | HTTP | 含义 |
|:--|:--|:--|
| 0 | 200 | 成功 |
| 400001 | 400 | 参数错误 |
| 401001 | 401 | 未授权 |
| 429001 | 429 | 请求过于频繁 |
| 500001 | 500 | 服务错误 |
| 500002 | 500 | LLM 不可用 |
| 500003 | 500 | 向量库不可用 |

---

## 二、核心接口

### 2.1 对话接口
**POST /v1/chat/completions**

请求：
```json
{
  "query": "问题内容",
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "options": {
    "max_iterations": 3,
    "top_k": 5,
    "include_reasoning": true,
    "include_sources": true
  }
}
```

响应：
```json
{
  "code": 0,
  "data": {
    "conversation_id": "conv_456",
    "answer": "回答内容",
    "reasoning": "<thought>推理过程</thought>",
    "sources": [
      {
        "doc_id": "doc_001",
        "doc_name": "文档名",
        "section_path": "章节路径",
        "relevance_score": 0.92,
        "snippet": "文档片段"
      }
    ],
    "retrieval_process": {
      "iterations": 2,
      "sufficiency_score": 0.88
    },
    "confidence": 0.85,
    "response_time_ms": 2340
  }
}
```

### 2.2 会话接口
- **POST /v1/conversations** - 创建会话
- **GET /v1/conversations/{id}/messages** - 获取历史
- **DELETE /v1/conversations/{id}** - 删除会话

### 2.3 文档接口
- **POST /v1/documents/upload** - 上传文档
- **GET /v1/documents/{id}/status** - 查询处理进度
- **GET /v1/documents** - 列出文档
- **DELETE /v1/documents/{id}** - 删除文档

### 2.4 反馈接口
**POST /v1/feedback**
```json
{
  "message_id": "msg_002",
  "conversation_id": "conv_456",
  "rating": 1,
  "user_id": "user_123"
}
```

### 2.5 系统接口
- **GET /v1/health** - 健康检查
- **GET /v1/statistics** - 系统统计

### 2.6 认证接口
- **POST /v1/auth/login** - 登录
- **POST /v1/auth/refresh** - 刷新 Token

---

## 三、速率限制

| 端点 | 限制 |
|:--|:--|
| /v1/chat/completions | 10 req/min |
| /v1/documents/upload | 5 req/min |
| 其他 | 100 req/min |

---

## 四、文档自动生成

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
