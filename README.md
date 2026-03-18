# RAG 智能对话系统 - Python 端

基于 LangGraph 和 Qwen2.5-7B 的技术文档智能问答系统。

## 项目结构

```
rag-python/
├── core/                    # 核心配置
│   ├── config.py           # 配置管理
│   └── logging.py          # 日志配置
├── agents/                  # Agent 层
│   ├── state.py            # 状态定义
│   ├── nodes.py            # 节点实现
│   └── graph.py            # 状态图
├── llm/                     # LLM 服务层
│   ├── vllm_client.py      # vLLM 客户端
│   └── prompts.py          # Prompt 模板
├── retrieval/               # 检索服务层
│   └── hybrid.py           # 混合检索
├── api/                     # API 服务层
│   └── routes/
│       ├── chat.py         # 对话接口
│       └── health.py       # 健康检查
├── main.py                  # 应用入口
├── test_pipeline.py         # 测试脚本
├── requirements.txt         # 依赖
└── .env                     # 环境变量
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，配置 LLM 服务地址：

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
```

### 3. 启动 vLLM 服务（可选）

如果要使用真实的 LLM 推理，需要先启动 vLLM 服务：

```bash
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

### 4. 运行测试

```bash
python test_pipeline.py
```

### 5. 启动 FastAPI 服务

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 使用示例

### 发起对话

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "UserController.login() 方法如何调用？",
    "user_id": "user_123",
    "options": {
      "max_iterations": 3,
      "include_reasoning": true
    }
  }'
```

### 健康检查

```bash
curl "http://localhost:8000/health"
```

## 核心功能

### 1. 混合检索
- 向量检索（FAISS）：捕捉语义
- BM25 检索：精确匹配
- 加权融合：兼顾两者

### 2. 多 Agent 协作
- **Retriever**：混合检索文档
- **Grader**：评估证据充分性
- **Rewriter**：基于缺失信息重写查询
- **Generator**：生成最终回答

### 3. 闭环迭代
- 动态路由：根据证据充分性决定流程
- 自适应迭代：最多 3 次检索
- 循环检测：避免陷入无限循环

## 配置说明

### 核心参数

| 参数 | 默认值 | 说明 |
|:--|:--|:--|
| MAX_ITERATIONS | 3 | 最大迭代次数 |
| TOP_K | 5 | 返回的文档数 |
| VECTOR_WEIGHT | 0.7 | 向量检索权重 |
| BM25_WEIGHT | 0.3 | BM25 检索权重 |

### LLM 参数

| 参数 | 说明 |
|:--|:--|
| Grader 温度 | 0.0（确定性评估） |
| Rewriter 温度 | 0.3（适度创造性） |
| Generator 温度 | 0.7（生成需要创造性） |

## 开发指南

### 添加新文档

编辑 `main.py` 中的 `sample_docs`：

```python
sample_docs = [
    {
        "doc_id": "doc_001",
        "content": "文档内容",
        "metadata": {"section_path": "章节路径"}
    }
]
```

### 自定义 Prompt

编辑 `llm/prompts.py` 中的 Prompt 模板。

### 扩展检索器

在 `retrieval/hybrid.py` 中添加向量检索实现。

## 性能指标

| 指标 | 目标 |
|:--|:--|
| 平均响应时间 | < 2.5s |
| 吞吐量 | > 100 req/s |
| 准确率 (EM) | > 60% |
| 可用性 | > 99.5% |

## 故障排查

### LLM 服务不可用

如果 LLM 服务不可用，系统会返回检索结果摘要。

### 向量库不可用

如果向量库不可用，系统会降级到 BM25 检索。

### 日志位置

日志文件位于 `logs/rag.log`。

## 下一步

- [ ] 集成真实向量库（FAISS）
- [ ] 实现 Cross-Encoder 重排
- [ ] 添加缓存机制（Redis）
- [ ] 实现用户反馈收集
- [ ] 添加数据库持久化

## 许可证

MIT
