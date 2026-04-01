# RAG 系统综合总结文档

## 一、系统概述

### 1.1 项目定位

这是一个基于 **LangGraph** 和 **Qwen2.5-7B** 的技术文档智能问答系统（RAG - Retrieval-Augmented Generation），采用 Python + FastAPI 实现智能推理层，支持多 Agent 协作、闭环迭代检索和混合检索。

### 1.2 核心特性

| 特性 | 说明 |
|:--|:--|
| 多 Agent 协作 | Retriever（检索）、Grader（评估）、Rewriter（重写）、Generator（生成）四个智能体分工协作 |
| 闭环迭代检索 | 支持动态路由和自适应迭代，最多 3 次检索尝试 |
| 混合检索 | 向量检索（语义理解）+ BM25（精确匹配）加权融合 |
| 可解释性 | 输出包含推理过程、引用来源和置信度评分 |
| 模块化设计 | 五层架构，松耦合高内聚，易于扩展和维护 |

---

## 二、系统架构

### 2.1 整体架构（五层）

```
┌─────────────────────────────────────────┐
│        用户交互层（Web 前端/客户端）      │
└────────────────┬────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────┐
│    应用服务层（FastAPI）                 │
│  - 对话接口 /v1/chat/completions        │
│  - 健康检查 /health                     │
│  - 文档管理 /v1/documents               │
│  - 会话管理 /v1/conversations           │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    智能推理层（LangGraph + Python）      │
│  - Retriever（检索）                    │
│  - Grader（评估）                       │
│  - Rewriter（重写）                     │
│  - Generator（生成）                    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    检索服务层                            │
│  - 混合检索（BM25 + 向量）              │
│  - Cross-Encoder 重排                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    数据存储层                            │
│  - FAISS 向量索引                        │
│  - MySQL 元数据                          │
│  - 原始文档存储                          │
└─────────────────────────────────────────┘
```

### 2.2 核心数据流

```
用户问题
   ↓
[FastAPI] 接收请求 → 验证 → 记录日志
   ↓
[LangGraph] 启动状态图
   ├─→ [Retriever] 混合检索 → Top-5 文档
   ├─→ [Grader] 评估充分性 → 布尔值 + 缺失信息
   ├─→ 条件路由
   │   ├─ 充分 → [Generator]
   │   └─ 不充分 → [Rewriter] → 回到 Retriever（闭环）
   └─→ [Generator] 生成回答 → 推理过程 + 引用来源
   ↓
[FastAPI] 返回结果 → 保存数据库 → 返回用户
```

---

## 三、核心模块实现

### 3.1 项目结构

```
rag-python/
├── core/                    # 核心配置层（3 个文件）
│   ├── __init__.py
│   ├── config.py           # Settings 配置类
│   └── logging.py          # loguru 日志配置
├── agents/                  # Agent 层（4 个文件）
│   ├── __init__.py
│   ├── state.py            # AgentState TypedDict（37 行）
│   ├── nodes.py            # 4 个节点实现（245 行）
│   └── graph.py            # LangGraph 状态图（84 行）
├── llm/                     # LLM 服务层（3 个文件）
│   ├── __init__.py
│   ├── vllm_client.py      # QwenClient 类（101 行）
│   └── prompts.py          # 4 个 Prompt 模板（66 行）
├── retrieval/               # 检索服务层（2 个文件）
│   ├── __init__.py
│   └── hybrid.py           # HybridRetriever 类（110 行）
├── api/                     # API 服务层（4 个文件）
│   ├── __init__.py
│   ├── routes/__init__.py
│   ├── chat.py             # 对话接口（125 行）
│   └── health.py           # 健康检查（21 行）
├── main.py                  # 应用入口（131 行）
├── test_pipeline.py         # 测试脚本（131 行）
├── requirements.txt         # 依赖列表
├── .env                     # 环境变量
└── README.md               # 使用说明
```

### 3.2 状态管理（AgentState）

| 字段                  | 类型          | 说明           |
| :------------------ | :---------- | :----------- |
| `query`             | str         | 当前查询（可能被重写）  |
| `original_query`    | str         | 用户原始问题       |
| `user_id`           | str         | 用户 ID        |
| `conversation_id`   | str         | 会话 ID        |
| `documents`         | List[Dict]  | 文档列表         |
| `document_scores`   | List[float] | 相关性分数        |
| `is_sufficient`     | bool        | 证据是否充分       |
| `missing_info`      | str         | 缺失信息         |
| `sufficiency_score` | float       | 充分性分数（0-1）   |
| `iterations`        | int         | 当前迭代次数       |
| `max_iterations`    | int         | 最大迭代次数       |
| `prev_missing_info` | str         | 上次缺失信息（循环检测） |
| `answer`            | str         | 最终回答         |
| `reasoning`         | str         | 推理过程         |
| `sources`           | List[str]   | 引用来源         |
| `confidence`        | float       | 置信度（0-1）     |

### 3.3 四个 Agent 节点

#### 3.3.1 Retriever（检索节点）

**职责**：执行混合检索，返回 Top-K 相关文档

**流程**：
```
query → 向量检索 (FAISS) + BM25 检索 → 加权融合 → Cross-Encoder 重排 → Top-5 文档
```

**关键参数**：
- 初筛 Top-K：10
- 最终 Top-K：5
- 向量权重：0.7
- BM25 权重：0.3

#### 3.3.2 Grader（评估节点）

**职责**：判断检索结果是否足以回答问题

**Prompt 设计**：
```
你是证据评估专家。
问题：{query}
文档内容：{context}
请判断这些文档是否足以回答问题。
输出 JSON: {"is_sufficient": true/false, "missing_info": "..."}
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.0（确定性评估）
- Max tokens：256

#### 3.3.3 Rewriter（重写节点）

**职责**：基于缺失信息生成更具体的查询

**Prompt 设计**：
```
你是查询重写专家。
原始问题：{original_query}
缺失信息：{missing_info}
请生成一个更具体、指向性更强的新查询。
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.3（适度创造性）
- Max tokens：128

#### 3.3.4 Generator（生成节点）

**职责**：基于充分文档生成最终回答

**Prompt 设计**：
```
你是技术文档智能助理。
文档内容：{context}
问题：{query}
请按以下步骤回答：
1. 在 <thought> 标签内输出推理过程
2. 在标签外输出最终答案
3. 答案中应引用相关章节路径
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.7（生成创造性）
- Max tokens：1024

### 3.4 动态路由逻辑

```python
def route_based_on_sufficiency(state):
    # 规则 1：充分性分数 > 0.85 → 生成
    if state["sufficiency_score"] > 0.85:
        return "sufficient"
    
    # 规则 2：达到最大迭代次数 → 强制生成
    if state["iterations"] >= MAX_ITER:
        return "sufficient"
    
    # 规则 3：缺失信息重复（陷入循环）→ 强制生成
    if state["prev_missing_info"] == state["missing_info"]:
        return "sufficient"
    
    # 其他情况 → 继续迭代
    return "insufficient"
```

---

## 四、API 设计

### 4.1 通用规范

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "request_id": "req_xxx",
  "timestamp": "2026-03-16T10:30:00Z",
  "data": {}
}
```

**错误码**：
| 错误码 | HTTP |  含义 |
|:--|:--|:--|
|  0 | 200 | 成功 |
| 400001 | 400 | 参数错误 |
| 401001 | 401 | 未授权 |
| 429001 | 429 | 请求过于频繁 |
| 500001 | 500 | 服务错误 |
| 500002 | 500 | LLM 不可用 |
| 500003 | 500 | 向量库不可用 |

### 4.2 核心接口

#### 4.2.1 对话接口
**POST /v1/chat/completions**

**请求**：
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

**响应**：
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

#### 4.2.2 会话接口
- **POST /v1/conversations** - 创建会话
- **GET /v1/conversations/{id}/messages** - 获取历史
- **DELETE /v1/conversations/{id}** - 删除会话

#### 4.2.3 文档接口
- **POST /v1/documents/upload** - 上传文档
- **GET /v1/documents/{id}/status** - 查询处理进度
- **GET /v1/documents** - 列出文档
- **DELETE /v1/documents/{id}** - 删除文档

#### 4.2.4 反馈接口
**POST /v1/feedback**
```json
{
  "message_id": "msg_002",
  "conversation_id": "conv_456",
  "rating": 1,
  "user_id": "user_123"
}
```

#### 4.2.5 系统与认证接口
- **GET /v1/health** - 健康检查
- **GET /v1/statistics** - 系统统计
- **POST /v1/auth/login** - 登录
- **POST /v1/auth/refresh** - 刷新 Token

### 4.3 速率限制

| 端点 | 限制 |
|:--|:--|
| /v1/chat/completions | 10 req/min |
| /v1/documents/upload | 5 req/min |
| 其他 | 100 req/min |

---

## 五、置信度评估机制

### 5.1 检索阶段置信度

| 指标 | 说明 |
|:--|:--|
| 相似度分数 | 余弦相似度，越接近 1 越相关 |
| 重排序分数 | Cross-Encoder 深度交互打分 |
| Top-K 聚合 | 观察前 K 个结果的分数分布 |

### 5.2 生成阶段置信度

| 指标 | 说明 |
|:--|:--|
| 困惑度 (PPL) | 越低表示模型越"顺理成章" |
| Token 概率均值 | 每个词生成概率的平均值 |
| 自洽性检测 | 多次生成内容一致性 |

### 5.3 当前实现

```python
confidence = 0.6（基础分）
           + sufficiency_score × 0.2
           + min(doc 数量/3, 1.0) × 0.2
# 最大值 1.0
```

### 5.4 拒答逻辑

1. **检索阈值**：最高相似度 < 0.6 → "知识库中暂无相关信息"
2. **幻觉检测**：关键实体不在文档中 → 标记低置信度

---

## 六、缓存策略

### 6.1 三层缓存架构

```
L1 缓存（内存）
├─ 最近 100 个查询结果
├─ TTL：1 小时
└─ 命中率目标：60%

L2 缓存（Redis）
├─ 最近 1000 个查询结果
├─ TTL：24 小时
└─ 命中率目标：30%

L3 缓存（向量库）
├─ 已计算的查询向量
├─ TTL：永久
└─ 避免重复计算 Embedding
```

### 6.2 缓存键设计

```python
query_cache_key = f"query:{hash(query)}:{user_id}"
vector_cache_key = f"vector:{hash(query)}"
doc_cache_key = f"doc:{doc_id}:{version}"
```

### 6.3 缓存失效策略

- **文档更新**：清除相关查询缓存、向量缓存，重建 FAISS 索引
- **术语词典更新**：清除所有查询缓存
- **定期清理**：每天凌晨 2 点清理过期缓存

---

## 七、错误处理与降级

### 7.1 故障场景与应对

| 故障场景 | 影响 | 降级方案 |
|:--|:--|:--|
| LLM 服务不可用 | 无法评估、重写、生成 | 直接返回检索结果摘要 |
| 向量库不可用 | 无法进行向量检索 | 仅使用 BM25 检索 |
| Redis 不可用 | 缓存失效 | 继续运行，性能下降 |
| 数据库不可用 | 无法保存反馈 | 继续运行，反馈丢失 |

### 7.2 降级实现

```python
def retrieve_with_fallback(query):
    try:
        return hybrid_retrieve(query)
    except VectorStoreError:
        return bm25_retrieve(query)  # 降级到 BM25
    except Exception as e:
        return []  # 最后降级

def generate_with_fallback(query, documents):
    try:
        return generate_answer(query, documents)
    except LLMError:
        return summarize_documents(documents)  # 返回文档摘要
```

---

## 八、性能指标

### 8.1 关键性能指标（KPI）

| 指标 | 目标 | 说明 |
|:--|:--|:--|
| 平均响应时间 | < 2.5s | P95 < 4s |
| 吞吐量 | > 100 req/s | 单机 |
| 缓存命中率 | > 60% | L1+L2 |
| 可用性 | > 99.5% | 月度 |
| EM 准确率 | > 60% | 测试集 |

### 8.2 优化方向

1. **向量检索优化**：FAISS GPU 加速、批量查询、预计算热点
2. **LLM 推理优化**：vLLM PagedAttention、批量推理、量化
3. **缓存优化**：智能预热、分布式缓存（Redis Cluster）
4. **并发优化**：异步处理、请求队列、连接池复用

---

## 九、监控与可观测性

### 9.1 关键指标

**系统级**：请求延迟（P50/P95/P99）、错误率、缓存命中率、并发连接数

**业务级**：用户满意度、回答准确率、平均迭代次数、文档覆盖率

**资源级**：CPU 使用率、内存使用率、GPU 使用率、磁盘 I/O

### 9.2 日志设计

每个请求记录：
- request_id（唯一标识）
- user_id、query、conversation_id
- iterations（迭代次数）
- response_time_ms（响应时间）
- cache_hit（是否命中缓存）
- error（错误信息）

---

## 十、部署架构

### 10.1 开发环境

```
- Spring Boot：localhost:8080
- FastAPI：localhost:8000
- MySQL：localhost:3306
- Redis：localhost:6379
- FAISS：本地文件
```

### 10.2 生产环境（高可用）

```
- Spring Boot：多实例 + 负载均衡
- FastAPI：多实例 + 负载均衡
- MySQL：主从复制 + 读写分离
- Redis：Cluster 模式
- FAISS：分布式索引
- vLLM：多 GPU 推理
```

---

## 十一、实现进度

### Phase 1：核心功能（✅ 已完成）

| 功能 | 状态 | 说明 |
|:--|:--|:--|
| LangGraph 状态图搭建 | ✅ | 已完成多节点编排与条件路由 |
| Retriever / Grader / Rewriter / Generator 四智能体 | ✅ | 节点职责已落地并联通 |
| 闭环迭代检索（最多 3 次） | ✅ | 支持证据不足时重写查询再检索 |
| FastAPI 服务封装 | ✅ | 对话、文档、健康检查接口可用 |
| 基础前端与接口联调 | ✅ | 已可通过前端页面发起问答 |

### Phase 2：检索增强与质量提升（✅ 已完成约 90%）

| 能力                          | 状态    | 说明                                                   |
| :-------------------------- | :---- | :--------------------------------------------------- |
| FAISS 向量检索                  | ✅     | `retrieval/vector_store.py` 已实现并持久化                  |
| Embedding 接入（Ollama/bge-m3） | ✅     | `retrieval/embedder.py` 已支持批量向量化                     |
| 混合检索（向量 + BM25）             | ✅     | `retrieval/hybrid.py` 已实现加权融合                        |
| 中文分词优化（jieba）               | ✅     | BM25 查询分词已接入                                         |
| 文档解析（txt/md/pdf/docx）       | ✅     | `retrieval/loader.py` 已支持多格式                         |
| 结构化分块与邻接补偿                  | ✅     | 结构感知分块 + 邻接 chunk 扩展已上线                              |
| 文档重载重建索引                    | ✅     | `/v1/documents/reload` 可重建 BM25+FAISS                |
| 索引一致性（文档 hash）              | ✅     | 文档变更自动触发索引重建，避免旧索引污染                                 |
| Cross-Encoder 重排            | ✅     | 已接入 `sentence-transformers` CrossEncoder，支持候选重排与降级回退 |
| 术语词典与规则校验                   | ✅ 已补充 | 已在文档新增术语词典（核心 RAG 概念、检索链路与评估指标） |

### Phase 3：性能与稳定性（🟡 部分完成）

| 能力 | 状态 | 说明 |
|:--|:--|:--|
| Redis 对话记忆 | ✅ | 会话历史已持久化 |
| 低相关检索快速拒答（fallback） | ✅ | 低分场景不再多轮检索，直接简洁回复 |
| 索引缓存策略 | ✅ | 缓存命中可跳过不必要重建 |
| L1/L2 查询结果缓存体系 | ⏳ 未完成 | 尚未形成完整 query 结果缓存链路 |
| 压测与并发优化专项 | ⏳ 未完成 | 未见系统化吞吐/延迟优化报告 |

### Phase 4：用户体验与产品化（🟡 部分完成）

| 能力 | 状态 | 说明 |
|:--|:--|:--|
| Web 聊天界面 | ✅ | `frontend/index.html` 可用 |
| 流式回答接口 | ✅ | `/v1/chat/stream` 已支持 SSE |
| 会话历史接口 | ✅ | 已支持历史查询与清理 |
| 用户反馈闭环 | ⏳ 未完成 | 缺少完整反馈存储与分析链路 |
| 可视化知识库管理 | ⏳ 未完成 | 目前以 API 操作为主 |
| 个性化推荐 | ⏳ 未完成 | 尚未实现推荐策略 |

### 当前总体结论

- 项目已稳定越过 **Phase 1**。
- 当前处于 **Phase 2 完成阶段**（核心检索增强能力已落地，正在向 Phase 3 稳定性与性能优化推进）。
- 已提前具备部分 **Phase 3/4** 能力（Redis 记忆、fallback、流式接口、前端页面）。

---

## 十二、技术栈

| 组件 | 技术 | 版本 |
|:--|:--|:--|
| Web 框架 | FastAPI | 0.109.0 |
| Agent 编排 | LangGraph | 0.0.20 |
| LLM 集成 | LangChain | 0.1.0 |
| 检索 | BM25 | 0.2.2 |
| 向量 | FAISS | 1.7.4 |
| Embedding | sentence-transformers | 2.3.0 |
| 日志 | loguru | 0.7.2 |
| 配置 | pydantic | 2.5.0 |

---

## 十三、术语词典（Phase 2 补充）

| 术语 | 英文/缩写 | 定义 | 在本项目中的作用 |
|:--|:--|:--|:--|
| 检索增强生成 | RAG | 先检索外部知识，再由模型生成回答的范式 | 降低幻觉、提升答案可追溯性 |
| 状态图编排 | LangGraph | 以有向图形式组织 Agent 节点与路由 | 驱动 Retriever/Grader/Rewriter/Generator 闭环 |
| 混合检索 | Hybrid Retrieval | 向量检索 + BM25 融合检索 | 同时兼顾语义召回与关键词精确匹配 |
| 向量检索 | Vector Retrieval | 将 query/doc 编码后按向量相似度召回 | 提升语义相关问题的召回率 |
| 稀疏检索 | BM25 | 基于词频与逆文档频率的传统检索算法 | 提升术语、字段、代码名等精确命中 |
| 重排模型 | Cross-Encoder Reranker | 对候选文档做逐对打分重排 | 提升 Top-K 文档排序质量 |
| 文档分块 | Chunking | 将长文档切成可检索粒度的片段 | 避免上下文超长，提升召回精度 |
| 邻接补偿 | Neighbor Expansion | 命中 chunk 时补充相邻 chunk | 缓解边界截断造成的信息缺失 |
| 证据充分性 | Sufficiency Score | Grader 对“文档是否足够回答问题”的评分 | 决定生成、重写还是继续检索 |
| 查询重写 | Query Rewrite | 基于缺失信息重写 query | 在证据不足时提升下一轮检索命中 |
| 兜底回答 | Fallback | 低相关场景直接返回“知识库未收录”提示 | 防止模型在无证据场景胡编 |
| 文档置信模式 | doc_grounded / fallback | 区分是否基于文档证据生成 | 便于前端展示可信度与来源状态 |
| SSE 流式输出 | Server-Sent Events | 服务端逐 token 推送前端 | 改善交互体验，降低首字延迟感知 |

---

## 十四、设计亮点

1. **LangGraph 状态图**：完全支持闭环迭代、动态路由决策、状态变更可追踪
2. **混合检索**：兼顾语义理解和精确匹配，加权融合策略可配置
3. **多 Agent 协作**：清晰的职责划分、显式的通信机制、易于扩展
4. **可解释性**：完整的推理过程（CoT）、引用来源追踪、置信度评分
5. **模块化设计**：松耦合高内聚、易于测试、易于维护

---

## 十五、快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行测试
```bash
python test_pipeline.py
```

### 3. 启动服务
```bash
python main.py
```

### 4. 测试 API
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"query": "问题内容", "user_id": "user_123"}'
```

### 5. 访问文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 十六、总结

**Phase 2 核心检索增强能力已完成**，系统已经具备 FAISS 向量检索、中文分词优化、结构化分块与邻接补偿、Cross-Encoder 重排、文档重载重建索引等关键能力。

**当前局限**：
- 术语词典已补充，规则校验能力尚未建设
- L1/L2 查询结果缓存体系尚未形成
- 压测与并发优化缺少系统化基准数据
- 用户反馈闭环与可视化知识库管理仍在规划中

**下一步计划**：
1. 建设术语词典与规则校验模块
2. 建立查询结果缓存（内存 + Redis）体系
3. 开展吞吐/延迟压测并做并发优化
4. 完善用户反馈闭环与知识库管理能力
5. 逐步推进产品化能力（推荐与运维可观测性）

---

**文档版本**：v1.0  
**生成时间**：2026-03-31  
**项目地址**：`E:\Project Design\project\python-rag\rag-python`
