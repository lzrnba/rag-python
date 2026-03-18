# RAG 系统架构设计文档

> **版本**：v1.0  
> **重点**：系统架构、数据流、模块交互

---

## 一、系统架构概览

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户交互层                               │
│              (Web 前端 / ChatBox / 客户端)                   │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────────┐
│                  应用服务层 (Spring Boot)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 文档管理模块  │  │ 用户认证模块  │  │ 反馈收集模块  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────────┐
│              智能推理层 (FastAPI + Python)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         LangGraph 状态图 (Agent 编排)               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │ Retriever│→ │  Grader  │→ │ Rewriter │          │    │
│  │  └──────────┘  └──────────┘  └──────────┘          │    │
│  │       ↑                            ↓                │    │
│  │       └────────────────────────────┘                │    │
│  │                    ↓                                │    │
│  │            ┌──────────────┐                         │    │
│  │            │  Generator   │                         │    │
│  │            └──────────────┘                         │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──┐  ┌──────▼──┐  ┌─────▼──────┐
│ 检索服务  │  │ LLM 服务 │  │ 缓存服务   │
│ (混合检索)│  │(vLLM)   │  │ (Redis)   │
└──────────┘  └─────────┘  └────────────┘
        │            │            │
┌───────▼──────────────────────────▼──────┐
│          数据存储层                      │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ FAISS    │  │ MySQL    │  │ 原始文档 │ │
│  │ 向量索引  │  │ 元数据   │  │ 存储   │ │
│  └──────────┘  └──────────┘  └────────┘ │
└──────────────────────────────────────────┘
```

### 1.2 核心数据流

```
用户问题
   ↓
[Spring Boot] 接收请求 → 验证权限 → 记录日志
   ↓
[FastAPI] 接收请求
   ↓
[LangGraph] 启动状态图
   ├─→ [Retriever] 混合检索
   │   ├─→ 向量检索 (FAISS)
   │   ├─→ BM25 检索
   │   ├─→ 加权融合
   │   └─→ Cross-Encoder 重排
   │
   ├─→ [Grader] 评估证据充分性
   │   ├─→ 调用 LLM 评估
   │   └─→ 输出: is_sufficient, missing_info
   │
   ├─→ 条件路由
   │   ├─ 充分 → [Generator]
   │   └─ 不充分 → [Rewriter]
   │
   ├─→ [Rewriter] 重写查询（如需要）
   │   └─→ 回到 Retriever（闭环）
   │
   └─→ [Generator] 生成最终回答
       ├─→ 调用 LLM 生成
       ├─→ 术语校验
       └─→ 提取引用来源
   ↓
[Spring Boot] 接收结果 → 保存到数据库 → 返回给用户
   ↓
用户收到回答
```

---

## 二、关键设计决策

### 2.1 为什么采用 LangGraph？

| 对比项 | LangChain Agent | LangGraph | 选择 |
|:--|:--|:--|:--|
| 动态路由 | 有限 | 完全支持 | ✓ LangGraph |
| 闭环迭代 | 不支持 | 原生支持 | ✓ LangGraph |
| 状态管理 | 隐式 | 显式 TypedDict | ✓ LangGraph |
| 可调试性 | 一般 | 优秀 | ✓ LangGraph |
| 学习曲线 | 平缓 | 陡峭 | - |

**结论**：LangGraph 完全满足"检索→评估→重写"的闭环需求。

### 2.2 为什么混合检索？

```
单一向量检索的问题：
- 精确匹配能力弱（API 名称、参数名容易漏掉）
- 对低频词不敏感

单一 BM25 的问题：
- 无法理解语义（同义词、近义词无法匹配）
- 对长文本检索效果差

混合检索的优势：
- 向量检索（权重 0.7）：捕捉语义
- BM25 检索（权重 0.3）：精确匹配
- 加权融合：兼顾两者
- 实验证明 F1 从 49.8% → 70.9%
```

### 2.3 为什么需要 Grader Agent？

```
没有 Grader 的问题：
- 无法判断检索结果是否足够
- 可能生成不完整或错误的回答
- 浪费计算资源（不必要的检索）

Grader 的作用：
1. 质量门控：确保只有充分的证据才生成回答
2. 反馈驱动：输出"缺失信息"指导 Rewriter
3. 早停机制：避免无限迭代
4. 可解释性：用户能看到评估过程
```

### 2.4 迭代策略

```python
# 当前设计：固定 3 次迭代
# 改进方案：动态迭代

def should_continue_iteration(state):
    # 1. 检查充分性分数（0-1）
    if state["sufficiency_score"] > 0.85:
        return False  # 充分，停止
    
    # 2. 检查迭代次数
    if state["iterations"] >= 3:
        return False  # 达到上限，停止
    
    # 3. 检查缺失信息是否重复（陷入循环）
    if is_similar(state["missing_info"], 
                  state["prev_missing_info"], 
                  threshold=0.9):
        return False  # 陷入循环，停止
    
    return True  # 继续迭代
```

---

## 三、模块详细设计

### 3.1 Retriever 模块（检索与重排）

**输入**：query（查询文本）

**输出**：documents（排序后的文档列表）

**流程**：
```
query
  ↓
[向量检索] FAISS + Embedding
  ├─ 计算查询向量
  ├─ 余弦相似度搜索
  └─ 返回 Top-10
  ↓
[BM25 检索]
  ├─ 分词处理
  ├─ BM25 评分
  └─ 返回 Top-10
  ↓
[加权融合]
  ├─ 归一化两种分数
  ├─ 加权求和（0.7 * vector + 0.3 * bm25）
  └─ 返回 Top-10
  ↓
[Cross-Encoder 重排]
  ├─ 构建 (query, doc) 对
  ├─ 深度交互打分
  └─ 返回 Top-5
  ↓
documents (Top-5)
```

**关键参数**：
- 初筛 Top-K：10（平衡召回和速度）
- 最终 Top-K：5（平衡质量和成本）
- 向量权重：0.7（语义优先）
- BM25 权重：0.3（精确匹配辅助）

### 3.2 Grader 模块（证据评估）

**输入**：query（原始问题），documents（检索结果）

**输出**：is_sufficient（布尔值），missing_info（缺失信息描述）

**Prompt 设计**：
```
你是证据评估专家。

问题：{query}

文档内容：
{context}

请判断这些文档是否足以回答问题。

输出 JSON：
{
  "is_sufficient": true/false,
  "missing_info": "如果不充分，描述缺少什么信息"
}
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.0（确定性评估）
- Max tokens：256

### 3.3 Rewriter 模块（查询重写）

**输入**：original_query（原始问题），missing_info（缺失信息）

**输出**：new_query（重写后的查询）

**Prompt 设计**：
```
你是查询重写专家。

原始问题：{original_query}
缺失信息：{missing_info}

请生成一个更具体、指向性更强的新查询，用于检索缺失的信息。

新查询：
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.3（适度创造性）
- Max tokens：128

### 3.4 Generator 模块（回答生成）

**输入**：query（原始问题），documents（最终文档），reasoning_enabled（是否需要推理）

**输出**：answer（回答），reasoning（推理过程）

**Prompt 设计**：
```
你是技术文档智能助理。

文档内容：
{context}

问题：{query}

请按以下步骤回答：
1. 在 <thought> 标签内输出推理过程
2. 在标签外输出最终答案
3. 答案中应引用相关章节路径
4. 如果文档中没有相关信息，明确回复：未找到相关信息

<thought>
```

**LLM 参数**：
- 模型：Qwen2.5-7B-Instruct
- 温度：0.7（生成需要创造性）
- Max tokens：1024

---

## 四、状态管理

### 4.1 AgentState 定义

```python
class AgentState(TypedDict):
    # 输入
    query: str                      # 当前查询（可能被重写）
    original_query: str             # 原始问题
    user_id: str                    # 用户 ID
    conversation_id: str            # 会话 ID
    
    # 检索结果
    documents: List[Dict]           # 文档列表
    document_scores: List[float]    # 相关性分数
    
    # 评估结果
    is_sufficient: bool             # 证据是否充分
    missing_info: str               # 缺失信息
    sufficiency_score: float        # 充分性分数（0-1）
    
    # 迭代控制
    iterations: int                 # 当前迭代次数
    max_iterations: int             # 最大迭代次数
    prev_missing_info: str          # 上一次的缺失信息
    
    # 输出
    answer: str                     # 最终回答
    reasoning: Optional[str]        # 推理过程
    sources: List[str]              # 引用来源
    confidence: float               # 置信度（0-1）
    
    # 元数据
    start_time: float               # 开始时间
    retrieval_times: List[float]    # 每次检索耗时
    error: Optional[str]            # 错误信息
```

### 4.2 状态转移图

```
初始化状态
    ↓
[retrieve_rerank]
    ↓
[grade]
    ↓
条件判断
├─ is_sufficient=True → [generate] → END
├─ iterations >= 3 → [generate] → END
└─ is_sufficient=False → [rewrite] → [retrieve_rerank]（循环）
```

---

## 五、缓存策略

### 5.1 三层缓存架构

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

### 5.2 缓存键设计

```python
# 查询缓存键
query_cache_key = f"query:{hash(query)}:{user_id}"

# 向量缓存键
vector_cache_key = f"vector:{hash(query)}"

# 文档缓存键
doc_cache_key = f"doc:{doc_id}:{version}"
```

### 5.3 缓存失效策略

```
文档更新时：
- 清除所有相关查询缓存
- 清除向量缓存
- 重建 FAISS 索引

术语词典更新时：
- 清除所有查询缓存（因为生成结果会变化）

定期清理：
- 每天凌晨 2 点清理过期缓存
```

---

## 六、错误处理与降级

### 6.1 故障场景与应对

| 故障场景 | 影响 | 降级方案 |
|:--|:--|:--|
| LLM 服务不可用 | 无法评估、重写、生成 | 直接返回检索结果摘要 |
| 向量库不可用 | 无法进行向量检索 | 仅使用 BM25 检索 |
| Redis 不可用 | 缓存失效 | 继续运行，性能下降 |
| 数据库不可用 | 无法保存反馈 | 继续运行，反馈丢失 |

### 6.2 降级实现

```python
def retrieve_with_fallback(query):
    try:
        # 尝试混合检索
        return hybrid_retrieve(query)
    except VectorStoreError:
        # 降级到 BM25
        logger.warning("Vector store unavailable, using BM25 only")
        return bm25_retrieve(query)
    except Exception as e:
        # 最后的降级：返回空结果
        logger.error(f"Retrieval failed: {e}")
        return []

def generate_with_fallback(query, documents):
    try:
        # 尝试完整流程
        return generate_answer(query, documents)
    except LLMError:
        # 降级：返回文档摘要
        logger.warning("LLM unavailable, returning document summary")
        return summarize_documents(documents)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "服务暂时不可用，请稍后重试"
```

---

## 七、性能优化

### 7.1 关键性能指标（KPI）

| 指标 | 目标 | 说明 |
|:--|:--|:--|
| 平均响应时间 | < 2.5s | P95 < 4s |
| 吞吐量 | > 100 req/s | 单机 |
| 缓存命中率 | > 60% | L1+L2 |
| 可用性 | > 99.5% | 月度 |
| EM 准确率 | > 60% | 测试集 |

### 7.2 优化方向

```
1. 向量检索优化
   - 使用 FAISS GPU 加速
   - 批量查询向量
   - 预计算热点查询

2. LLM 推理优化
   - 使用 vLLM PagedAttention
   - 批量推理
   - 量化模型（可选）

3. 缓存优化
   - 智能预热（基于历史查询）
   - 分布式缓存（Redis Cluster）

4. 并发优化
   - 异步处理
   - 请求队列管理
   - 连接池复用
```

---

## 八、监控与可观测性

### 8.1 关键指标

```
系统级指标：
- 请求延迟（P50/P95/P99）
- 错误率
- 缓存命中率
- 并发连接数

业务级指标：
- 用户满意度（点赞率）
- 回答准确率（EM）
- 平均迭代次数
- 文档覆盖率

资源级指标：
- CPU 使用率
- 内存使用率
- GPU 使用率
- 磁盘 I/O
```

### 8.2 日志设计

```
每个请求记录：
- request_id：唯一标识
- user_id：用户
- query：查询内容
- conversation_id：会话
- iterations：迭代次数
- response_time_ms：响应时间
- cache_hit：是否命中缓存
- error：错误信息（如有）
```

---

## 九、部署架构

### 9.1 开发环境

```
本地开发：
- Spring Boot：localhost:8080
- FastAPI：localhost:8000
- MySQL：localhost:3306
- Redis：localhost:6379
- FAISS：本地文件
```

### 9.2 生产环境

```
高可用部署：
- Spring Boot：多实例 + 负载均衡
- FastAPI：多实例 + 负载均衡
- MySQL：主从复制 + 读写分离
- Redis：Cluster 模式
- FAISS：分布式索引
- vLLM：多 GPU 推理
```

---

## 十、实现优先级

### Phase 1：核心功能（必须）
- [ ] LangGraph 状态图搭建
- [ ] Retriever 模块（混合检索）
- [ ] Grader 模块（证据评估）
- [ ] Rewriter 模块（查询重写）
- [ ] Generator 模块（回答生成）
- [ ] FastAPI 服务封装

### Phase 2：质量保证（重要）
- [ ] 术语词典与校验
- [ ] 结构化文档解析
- [ ] 可解释性输出
- [ ] 基础监控

### Phase 3：性能优化（推荐）
- [ ] 缓存机制
- [ ] 向量预计算
- [ ] 并发优化
- [ ] 降级方案

### Phase 4：用户体验（长期）
- [ ] 用户反馈机制
- [ ] 知识库管理界面
- [ ] 查询历史
- [ ] 个性化推荐

---

**文档结束**
