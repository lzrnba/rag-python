# RAG 系统运行逻辑详解

> 文档生成时间：2026-03-19

---

## 一、系统总体架构

```
用户请求
    │
    ▼
 FastAPI 接口层 (/v1/chat/completions)
    │
    ▼
 LangGraph 状态图（四节点协作）
    │
    ├── retrieve_rerank  检索节点
    ├── grade            评估节点
    ├── rewrite          重写节点
    └── generate         生成节点
    │
    ▼
 返回结构化响应（答案 + 推理过程 + 引用来源 + 置信度）
```

---

## 二、请求入口（API 层）

**文件**：`api/routes/chat.py`

用户通过 `POST /v1/chat/completions` 发送请求，携带：

| 字段 | 说明 |
|:--|:--|
| `query` | 用户问题 |
| `user_id` | 用户 ID |
| `conversation_id` | 会话 ID（可选，自动生成） |
| `options.max_iterations` | 最大迭代次数（默认 3） |
| `options.top_k` | 检索文档数量（默认 5） |

接口收到请求后，构建 **初始状态（AgentState）** 并调用 LangGraph 状态图。

---

## 三、LangGraph 状态图流程

**文件**：`agents/graph.py`

```
retrieve_rerank
      │
      ▼
    grade ──── sufficiency_score > 0.85 ──────────────────┐
      │                                                    │
      │── 达到最大迭代次数 ────────────────────────────────│
      │                                                    │
      │── 缺失信息重复（循环检测）────────────────────────│
      │                                                    ▼
    rewrite                                           generate
      │                                                    │
      └──────────── retrieve_rerank ◄───────────────────┘ │
                                                          ▼
                                                         END
```

### 路由规则（route_based_on_sufficiency）

优先级从高到低：

1. **充分性分数 > 0.85** → 直接生成答案
2. **迭代次数 ≥ max_iterations** → 强制生成（防止超时）
3. **缺失信息与上次相同** → 强制生成（防止死循环）
4. **其他** → 重写查询，再次检索

---

## 四、四个节点详解

**文件**：`agents/nodes.py`

---

### 节点 1：retrieve_rerank（检索节点）

**职责**：根据当前查询从文档库中检索相关文档

**执行步骤**：

```
输入：state["query"]（当前查询，可能已被重写）
  │
  ▼
调用 HybridRetriever.search(query, k=top_k)
  │
  ▼
更新状态：
  - state["documents"]        = 检索到的文档列表
  - state["document_scores"]  = 各文档得分
  - state["iterations"]       += 1
  - state["retrieval_times"]  追加本次耗时
```

---

### 节点 2：grade（评估节点）

**职责**：判断当前检索到的文档是否足以回答问题

**执行步骤**：

```
输入：state["documents"] + state["original_query"]
  │
  ├── LLM 不可用 → 强制 sufficiency_score=1.0，跳过评估
  │
  ├── 无文档 → sufficiency_score=0.0，missing_info="未检索到相关文档"
  │
  └── 正常流程：
        构建 GRADER_PROMPT（原始问题 + 上下文）
          │
          ▼
        调用 LLM（temperature=0.0，确定性输出）
          │
          ▼
        解析 JSON 响应：
          {
            "is_sufficient": true/false,
            "missing_info": "缺失的信息描述",
            "sufficiency_score": 0.0~1.0
          }
          │
          ▼
        更新状态：is_sufficient / missing_info / sufficiency_score
```

---

### 节点 3：rewrite（重写节点）

**职责**：基于缺失信息生成新的、指向性更强的查询

**执行步骤**：

```
输入：state["original_query"] + state["missing_info"]
  │
  ├── LLM 不可用 → 保持原始查询不变（防止死循环）
  │
  └── 正常流程：
        构建 REWRITER_PROMPT
          │
          ▼
        调用 LLM（temperature=0.3）
          │
          ▼
        更新状态：
          - state["prev_missing_info"] = 上次缺失信息（用于循环检测）
          - state["query"]             = 新查询
```

---

### 节点 4：generate（生成节点）

**职责**：基于确认充分的文档，使用 CoT 思维链生成最终答案

**执行步骤**：

```
输入：state["documents"] + state["original_query"]
  │
  ├── LLM 不可用 → 直接拼接文档内容作为答案
  │
  ├── 无文档 → 返回"未找到相关信息"
  │
  └── 正常流程：
        构建 GENERATOR_PROMPT（文档上下文 + 问题）
          │
          ▼
        调用 LLM（temperature=0.7，max_tokens=1024）
          │
          ▼
        解析 CoT 格式：
          <thought>推理过程...</thought>
          最终答案...
          │
          ▼
        计算置信度：
          confidence = 0.6（基础）
                     + sufficiency_score × 0.2
                     + min(doc数量/3, 1.0) × 0.2
          （最大值 1.0）
          │
          ▼
        更新状态：answer / reasoning / sources / confidence
```

---

## 五、混合检索器详解

**文件**：`retrieval/hybrid.py`

### 当前实现（BM25 单路检索）

```
输入：query 文本
  │
  ▼
BM25 分词检索
  - 对所有文档内容按空格分词
  - 使用 BM25Okapi 算法计算相关性分数
  │
  ▼
取 Top-K×2 候选文档
  │
  ▼
归一化分数（Min-Max 归一化到 0~1）
  │
  ▼
返回 Top-K 文档（按 final_score 降序）
```

### 计划中的混合检索（向量 + BM25）

```
向量检索（FAISS）          BM25 检索
    │                          │
    └──────── 融合分数 ────────┘
              │
    vector_weight × 向量分数
  + bm25_weight   × BM25分数
              │
    Cross-Encoder 重排（待实现）
              │
           Top-K 文档
```

**配置参数**（`core/config.py`）：
- `VECTOR_WEIGHT`：向量检索权重
- `BM25_WEIGHT`：BM25 检索权重
- `TOP_K`：返回文档数量

---

## 六、完整请求生命周期

```
① 用户发送问题
       │
② API 层构建 AgentState 初始状态
       │
③ LangGraph 启动，进入 retrieve_rerank 节点
       │
④ HybridRetriever 执行 BM25 检索，返回 Top-K 文档
       │
⑤ 进入 grade 节点，LLM 评估文档充分性
       │
       ├── 充分（score > 0.85）→ 进入 ⑧
       │
       └── 不充分 → 进入 ⑥
             │
⑥ 进入 rewrite 节点，LLM 重写查询
       │
⑦ 回到 retrieve_rerank，用新查询再次检索 → 回到 ⑤
   （最多循环 max_iterations 次）
       │
⑧ 进入 generate 节点，LLM 生成 CoT 答案
       │
⑨ API 层组装响应，返回给用户：
   {
     answer:           最终答案
     reasoning:        推理过程（<thought> 内容）
     sources:          引用文档路径列表
     confidence:       置信度 0~1
     retrieval_process: 迭代次数、文档数量、耗时等
     response_time_ms: 总耗时
   }
```

---

## 七、状态字段说明

**文件**：`agents/state.py`

| 字段 | 类型 | 说明 |
|:--|:--|:--|
| `query` | str | 当前查询（可能被重写） |
| `original_query` | str | 用户原始问题 |
| `documents` | List[dict] | 检索到的文档列表 |
| `is_sufficient` | bool | 文档是否充分 |
| `missing_info` | str | 缺失信息描述 |
| `sufficiency_score` | float | 充分性分数 0~1 |
| `iterations` | int | 当前迭代次数 |
| `max_iterations` | int | 最大迭代次数 |
| `prev_missing_info` | str | 上次缺失信息（循环检测用） |
| `answer` | str | 最终答案 |
| `reasoning` | str | CoT 推理过程 |
| `sources` | List[str] | 引用来源列表 |
| `confidence` | float | 置信度 0~1 |
| `top_k` | int | 检索文档数量 |

---

## 八、当前局限与后续计划

### 当前局限

| 问题 | 原因 |
|:--|:--|
| 只有 BM25，无向量检索 | FAISS 向量库尚未集成 |
| 中文分词效果差 | BM25 按空格分词，不适合中文 |
| 文档加载只支持 .txt/.md | PDF/Word 解析未实现 |
| 无多轮对话上下文 | 每次请求独立，不保存历史 |

### 后续计划（Phase 2）

- 接入 FAISS 向量检索，使用 `bge-m3` 做 Embedding
- 引入 jieba/结巴分词优化中文 BM25
- 支持 PDF/Word 文档解析
- 添加 Cross-Encoder 重排
- 实现多轮对话记忆
