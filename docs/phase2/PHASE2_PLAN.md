# Phase 2 规划文档 - 检索增强实现

> 状态：🚧 进行中
> 基于 Phase 1 基础框架，实现真正的混合检索能力

---

## 一、Phase 2 目标

| 目标 | 说明 |
|:--|:--|
| 向量检索 | 接入 FAISS + bge-m3，实现语义检索 |
| 中文优化 | jieba 分词替换空格分词，提升 BM25 中文效果 |
| 文档解析 | 支持 PDF / Word / MD / TXT 文档解析 |
| 文档分块 | 实现滑动窗口分块，避免长文档截断 |
| 混合融合 | 向量（0.7）+ BM25（0.3）加权融合 |
| 索引重建 | 文档重载接口支持向量索引自动重建 |

---

## 二、任务分解

### Task 1：安装新依赖

```
faiss-cpu          向量索引
jieba              中文分词
python-docx        Word 文档解析
pypdf              PDF 解析
langchain-text-splitters  文档分块
```

### Task 2：Embedding 接入（retrieval/embedder.py）

- 使用 Ollama bge-m3 模型生成向量
- 调用 `http://localhost:11434/v1/embeddings`
- 支持批量 Embedding
- 缓存已计算的向量（避免重复计算）

### Task 3：文档解析与分块（retrieval/loader.py）

- 支持格式：`.txt` `.md` `.pdf` `.docx`
- 分块策略：滑动窗口
  - chunk_size = 500 字符
  - chunk_overlap = 50 字符
- 每个 chunk 保留来源元数据

### Task 4：FAISS 向量库（retrieval/vector_store.py）

- 构建 FAISS IndexFlatIP（内积相似度）
- 支持持久化（保存/加载 index 文件）
- 支持增量添加文档
- 存储路径：`data/vectorstore/`

### Task 5：混合检索器升级（retrieval/hybrid.py）

**当前：** 仅 BM25，空格分词

**升级后：**
```
输入 query
  │
  ├── jieba 分词 → BM25 检索 → 归一化分数
  │
  └── bge-m3 Embedding → FAISS 检索 → 归一化分数
  │
  └── 加权融合：final = 0.7×向量分 + 0.3×BM25分
  │
  └── Top-K 文档
```

### Task 6：文档重载接口升级（api/routes/documents.py）

- 解析文档 → 分块 → Embedding → 重建 FAISS 索引
- 支持增量加载（只处理新增文档）
- 返回加载状态和 chunk 数量

---

## 三、实现方案

### 3.1 目录结构变化

```
retrieval/
  hybrid.py        ← 升级：混合检索融合
  embedder.py      ← 新增：Embedding 封装
  vector_store.py  ← 新增：FAISS 向量库
  loader.py        ← 新增：文档解析与分块

data/
  documents/       ← 用户文档目录（已有）
  vectorstore/     ← 新增：FAISS 索引存储
    index.faiss
    index.pkl
```

### 3.2 Embedding 模块设计

```python
class OllamaEmbedder:
    def __init__(self, model='bge-m3:latest', base_url='http://localhost:11434')
    def embed(self, texts: List[str]) -> np.ndarray  # shape: (n, dim)
    def embed_one(self, text: str) -> np.ndarray
```

### 3.3 文档加载器设计

```python
class DocumentLoader:
    def load_file(self, path: str) -> List[Dict]   # 加载单个文件
    def load_dir(self, dir: str) -> List[Dict]     # 加载目录
    def chunk(self, docs: List[Dict]) -> List[Dict] # 分块
```

### 3.4 向量库设计

```python
class FAISSVectorStore:
    def build(self, chunks: List[Dict], embedder: OllamaEmbedder)
    def search(self, query: str, k: int) -> List[Dict]
    def save(self, path: str)
    def load(self, path: str)
```

### 3.5 混合检索器升级

```python
class HybridRetriever:
    def initialize(self, documents: List[Dict])
      # 1. 文档分块
      # 2. 构建 BM25（jieba 分词）
      # 3. 构建 FAISS 索引

    def search(self, query: str, k: int) -> List[Dict]
      # 1. BM25 检索（jieba 分词 query）
      # 2. 向量检索（bge-m3 embed query）
      # 3. 分数归一化
      # 4. 加权融合
      # 5. 返回 Top-K
```

---

## 四、实施顺序

```
步骤 1：安装依赖，更新 requirements.txt
步骤 2：实现 retrieval/embedder.py
步骤 3：实现 retrieval/loader.py
步骤 4：实现 retrieval/vector_store.py
步骤 5：升级 retrieval/hybrid.py
步骤 6：升级 api/routes/documents.py
步骤 7：更新 main.py 使用新的加载流程
步骤 8：测试验证
```

---

## 五、验收标准

| 测试场景 | 期望结果 |
|:--|:--|
| 中文问题检索 | jieba 分词后 BM25 召回率提升 |
| 语义相似问题 | 向量检索能找到语义相关文档 |
| 加载 MD 文档 | 自动分块，正确建立索引 |
| 加载 PDF 文档 | 解析文本，正确建立索引 |
| 重载文档接口 | 新增文档后调用接口立即生效 |
| 向量库持久化 | 重启服务后索引仍然有效 |

---

## 六、风险与应对

| 风险 | 应对 |
|:--|:--|
| bge-m3 Embedding 慢 | 批量处理 + 持久化缓存 |
| PDF 解析乱码 | 优先 pypdf，失败降级到文本提取 |
| FAISS 内存占用大 | 使用 IndexFlatIP（精确但内存友好） |
| Ollama Embedding 接口超时 | 设置 120s 超时，分批处理 |
