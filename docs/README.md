# 项目文档索引

## 目录结构

```
docs/
├── phase0/   项目总览与设计
├── phase1/   第一阶段：基础框架
├── phase2/   第二阶段：检索增强
└── README.md
```

## Phase 0 - 项目总览

| 文档 | 说明 |
|:--|:--|
| phase0/PROJECT_OVERVIEW.md | 项目背景、目标、技术选型 |
| phase0/ARCHITECTURE_DESIGN.md | 整体架构、模块划分 |
| phase0/API_DESIGN.md | 接口规范 |
| phase0/DEPLOY.md | 部署说明 |
| phase0/README.md | 快速开始 |

## Phase 1 - 基础框架

| 文档 | 说明 |
|:--|:--|
| phase1/PHASE1_SUMMARY.md | 第一阶段总结 |
| phase1/IMPLEMENTATION_COMPLETE.md | 实现完成报告 |
| phase1/RAG_LOGIC.md | RAG 运行逻辑详解 |

**实现内容：** FastAPI + LangGraph 四节点 + BM25 检索 + Ollama qwen2.5:7b + 前端界面

## Phase 2 - 检索增强（进行中）

| 文档 | 说明 |
|:--|:--|
| phase2/PHASE2_PLAN.md | 第二阶段规划 |

**实现内容：** FAISS 向量检索 + bge-m3 Embedding + jieba 分词 + 文档解析分块 + 混合检索融合

## 快速启动

```powershell
conda activate chatglm3
cd "e:\Project Design\project\python-rag\rag-python"
python main.py
```
