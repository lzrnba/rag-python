# 快速开始

基于 LangGraph 和 Qwen2.5-7B 的技术文档智能问答系统。

## 环境要求

- Python 3.10+
- conda 环境：chatglm3
- Ollama（本地 LLM 推理）

## 快速启动

```powershell
# 1. 激活环境
conda activate chatglm3

# 2. 安装依赖
pip install -r requirements.txt

# 3. 确认 Ollama 已启动，拉取模型
ollama pull qwen2.5:7b
ollama pull bge-m3

# 4. 启动服务
cd "e:\Project Design\project\python-rag\rag-python"
python main.py

# 5. 打开前端
Start-Process frontend/index.html
```

## 访问地址

| 地址 | 说明 |
|:--|:--|
| `frontend/index.html` | 前端聊天界面 |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/health` | 服务健康检查 |

## 文档目录

详见 [docs/README.md](docs/README.md)
