面向财报分析与股票研究场景的金融研究智能体系统，集成财报文档检索（RAG）、实时行情查询（Yahoo Finance MCP）和多轮会话状态管理，支持历史财务分析、实时行情查询以及混合问题联合回答。

## Overview

这个项目主要解决三类问题：

- 财报类问题：从 10-K、10-Q、财务报表等历史文档中检索证据
- 行情类问题：获取最新股价、历史走势、估值、新闻等市场数据
- 混合类问题：当用户同时问“财报表现 + 最近股价走势”时，自动拆解并联合分析

它不是一个单纯“让大模型自由调用工具”的 Demo，而是一个带有明确分层架构的金融研究 Agent：

- 查询理解与任务路由
- 财报文档处理与 RAG 检索
- 实时行情工具调用
- 多轮会话状态与记忆管理

## Features

- 文档 RAG：对 10-K/10-Q 等金融文档进行结构化处理、向量检索和重排
- 实时行情查询：通过 Yahoo Finance MCP 获取报价、历史走势、新闻和分析师信息
- 混合问题联动：自动拆解文档子问题与市场子问题，再统一综合输出
- 多轮上下文连续性：支持主公司、年份、时间区间、对比对象的跨轮继承
- 稳定性机制：支持子进程隔离、超时控制、熔断和备用数据源降级
- 回归验证：内置 hybrid 路由、状态继承、记忆一致性等专项检查脚本

## Architecture

```text
User Query
   |
   v
agent.py
   |
   v
dialog_state_manager.py ---- memory_manager.py
   |                               |
   v                               v
query_preprocessor.py ----> intent_router.py
   |                               |
   |                    +----------+----------+
   |                    |                     |
   v                    v                     v
historical_rag      live_market            hybrid
   |                    |                     |
   v                    v                     v
retrieval.py        mcp_tools.py       agent_app.py/_run_hybrid_workflow
   |                    |                     |
   v                    v                     v
vector_store.py     MCP / yfinance      synthesis by LLM
   |
   v
Qdrant
```

### 模块说明

- `agent.py`：CLI 入口，负责线程 ID、会话重置和调用编排
- `agent_app.py`：组装 Agent、执行流式输出、处理 hybrid 显式工作流
- `query_preprocessor.py`：提取公司名、Ticker、年份、季度、时间区间，并做中英术语映射
- `intent_router.py`：判断问题属于文档、行情、混合还是澄清类
- `retrieval.py`：实现财报检索、多 query 检索、rerank 与证据输出
- `vector_store.py`：处理 Qdrant 连接、文档入库、元数据组织
- `pdf_pipeline.py`：把 PDF 解析为 Markdown、表格和图片描述
- `mcp_tools.py`：封装 Yahoo Finance MCP 调用，并在失败时降级到 `yfinance`
- `dialog_state_manager.py`：管理多轮会话中的主公司、年份、比较状态等结构化状态
- `memory_manager.py`：压缩长会话历史，保留最近若干轮与摘要信息
- `sandbox_scheduler.py`：对 MCP 调用做子进程隔离、超时、熔断和并发限制

## Repository Structure

```text
jinrong_agent/
├── agent.py
├── agent_app.py
├── config.py
├── dialog_state_manager.py
├── intent_router.py
├── mcp_tools.py
├── mcp_worker.py
├── memory_manager.py
├── model_factory.py
├── pdf_pipeline.py
├── query_preprocessor.py
├── retrieval.py
├── runtime_checks.py
├── sandbox_scheduler.py
├── vector_store.py
├── eval_*.py
├── data/
│   ├── rag_data/
│   ├── adaptive_memory/
│   └── dialog_state/
├── output/
└── qdrant_data/
```

## Tech Stack

- Python 3.10+
- LangChain / LangGraph
- Qdrant
- Docling
- MCP (Yahoo Finance)
- yfinance
- sentence-transformers / Cross-Encoder

依赖见 [`requirements.txt`](./requirements.txt)。

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

复制环境变量模板：

```bash
cp .env.example .env
```

至少配置一个模型供应商：

- Google：`GOOGLE_API_KEY`
- OpenAI：`OPENAI_API_KEY`
- Qwen / DashScope：`DASHSCOPE_API_KEY`

模型切换由 `MODEL_PROVIDER` 控制，见 [`model_factory.py`](./model_factory.py)。

### 3. Start Qdrant

```bash
docker compose up -d
```

默认地址为 `http://localhost:6333`，配置见 [`config.py`](./config.py)。

## Data Preparation

### 1. Put PDFs Into `data/rag_data/`

将原始财报 PDF 放到 `data/rag_data/` 目录下。

### 2. Parse PDF Into Structured Outputs

```bash
python -c "from pdf_pipeline import process_all_pdfs; process_all_pdfs('data/rag_data')"
```

解析产物会输出到 `output/<company>/`，包括：

- `markdown/`
- `tables/`
- `images/`
- `images_desc/`

### 3. Ingest Parsed Data Into Qdrant

```bash
python -c "from vector_store import ingest_all; ingest_all('output')"
```

## Usage

### Basic Query

```bash
python agent.py --query "分析一下特斯拉最近的股价趋势，并给出投资建议"
```

### Multi-turn Query

```bash
python agent.py --thread-id demo --query "分析 Apple 2024 年营收和利润"
python agent.py --thread-id demo --query "那最近 1 个月股价走势呢"
```

### Reset Session State

```bash
python agent.py --thread-id demo --reset-session --query "分析 Apple 2024 年营收"
```

### Run Without Memory

```bash
python agent.py --no-memory --query "分析 Apple 2024 年营收"
```

## Example Queries

- `分析 Apple 2024 年营收、利润率和现金流`
- `TSLA 最近 3 个月股价走势怎么样`
- `英伟达财报表现和最近股价走势哪个更重要`
- `把 Apple 和 Microsoft 的盈利能力对比一下`
- `那 2023 年呢`

## Evaluation

运行专项检查脚本：

```bash
python eval_hybrid.py
python eval_dialog_state.py
python eval_memory_consistency.py
python eval_market_params.py
python eval_session_controls.py
```

也可以运行一键运行时检查：

```bash
python validate_runtime.py
```

## Design Notes

### 为什么 hybrid 问题单独编排

当一个问题同时涉及财报和行情时，如果完全交给模型自由决定调用路径，容易出现：

- 漏调某一侧工具
- 财报证据与实时行情证据混用
- 结论偏向单一数据源

因此本项目对高风险的混合问题采用显式工作流：先拆解子问题，再分别调用文档检索和行情工具，最后让模型做综合总结。

### 为什么多轮上下文分成“记忆”和“状态”

- `memory`：解决长会话 token 不够的问题，压缩历史信息
- `dialog_state`：解决当前轮任务主语、年份、比较对象等强结构信息继承问题

两者分离可以降低长会话中旧上下文污染当前任务的风险。

## FAQ

### 1. Qdrant 连接失败怎么办？

- 确认已执行 `docker compose up -d`
- 检查 `http://localhost:6333` 是否可访问
- 检查 [`config.py`](./config.py) 中的 Qdrant 配置

### 2. 模型报缺少环境变量怎么办？

- 确认 `.env` 中配置了与 `MODEL_PROVIDER` 对应的 Key
- 重新激活虚拟环境并重新运行

### 3. MCP 不可用怎么办？

- 项目会自动降级到 `yfinance`
- 如果你希望优先走 MCP，确认系统存在 `uvx` 命令，并能启动 Yahoo Finance MCP Server







