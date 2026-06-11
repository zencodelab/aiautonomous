# 🤖 TaskEngine — Autonomous Task Execution Engine

An autonomous task-execution engine that transforms **plain-text queries** into **structured, multi-step operations** using LLM reasoning, vector-based knowledge retrieval (Pinecone), and the **Plan → Execute → Reflect** pattern built on **LangChain + LangGraph**.

## ✨ Features

- **Intelligent Task Decomposition** — LLM-powered planner breaks complex queries into atomic, ordered steps
- **Autonomous Execution** — ReAct agents execute each step using a rich set of tools
- **Quality Reflection** — Built-in reflector evaluates results and triggers re-planning if needed
- **Knowledge-Augmented** — Pinecone vector store provides domain knowledge and past execution context
- **Multiple Interfaces** — CLI, REST API, and Python SDK
- **Tool Suite** — Web search, math computation, code execution, file operations, knowledge retrieval

## 🏗️ Architecture

```
User Query (plain text)
    │
    ▼
┌─────────┐     ┌─────────────────┐     ┌─────────────┐
│ Planner │────▶│ Step Executor   │────▶│  Reflector  │
│ (LLM)   │     │ (ReAct Agent)   │     │  (Validate) │
└─────────┘     └────────┬────────┘     └──────┬──────┘
    ▲                    │                     │
    │                    ▼                     │
    │           ┌────────────────┐             │
    └───────────│ Knowledge Store│◀────────────┘
   re-plan      │  (Pinecone)   │   store results
                └────────────────┘
```

### Design Pattern: Plan → Execute → Reflect → (Re-plan)

| Component | Framework | Role |
|---|---|---|
| **Planner** | LangChain (`ChatOpenAI` + `langchain_core`) | Decomposes query into atomic steps with tool hints; uses Pinecone few-shot context from past runs |
| **Executor** | LangGraph (`langgraph.prebuilt.create_react_agent`) | Runs each step as a ReAct agent with tool calling and timeout |
| **Reflector** | LangChain (`ChatOpenAI`) | Scores output quality 0.0–1.0; triggers re-planning if below threshold |
| **Knowledge Store** | `langchain-pinecone` | Stores domain knowledge and past execution results for retrieval |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://app.pinecone.io/)

### Installation

```bash
# Clone the repository
git clone https://github.com/zencodelab/aiautonomous.git
cd aiautonomous

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Usage

#### CLI

```bash
# Execute a task
python cli.py run "Research the latest Python 3.13 features and create a summary document"

# Generate a plan (dry run)
python cli.py plan "Calculate compound interest on $10,000 at 5% for 10 years"

# Add knowledge to the vector store
python cli.py knowledge add path/to/document.txt

# Start the REST API server
python cli.py serve
```

#### Python SDK

```python
import asyncio
from taskengine.config import get_settings
from taskengine.engine import TaskEngine

async def main():
    engine = TaskEngine(get_settings())
    report = await engine.run("Your task description here")
    print(f"Success: {report.success}")
    print(f"Summary: {report.summary}")

asyncio.run(main())
```

#### REST API

```bash
# Start the server
python cli.py serve

# Submit a task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 2^10 and explain the result"}'

# Check status
curl http://localhost:8000/tasks/{task_id}

# Add knowledge
curl -X POST http://localhost:8000/knowledge \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Important domain knowledge..."]}'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_planner.py -v

# Run with coverage
pytest tests/ -v --cov=taskengine
```

## 📁 Project Structure

```
aiautonomous/
├── pyproject.toml              # Project config & dependencies
├── .env.example                # Environment variable template
├── cli.py                      # CLI entry point (Typer + Rich)
├── src/taskengine/
│   ├── config.py               # Centralized settings
│   ├── models.py               # Pydantic data models
│   ├── planner.py              # Task decomposition (LLM)
│   ├── executor.py             # Step execution (ReAct agent)
│   ├── reflector.py            # Quality validation
│   ├── engine.py               # Main orchestrator
│   ├── vectorstore.py          # Pinecone integration
│   ├── tools/
│   │   ├── web_search.py       # DuckDuckGo search
│   │   ├── file_ops.py         # Sandboxed file I/O
│   │   ├── code_executor.py    # Python code runner
│   │   ├── math_tool.py        # Safe math evaluator
│   │   └── knowledge.py        # Knowledge retrieval
│   └── api/
│       └── server.py           # FastAPI REST server
└── tests/
    ├── test_planner.py
    ├── test_executor.py
    ├── test_engine.py
    └── test_vectorstore.py
```

## ⚙️ Configuration

All settings are loaded from environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL_NAME` | `gpt-4o` | LLM model name |
| `OPENAI_TEMPERATURE` | `0.1` | Sampling temperature |
| `PINECONE_API_KEY` | (required) | Pinecone API key |
| `PINECONE_INDEX_NAME` | `taskengine-knowledge` | Index name |
| `PINECONE_CLOUD` | `aws` | Cloud provider |
| `PINECONE_REGION` | `us-east-1` | Region |
| `MAX_STEPS` | `15` | Max steps per plan |
| `MAX_REPLANS` | `2` | Max re-planning cycles |
| `STEP_TIMEOUT_SECONDS` | `120` | Per-step timeout |

## 📄 License

MIT
