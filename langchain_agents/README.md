# LangChain-Based Agents 🔗

[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

This directory contains agent implementations built using the LangChain framework. Each agent is designed for specific use cases while maintaining the project's core principles of modularity and extensibility.

## 📦 Available Agents

### [RAG Agent](rag_agent/README.md)
Implements Retrieval-Augmented Generation for document Q&A:
- PDF processing and text extraction
- Vector storage with ChromaDB
- Multiple LLM provider support

## 📁 Directory Structure

```
langchain_agents/
├── README.md              # This file
└── rag_agent/            # RAG implementation
    ├── README.md         # RAG documentation
    ├── __init__.py       # Package exports
    ├── main.py          # Entry point
    ├── requirements.txt  # RAG dependencies
    ├── agents/          # Agent implementations
    │   ├── embedding_agent.py
    │   ├── logger_agent.py
    │   ├── pdf_extractor_agent.py
    │   ├── processing_agent.py
    │   ├── rag_agent.py
    │   └── vectordb_agent.py
    └── llm/            # LLM interface
        └── llm_caller.py

```

## 🔄 Common Components

### LLM Wrappers
Shared LLM provider implementations in [`/wrappers`](../wrappers/):
- [`openai_wrapper.py`](../wrappers/openai_wrapper.py)
- [`groq_wrapper.py`](../wrappers/groq_wrapper.py)
- [`huggingface_wrapper.py`](../wrappers/huggingface_wrapper.py)
- [`gemini_wrapper.py`](../wrappers/gemini_wrapper.py)
- [`llama_wrapper.py`](../wrappers/llama_wrapper.py)

### Requirements
Agent-specific requirements are in their respective directories, automatically included in the [root requirements.txt](../requirements.txt) through [`update_root_requirements.py`](../update_root_requirements.py).

## 🛠️ Development

### Adding a New LangChain Agent

1. Create a new directory:
```
langchain_agents/
└── your_agent_name/
    ├── __init__.py         # Package exports
    ├── main.py            # Entry point
    ├── README.md          # Agent documentation
    ├── requirements.txt    # Agent dependencies
    ├── agents/            # Agent components
    └── config/            # Configuration
```

2. Follow [LangChain's best practices](https://python.langchain.com/docs/guides/practices) for:
- Chain composition
- Prompt management
- Tool integration
- Memory handling

3. Use existing wrappers or add new ones in the root [`/wrappers`](../wrappers/) directory

## 📚 Resources

- [Project Documentation](../README.md)
- [LangChain Documentation](https://python.langchain.com/)