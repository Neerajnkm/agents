# RAG Agent System 🤖 

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-yellow.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain that processes PDF documents and provides intelligent question-answering capabilities using various LLM providers.

## 📑 Table of Contents

1. [Features](#-features)
2. [Architecture](#-architecture)
3. [Prerequisites](#-prerequisites)
4. [System Requirements](#-system-requirements)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Project Structure](#-project-structure)
8. [Contributing](#-contributing)
9. [License](#-license)
10. [Support](#-support)

## 🌟 Features

- 📄 PDF document extraction and processing
- 🔄 Modular agent-based architecture
- 💾 Vector database storage using ChromaDB
- 🤖 Multiple LLM provider support (OpenAI, Gemini, Groq, LLaMA, HuggingFace)
- 🔍 Intelligent document retrieval and question answering
- 📝 Comprehensive logging system

## 🏗️ Architecture

The system is built using a modular agent-based architecture:

- **PDF Extractor Agent**: Handles document extraction
- **Processing Agent**: Text processing and cleaning
- **Embedding Agent**: Generates embeddings for text chunks
- **VectorDB Agent**: Manages ChromaDB interactions
- **RAG Agent**: Orchestrates question-answering process
- **Logger Agent**: Provides system-wide logging

## 🔧 Prerequisites

### Python Environment
- Python 3.12 or higher
- Virtual environment (recommended)

### System Dependencies
- Tesseract OCR for text extraction
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# CentOS/RHEL
sudo yum install -y tesseract

# macOS
brew install tesseract
```

- Poppler utils for PDF processing
```bash
# Ubuntu/Debian
sudo apt-get install -y poppler-utils

# CentOS/RHEL
sudo yum install -y poppler-utils

# macOS
brew install poppler
```

## 💻 System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: 1GB free space minimum
- **GPU**: Optional but recommended for faster processing
- **Internet**: Required for LLM API calls

## 🚀 Installation

### 1. Navigate to the agent directory
```bash
cd langchain_agents/rag_agent
```

### 2. Install dependencies
```bash
# Install Python packages from the root requirements.txt
# Make sure you're in the project root first
cd ../../
pip install -r requirements.txt
cd langchain_agents/rag_agent
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_key
GROQ_API_KEY=your_groq_key
CHROMADB_PERSIST_DIR=your_chroma_dir
LOG_LEVEL=INFO
```

## 💻 Usage

### Run the main script
From the rag_agent directory:
```bash
python main.py
```

Or from anywhere in the project:
```bash
python -m langchain_agents.rag_agent.main
```

### Available Options:
1. Extract PDF and update vector DB
2. Q&A using existing vector DB
3. Exit

## 📁 Project Structure

```
rag_agent/
├── README.md
├── __init__.py             # Package exports
├── main.py                 # Entry point
├── requirements.txt        # Agent-specific requirements
├── agents/                 # Agent components
│   ├── embedding_agent.py    # Handles embeddings
│   ├── logger_agent.py       # Logging system
│   ├── pdf_extractor_agent.py # PDF processing
│   ├── processing_agent.py    # Text processing
│   ├── rag_agent.py          # Main RAG logic
│   └── vectordb_agent.py     # Vector DB operations
└── llm/                    # LLM interface
    └── llm_caller.py        # Centralized LLM calls
```

### Related Files

#### Core Components
- Entry Point: [`main.py`](main.py)
- Package Definition: [`__init__.py`](__init__.py)
- Dependencies: [`requirements.txt`](requirements.txt)

#### LLM Integration
Located in root [`/wrappers`](../../wrappers/) directory:
- OpenAI: [`openai_wrapper.py`](../../wrappers/openai_wrapper.py)
- Google: [`gemini_wrapper.py`](../../wrappers/gemini_wrapper.py)
- Groq: [`groq_wrapper.py`](../../wrappers/groq_wrapper.py)
- HuggingFace: [`huggingface_wrapper.py`](../../wrappers/huggingface_wrapper.py)
- LLaMA: [`llama_wrapper.py`](../../wrappers/llama_wrapper.py)

#### Storage and Testing
- Vector Database: [`/chroma_db`](../../chroma_db/)

#### Documentation
- Framework Guide: [`../README.md`](../README.md)
- Project Overview: [`../../README.md`](../../README.md)
- License: [`../../LICENSE`](../../LICENSE)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
```bash
git checkout -b feature/YourFeature
```
3. Commit your changes
```bash
git commit -m 'Add some feature'
```
4. Push to the branch
```bash
git push origin feature/YourFeature
```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## 📧 Support

For issues, questions, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/Neerajnkm/agents/issues)
- **Documentation**: Check the docstrings in code

---