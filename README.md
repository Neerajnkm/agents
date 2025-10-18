# AI Agents System 🤖

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A collection of specialized AI agents designed for various tasks including document processing, retrieval-augmented generation (RAG), and intelligent automation. The system is built to be modular, extensible, and framework-agnostic, allowing integration with multiple AI/ML libraries and models.

## 🌟 Overview

This repository contains intelligent agents designed to handle complex tasks using various AI frameworks and models. Each agent is specialized for specific tasks while maintaining interoperability and extensibility.

### Current Agents

#### [RAG Agent System](langchain_agents/rag_agent/README.md)
A sophisticated Retrieval-Augmented Generation system that:
- Processes PDF documents with advanced text extraction
- Provides intelligent question-answering capabilities
- Supports multiple LLM providers (OpenAI, Groq, HuggingFace, etc.)
- Uses vector databases for efficient information retrieval

## 🏗️ Project Structure

```
.
├── LICENSE                 # MIT License file
├── README.md              # Root documentation
├── requirements.txt       # Root requirements (auto-updated)
├── update_root_requirements.py  # Script to update root requirements
├── chroma_db/            # Vector database storage
│   └── chroma.sqlite3    # Database file
├── langchain_agents/     # LangChain implementations
│   ├── README.md        # LangChain agents documentation
│   └── rag_agent/       # RAG system implementation
│       ├── README.md    # RAG-specific documentation
│       ├── __init__.py  # Package exports
│       ├── main.py      # Entry point
│       ├── requirements.txt  # RAG-specific requirements
│       ├── agents/      # Agent components
│       └── llm/         # LLM interface
└── wrappers/            # LLM provider wrappers
    ├── __init__.py
    ├── gemini_wrapper.py
    ├── groq_wrapper.py
    ├── huggingface_wrapper.py
    ├── llama_wrapper.py
    └── openai_wrapper.py
```

## 🚀 Getting Started

1. Clone the repository
```bash
git clone git@github.com:Neerajnkm/agents.git
cd Agents
```

2. Create and activate a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies
```bash
python update_root_requirements.py  # Updates root requirements.txt
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_key
GROQ_API_KEY=your_groq_key
CHROMADB_PERSIST_DIR=chroma_db
LOG_LEVEL=INFO
```

## 🔧 Prerequisites

### System Dependencies
Required for PDF processing and text extraction:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

**CentOS/RHEL:**
```bash
sudo yum install -y tesseract poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

### System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: 1GB free space minimum
- **GPU**: Optional but recommended for faster processing
- **Internet**: Required for API calls to model providers

## 📚 Documentation

Each agent system has its own detailed documentation in its respective directory:
- [RAG Agent Documentation](langchain_agents/rag_agent/README.md)

## 🛠️ Development

### Adding a New Agent System

You can create new agents using any AI framework (e.g., LangChain, Semantic Kernel, AutoGen, custom implementation). Follow these general steps:

1. Create a new directory structure:
```
agents/
├── your_agent_name/
│   ├── __init__.py
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   └── config/           # Configuration files
│   ├── core/            # Core agent logic
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── utils/           # Utility functions
│   └── tests/           # Test files
```

2. Add agent-specific requirements in `requirements.txt`
3. Run `python update_root_requirements.py` to update root requirements
4. Follow framework-specific guidelines below

### Framework-Specific Guidelines

#### Using LangChain
For LangChain-based agents (like the existing RAG agent):
1. Create agent under `langchain_agents/`
2. Implement agent components following LangChain patterns:
   - Use BaseAgent for agent implementations
   - Implement Tools and Chains as needed
   - Set up proper prompts and templates
3. Add necessary wrappers in `wrappers/` for LLM providers

#### Using Semantic Kernel
For Microsoft Semantic Kernel based agents:
1. Create agent under `semantic_kernel_agents/`
2. Implement agent components:
   - Define Skills (equivalent to tools)
   - Set up Semantic Functions
   - Configure proper prompts in skprompt.txt files
3. Add kernel configuration in config/

#### Using AutoGen
For Microsoft AutoGen based agents:
1. Create agent under `autogen_agents/`
2. Define agent types and roles:
   - Assistant agents
   - User proxies
   - Group chats
3. Configure agent interactions and workflows

#### Custom Implementation
For custom agent implementations:
1. Create agent under `custom_agents/`
2. Follow clean architecture principles:
   - Clear separation of concerns
   - Define interfaces for components
   - Implement proper dependency injection
3. Document architecture decisions

### Design Principles
- **Modularity**: Each agent should be self-contained with clear interfaces
- **Framework Agnostic**: Support multiple AI frameworks and model providers
- **Extensibility**: Easy to add new capabilities and integrations
- **Configurability**: Flexible configuration for different use cases
- **Testing**: Include comprehensive tests for agent behaviors
- **Documentation**: Provide clear usage instructions and examples

### Testing Requirements
For any new agent implementation:
1. Add unit tests for core components
2. Include integration tests for agent workflows
3. Add performance benchmarks if applicable
4. Document testing procedures in agent's README

### Documentation Requirements
Each new agent should include:
1. Clear README with setup instructions
2. API documentation if exposing interfaces
3. Usage examples with expected outputs
4. Configuration guide
5. Troubleshooting section

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### What does this mean?

- ✅ You can freely use, copy, modify, merge, publish, distribute this software
- ✅ You can use it for commercial purposes
- ✅ You can make changes to the code and keep them private or share them
- ❗ The only requirement is to include the original license and copyright notice in any copy of the software/source
- ⛔ No warranty is provided

For more details, refer to the [LICENSE](LICENSE) file in the root directory.

## 📧 Support

For issues, questions, or feature requests:
- **GitHub Issues**: [Create an issue](https://github.com/Neerajnkm/agents/issues)
- **Documentation**: Check the READMEs and docstrings in code