"""
Main entry point for the RAG Agent system. Handles PDF extraction, processing, embedding, vector DB storage, and Q&A.

Run this script to interactively extract data from PDFs and perform retrieval-augmented question answering.
"""
import os
from dotenv import load_dotenv
from .agents.pdf_extractor_agent import PDFExtractorAgent
from .agents.processing_agent import ProcessingAgent
from .agents.embedding_agent import EmbeddingAgent
from .agents.vectordb_agent import VectorDBAgent
from .agents.rag_agent import RAGAgent
from .agents.logger_agent import LoggerAgent
from .llm.llm_caller import LLMCentralCaller
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional

class AgentState(TypedDict, total=False):
    pdf_path: Optional[str]
    query: Optional[str]
    extracted: Optional[dict]
    processed: Optional[dict]
    embeddings: Optional[dict]
    answer: Optional[str]

# Load environment variables
load_dotenv()

# Initialize logger
logger_agent = LoggerAgent()
logger = logger_agent.get_logger()

# Initialize agents
pdf_extractor = PDFExtractorAgent(logger=logger)
processor = ProcessingAgent(logger=logger)
embedder = EmbeddingAgent(logger=logger)
vectordb = VectorDBAgent(logger=logger)
llm_caller = LLMCentralCaller()
rag_agent = RAGAgent(llm_caller, vectordb, logger=logger)

def extractor_node(state):
    """
    Extract content from the PDF file specified in the state and update the state with extracted data.

    Args:
        state (dict): The current agent state containing 'pdf_path'.

    Returns:
        dict: Updated state with 'extracted' content.
    """
    pdf_path = state.get("pdf_path")
    extracted = pdf_extractor.extract(pdf_path)
    logger.info(f"extractor_node: extracted content keys: {list(extracted.keys()) if extracted else extracted}")
    logger.info(f"extractor_node: extracted text length: {len(extracted.get('text', [])) if extracted else 0}")
    state["extracted"] = extracted
    return state

def processor_node(state):
    """
    Process the extracted content in the state and update the state with processed data.

    Args:
        state (dict): The current agent state containing 'extracted' content.

    Returns:
        dict: Updated state with 'processed' content.
    """
    extracted = state.get("extracted")
    logger.info(f"processor_node: received extracted content: {type(extracted)}, keys: {list(extracted.keys()) if extracted else extracted}")
    processed = processor.process(extracted)
    state["processed"] = processed
    return state

def embedder_node(state):
    """
    Generate embeddings for the processed content in the state and update the state with embeddings.

    Args:
        state (dict): The current agent state containing 'processed' content.

    Returns:
        dict: Updated state with 'embeddings'.
    """
    processed = state.get("processed")
    if processed is None:
        logger.warning("No processed content provided to embedder. Returning empty embeddings.")
        embeddings = embedder.embed({})
    else:
        embeddings = embedder.embed(processed)
    state["embeddings"] = embeddings
    return state

def vectordb_node(state):
    """
    Store embeddings from the state in the vector database.

    Args:
        state (dict): The current agent state containing 'embeddings'.

    Returns:
        dict: The unchanged state after storage.
    """
    embeddings = state.get("embeddings")
    if embeddings is None:
        logger.warning("No embeddings provided to vectordb. Skipping storage.")
        vectordb.store({})
    else:
        vectordb.store(embeddings)
    return state

def rag_node(state):
    """
    Answer a user query using the RAG agent and update the state with the answer.

    Args:
        state (dict): The current agent state containing 'query'.

    Returns:
        dict: Updated state with 'answer'.
    """
    query = state.get("query")
    answer = rag_agent.answer_query(query)
    state["answer"] = answer
    return state

def build_agent_graph():
    """
    Build the agent workflow graph for document processing and Q&A.

    Returns:
        StateGraph: The constructed agent workflow graph.
    """
    graph = StateGraph(AgentState)
    graph.add_node("extractor", extractor_node)
    graph.add_node("processor", processor_node)
    graph.add_node("embedder", embedder_node)
    graph.add_node("vectordb", vectordb_node)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "processor")
    graph.add_edge("processor", "embedder")
    graph.add_edge("embedder", "vectordb")
    graph.add_edge("vectordb", END)
    graph.add_edge("rag", END)
    return graph

if __name__ == "__main__":
    agent_graph = build_agent_graph()
    runnable_graph = agent_graph.compile()
    while True:
        print("Select activity:")
        print("1. Extract PDF and update vector DB")
        print("2. Q&A using existing vector DB")
        print("3. Exit")
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice == "1":
            pdf_path = input("Enter the path to the PDF file: ").strip()
            state = {"pdf_path": pdf_path}
            result_state = runnable_graph.invoke(state)
            print("Extraction and vector DB update complete.")
            print("Final state after extraction:", result_state)
        elif choice == "2":
            user_query = input("Enter your question: ").strip()
            answer = rag_agent.answer_query(user_query)
            print(f"Answer: {answer}")
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")