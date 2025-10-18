import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) agent for answering queries using vector database retrieval and LLMs.

    Methods:
        answer_query(user_query, top_k=3): Answers a user query using retrieved context and an LLM.
    """
    def __init__(self, llm_caller, vectordb_agent, logger=None):
        """
        Initialize the RAGAgent.

        Args:
            llm_caller: An object responsible for calling the language model.
            vectordb_agent: An agent for querying the vector database.
            logger (logging.Logger, optional): Logger instance for logging. Defaults to None.
        """
        self.llm_caller = llm_caller
        self.vectordb_agent = vectordb_agent
        self.logger = logger or logging.getLogger(__name__)
        self.text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def answer_query(self, user_query, top_k=3):
        """
        Answer a user query by retrieving relevant context from the vector database and generating a response using an LLM.

        Args:
            user_query (str): The user's question.
            top_k (int, optional): Number of top results to retrieve from the vector database. Defaults to 3.

        Returns:
            str: The generated answer from the LLM based on retrieved context.

        Raises:
            ValueError: If user_query is None.
        """
        self.logger.info(f"Answering user query: {user_query}")
        # Ensure user_query is a string and not None
        if user_query is None:
            raise ValueError("user_query cannot be None")
        user_query = str(user_query)
        # Embed the query
        query_embedding = self.text_embedder.embed_query(user_query)
        # Query all collections
        all_results = self.vectordb_agent.query_all_collections(query_embedding, top_k=top_k)
        # Compose context for LLM from all results
        context = "\n".join([f"[{col}] {docs}" for col, docs in all_results.items() if docs])
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        # Call LLM
        answer = self.llm_caller.call(prompt)
        return answer