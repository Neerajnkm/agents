import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGAgent:
    def __init__(self, llm_caller, vectordb_agent, logger=None):
        self.llm_caller = llm_caller
        self.vectordb_agent = vectordb_agent
        self.logger = logger or logging.getLogger(__name__)
        self.text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def answer_query(self, user_query, top_k=3):
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