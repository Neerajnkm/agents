import logging
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os

load_dotenv()


class VectorDBAgent:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PERSIST_DIR"), settings=Settings(allow_reset=True))
        self.text_collection = self.chroma_client.get_or_create_collection("rag_text_collection")
        self.image_collection = self.chroma_client.get_or_create_collection("rag_image_collection")
        self.diagram_collection = self.chroma_client.get_or_create_collection("rag_diagram_collection")
        self.chart_collection = self.chroma_client.get_or_create_collection("rag_chart_collection")
        self.table_collection = self.chroma_client.get_or_create_collection("rag_table_collection")
        self.chart_image_collection = self.chroma_client.get_or_create_collection("rag_chart_image_collection")
        self.chart_text_collection = self.chroma_client.get_or_create_collection("rag_chart_text_collection")
        self.logger.info("Initialized VectorDBAgent with collections.")

    def store(self, embeddings):
        self.logger.info("Storing embeddings in vector DB")
        # Store text embeddings
        for idx, (chunk, emb) in enumerate(embeddings.get("text", [])):
            self.text_collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[f"text_{idx}"]
            )
        # Store image embeddings
        for idx, (img, emb) in enumerate(embeddings.get("images", [])):
            self.image_collection.add(
                documents=[str(img)],
                embeddings=[emb],
                ids=[f"image_{idx}"]
            )
        # Store diagram embeddings
        for idx, (diagram, emb) in enumerate(embeddings.get("diagrams", [])):
            self.diagram_collection.add(
                documents=[str(diagram)],
                embeddings=[emb],
                ids=[f"diagram_{idx}"]
            )
        # Store chart embeddings in separate collections based on dimension
        for idx, (chart, emb) in enumerate(embeddings.get("charts", [])):
            if isinstance(emb, list) and len(emb) == 384:
                self.chart_text_collection.add(
                    documents=[chart],
                    embeddings=[emb],
                    ids=[f"chart_text_{idx}"]
                )
            elif isinstance(emb, list) and len(emb) == 512:
                self.chart_image_collection.add(
                    documents=[chart],
                    embeddings=[emb],
                    ids=[f"chart_image_{idx}"]
                )
        # Store table embeddings
        for idx, (table, emb) in enumerate(embeddings.get("tables", [])):
            if isinstance(table, dict) and "data" in table:
                doc = table["data"]
            else:
                doc = str(table)
            self.table_collection.add(
                documents=[doc],
                embeddings=[emb],
                ids=[f"table_{idx}"]
            )

    def query(self, query_embedding, top_k=3):
        self.logger.info("Querying vector DB")
        # Use persistent ChromaDB location
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PERSIST_DIR"))
        collection = chroma_client.get_or_create_collection("rag_text_collection")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []

    def query_all_collections(self, query_embedding, top_k=3):
        """Query all collections in the persistent ChromaDB and return results."""
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PERSIST_DIR"))
        results = {}
        for col in chroma_client.list_collections():
            collection = chroma_client.get_collection(col.name)
            try:
                res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
                results[col.name] = res["documents"][0] if res["documents"] else []
            except Exception as e:
                results[col.name] = f"Error: {e}"
        return results

    def list_collections(self):
        """Return a list of all collection names in the persistent ChromaDB."""
        chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PERSIST_DIR"))
        return [col.name for col in chroma_client.list_collections()]