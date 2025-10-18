import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import numpy as np

class EmbeddingAgent:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, chunks):
        # self.logger.info("Embedding text, images, diagrams, charts, tables")
        text_chunks = chunks.get("text_chunks", [])
        images = chunks.get("images", [])
        diagrams = chunks.get("diagrams", [])
        charts = chunks.get("charts", [])
        tables = chunks.get("tables", [])
        # Filter out None values from text_chunks
        text_chunks = [t for t in text_chunks if t is not None]
        # Also ensure all are strings
        text_chunks = [str(t) for t in text_chunks]
        # Embed text
        text_embeddings = self.text_embedder.embed_documents(text_chunks) if text_chunks else []
        # Embed images (placeholder: flatten to vector, replace with CLIP for real use)
        image_embeddings = [np.array(img).flatten()[:512].tolist() for img in images] if images else []
        diagram_embeddings = [np.array(img).flatten()[:512].tolist() for img in diagrams] if diagrams else []
        # Embed charts
        chart_embeddings = []
        chart_docs = []
        for chart in charts:
            if isinstance(chart, dict) and 'data' in chart:
                chart_emb = self.text_embedder.embed_documents([chart['data']])[0]
                chart_embeddings.append(chart_emb)
                chart_docs.append(chart['data'])
            else:
                chart_emb = np.array(chart).flatten()[:512].tolist()
                chart_embeddings.append(chart_emb)
                chart_docs.append(str(chart))
        # Embed tables
        table_embeddings = []
        table_docs = []
        for table in tables:
            if isinstance(table, dict) and 'data' in table:
                table_emb = self.text_embedder.embed_documents([table['data']])[0]
                table_embeddings.append(table_emb)
                table_docs.append(table['data'])
            else:
                table_emb = self.text_embedder.embed_documents([str(table)])[0]
                table_embeddings.append(table_emb)
                table_docs.append(str(table))
        
        # If no embeddings are generated, return None
        if not text_embeddings and not image_embeddings and not diagram_embeddings and not chart_embeddings and not table_embeddings:
            return None

        return {
            "text": list(zip(text_chunks, text_embeddings)),
            "images": list(zip(images, image_embeddings)),
            "diagrams": list(zip(diagrams, diagram_embeddings)),
            "charts": list(zip(chart_docs, chart_embeddings)),
            "tables": list(zip(table_docs, table_embeddings)),
        }