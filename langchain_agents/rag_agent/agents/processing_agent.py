import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ProcessingAgent:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def process(self, extracted_content):
        if extracted_content is None:
            self.logger.warning("No extracted content provided to processor. Returning empty result.")
            return {"text_chunks": [], "images": [], "diagrams": [], "charts": [], "tables": []}
        self.logger.info("Processing extracted content: chunking text and preparing images, diagrams, charts, tables")
        # Chunk text
        text_chunks = []
        for text in extracted_content.get("text", []):
            text_chunks.extend(self.text_splitter.split_text(text))
        images = extracted_content.get("images", [])
        diagrams = extracted_content.get("diagrams", [])
        charts = extracted_content.get("charts", [])
        tables = extracted_content.get("tables", [])
        return {
            "text_chunks": text_chunks,
            "images": images,
            "diagrams": diagrams,
            "charts": charts,
            "tables": tables
        }