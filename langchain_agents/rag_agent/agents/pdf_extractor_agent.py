import logging
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import tempfile
import cv2
import numpy as np
import pytesseract

class PDFExtractorAgent:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def extract(self, pdf_path):
        self.logger.info(f"Extracting content from {pdf_path}")
        text_chunks = []
        images = []
        diagrams = []
        charts = []
        tables = []
        # Extract text
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
                # Table extraction using PyPDF2
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        obj = annot.get_object()
                        if obj.get('/Subtype') == '/Widget' and obj.get('/FT') == '/Btn':
                            tables.append(obj)
            self.logger.info(f"Extracted text chunks: {text_chunks}")
            self.logger.info(f"Extracted tables (PyPDF2): {tables}")
        except Exception as e:
            self.logger.error(f"Text/table extraction failed: {e}")
        # Extract images
        try:
            with tempfile.TemporaryDirectory() as path:
                pil_images = convert_from_path(pdf_path, output_folder=path)
                self.logger.info(f"Extracted images: {len(pil_images)}")
                for img in pil_images:
                    images.append(img)
                    # Convert PIL image to OpenCV format
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    # Simple diagram detection: edge detection
                    edges = cv2.Canny(cv_img, 100, 200)
                    if np.sum(edges) > 10000:  # Arbitrary threshold
                        diagrams.append(img)
                    # Chart detection: look for large colored regions
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 5000:  # Arbitrary threshold for chart
                            charts.append(img)
                            # Extract chart data using OCR
                            chart_text = pytesseract.image_to_string(img)
                            if chart_text:
                                charts.append({'image': img, 'data': chart_text})
                    # Table extraction from image using OCR
                    table_text = pytesseract.image_to_string(img, config='--psm 6')
                    if '|' in table_text or '\t' in table_text:
                        tables.append({'image': img, 'data': table_text})
        except Exception as e:
            self.logger.error(f"Image/diagram/chart/table extraction failed: {e}")
        self.logger.info(f"Final extracted content: text={len(text_chunks)}, images={len(images)}, diagrams={len(diagrams)}, charts={len(charts)}, tables={len(tables)}")
        return {"text": text_chunks, "images": images, "diagrams": diagrams, "charts": charts, "tables": tables}