import logging
import os
from dotenv import load_dotenv
from langsmith import trace

class LoggerAgent:
    def __init__(self):
        load_dotenv()
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_dest = os.getenv("LOG_DEST", "console")
        log_format = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s %(message)s")
        self.logger = logging.getLogger("MultiAgentSystem")
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        formatter = logging.Formatter(log_format)
        if log_dest == "file":
            log_file = os.getenv("LOG_FILE", "agents.log")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        # LangSmith trace integration
        self.tracer = trace

    def get_logger(self):
        return self.logger

    def start_trace(self, name):
        return self.tracer.start_trace(name)

    def end_trace(self, trace_id):
        self.tracer.end_trace(trace_id)