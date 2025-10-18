import logging
import os
from dotenv import load_dotenv
from langsmith import trace

class LoggerAgent:
    """
    Agent for configuring and providing logging and tracing capabilities for the multi-agent system.

    Methods:
        get_logger(): Returns the configured logger instance.
        start_trace(name): Starts a trace with the given name (LangSmith integration).
        end_trace(trace_id): Ends the trace with the given trace ID.
    """
    def __init__(self):
        """
        Initialize the LoggerAgent by configuring logging based on environment variables and setting up tracing.
        """
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
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: The logger instance for the multi-agent system.
        """
        return self.logger

    def start_trace(self, name):
        """
        Start a trace with the given name using LangSmith tracing.

        Args:
            name (str): The name of the trace session.

        Returns:
            Any: The trace object or identifier.
        """
        return self.tracer.start_trace(name)

    def end_trace(self, trace_id):
        """
        End the trace session with the given trace ID.

        Args:
            trace_id (Any): The identifier of the trace session to end.
        """
        self.tracer.end_trace(trace_id)