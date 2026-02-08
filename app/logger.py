"""
Logging utilities for the Data Analysis API.
Provides structured logging for all API operations and subprocess executions.
"""
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name: str = "data_analysis_api", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a comprehensive logger for the application.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - daily rotating logs
    log_filename = LOGS_DIR / f"api_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler - separate error log
    error_log_filename = LOGS_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_log_filename, mode='a', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

def log_request(logger: logging.Logger, request_id: str, files: List[UploadFile], 
                question: str, analysis_type: str) -> None:
    """
    Log incoming API request details.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        files: List of uploaded files
        question: Analysis question
        analysis_type: Type of analysis requested
    """
    file_info = []
    total_size = 0
    
    for file in files:
        size = 0
        if hasattr(file, 'size') and file.size:
            size = file.size
            total_size += size
        elif hasattr(file, 'file'):
            # Try to get file size by seeking
            try:
                current_pos = file.file.tell()
                file.file.seek(0, 2)  # Seek to end
                size = file.file.tell()
                file.file.seek(current_pos)  # Restore position
                total_size += size
            except:
                size = 0
        
        file_info.append({
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': size
        })
    
    logger.info(
        f"NEW REQUEST | ID: {request_id} | "
        f"Files: {len(files)} ({total_size:,} bytes) | "
        f"Type: {analysis_type} | "
        f"Question: {question[:100]}{'...' if len(question) > 100 else ''}"
    )
    
    # Log detailed file information
    for i, info in enumerate(file_info):
        logger.info(
            f"FILE {i+1} | ID: {request_id} | "
            f"Name: {info['filename']} | "
            f"Type: {info['content_type']} | "
            f"Size: {info['size_bytes']:,} bytes"
        )

def log_execution(logger: logging.Logger, request_id: str, execution_result: Dict[str, Any]) -> None:
    """
    Log code execution results.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        execution_result: Execution result dictionary
    """
    success = execution_result.get('success', False)
    execution_time = execution_result.get('execution_time', 0)
    stdout_lines = len(execution_result.get('stdout', '').split('\n'))
    stderr_lines = len(execution_result.get('stderr', '').split('\n'))
    
    status = "SUCCESS" if success else "FAILED"
    
    logger.info(
        f"EXECUTION {status} | ID: {request_id} | "
        f"Time: {execution_time:.2f}s | "
        f"Stdout: {stdout_lines} lines | "
        f"Stderr: {stderr_lines} lines"
    )
    
    if not success:
        error = execution_result.get('error', 'Unknown error')
        logger.error(f"EXECUTION ERROR | ID: {request_id} | Error: {error}")
    
    # Log stdout/stderr if they contain important information
    stdout = execution_result.get('stdout', '')
    stderr = execution_result.get('stderr', '')
    
    if stdout.strip():
        logger.debug(f"STDOUT | ID: {request_id} | {stdout[:500]}{'...' if len(stdout) > 500 else ''}")
    
    if stderr.strip():
        logger.warning(f"STDERR | ID: {request_id} | {stderr[:500]}{'...' if len(stderr) > 500 else ''}")

def log_llm_interaction(logger: logging.Logger, request_id: str, prompt_length: int, 
                       response_length: int, model_used: str) -> None:
    """
    Log LLM interaction details.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        prompt_length: Length of prompt sent to LLM
        response_length: Length of response received
        model_used: Name of the LLM model used
    """
    logger.info(
        f"LLM INTERACTION | ID: {request_id} | "
        f"Model: {model_used} | "
        f"Prompt: {prompt_length} chars | "
        f"Response: {response_length} chars"
    )

def log_docker_operation(logger: logging.Logger, request_id: str, operation: str, 
                        container_id: str = None, duration: float = None) -> None:
    """
    Log Docker container operations.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        operation: Operation type (create, start, stop, remove)
        container_id: Docker container ID
        duration: Operation duration in seconds
    """
    log_msg = f"DOCKER {operation.upper()} | ID: {request_id}"
    if container_id:
        log_msg += f" | Container: {container_id[:12]}"
    if duration is not None:
        log_msg += f" | Duration: {duration:.2f}s"
    
    logger.info(log_msg)

def log_file_operation(logger: logging.Logger, request_id: str, operation: str, 
                      file_path: str, size: int = None) -> None:
    """
    Log file operations.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        operation: Operation type (save, read, delete, create)
        file_path: File path
        size: File size in bytes
    """
    log_msg = f"FILE {operation.upper()} | ID: {request_id} | Path: {file_path}"
    if size is not None:
        log_msg += f" | Size: {size:,} bytes"
    
    logger.info(log_msg)

def log_api_request(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    method: str,
    files_count: int = 0,
    question_preview: str = "",
    **kwargs
):
    """
    Log API request details.
    
    Args:
        logger: Logger instance
        request_id: Request ID
        endpoint: API endpoint
        method: HTTP method
        files_count: Number of files uploaded
        question_preview: Preview of the question
        **kwargs: Additional parameters to log
    """
    log_msg = f"API REQUEST | ID: {request_id} | {method} {endpoint}"
    
    if files_count > 0:
        log_msg += f" | Files: {files_count}"
    
    if question_preview:
        log_msg += f" | Question: {question_preview}"
    
    # Add any additional parameters
    for key, value in kwargs.items():
        log_msg += f" | {key}: {value}"
    
    logger.info(log_msg)

class RequestLogger:
    """Context manager for request-specific logging."""
    
    def __init__(self, logger: logging.Logger, request_id: str):
        self.logger = logger
        self.request_id = request_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"REQUEST START | ID: {self.request_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"REQUEST COMPLETE | ID: {self.request_id} | Duration: {duration:.2f}s")
        else:
            self.logger.error(
                f"REQUEST FAILED | ID: {self.request_id} | "
                f"Duration: {duration:.2f}s | Error: {exc_val}"
            )
    
    def info(self, message: str):
        """Log info message with request ID."""
        self.logger.info(f"ID: {self.request_id} | {message}")
    
    def error(self, message: str):
        """Log error message with request ID."""
        self.logger.error(f"ID: {self.request_id} | {message}")
    
    def warning(self, message: str):
        """Log warning message with request ID."""
        self.logger.warning(f"ID: {self.request_id} | {message}")
    
    def debug(self, message: str):
        """Log debug message with request ID."""
        self.logger.debug(f"ID: {self.request_id} | {message}")


# Create a global logger instance for easy import
logger = setup_logger()
