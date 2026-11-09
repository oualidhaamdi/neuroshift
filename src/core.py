"""
Core utilities for Brain Shift Analysis.
Includes centralized logging and configuration loading.
"""

import sys
import time
import logging
import os
from pathlib import Path
import importlib.util

class PipelineLogger(logging.Logger):
    """Logger with pipeline-specific formatting and timing."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._timers = {}

    def section(self, message: str):
        self.info("=" * 40)
        self.info(message.upper())
        self.info("=" * 40)

    def success(self, message: str):
        self.info(f"[OK] {message}")

    def start_timer(self, name: str):
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> float:
        if name not in self._timers:
            self.warning(f"Timer '{name}' was never started.")
            return 0.0
        return time.monotonic() - self._timers.pop(name)

    def setup_file_logging(self, output_dir):
        log_path = Path(output_dir) / "pipeline.log"
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        
        root = logging.getLogger("brain_shift")
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            root.addHandler(fh)
            self.info(f"Log file: {log_path}")

def setup_logger(name: str = __name__, level: int = logging.INFO) -> PipelineLogger:
    logging.setLoggerClass(PipelineLogger)
    full_name = f"brain_shift.{name}" if not name.startswith("brain_shift") else name
    logger = logging.getLogger(full_name)
    logger.setLevel(level)

    root = logging.getLogger("brain_shift")
    root.setLevel(level)

    if not root.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    return logger # type: ignore

def load_config():
    """Find and load config.py from project root."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "config.py").exists():
            break
        current = current.parent
    else:
        raise FileNotFoundError("config.py not found.")

    spec = importlib.util.spec_from_file_location("config", current / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config

logger = setup_logger("brain_shift")
