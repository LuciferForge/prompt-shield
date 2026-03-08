"""prompt-shield — Lightweight prompt injection detector for LLM applications."""

from .core.scanner import PromptScanner, ScanResult
from .core.exceptions import InjectionRiskError
from .core.patterns import PATTERNS, CATEGORIES

__version__ = "0.2.0"
__all__ = ["PromptScanner", "ScanResult", "InjectionRiskError", "PATTERNS", "CATEGORIES"]
