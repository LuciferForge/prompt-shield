"""prompt-shield — Lightweight prompt injection detector + output scanner for LLM applications."""

from .core.scanner import PromptScanner, ScanResult
from .core.output_scanner import OutputScanner, OutputScanResult, OutputFinding
from .core.exceptions import InjectionRiskError
from .core.patterns import PATTERNS, CATEGORIES

__version__ = "0.3.0"
__all__ = [
    "PromptScanner", "ScanResult",
    "OutputScanner", "OutputScanResult", "OutputFinding",
    "InjectionRiskError",
    "PATTERNS", "CATEGORIES",
]
