"""Core scanner for prompt-shield."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

from .exceptions import InjectionRiskError
from .patterns import COMPILED_PATTERNS, CATEGORIES, score_to_severity


@dataclass
class ScanResult:
    """Result of scanning a single prompt."""

    text: str
    risk_score: int
    severity: str
    matches: List[dict] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.severity == "SAFE"

    def matches_by_category(self, category: str) -> List[dict]:
        """Return only matches from a specific category."""
        return [m for m in self.matches if m.get("category") == category]

    def __repr__(self) -> str:
        return (
            f"ScanResult(severity={self.severity!r}, score={self.risk_score}, "
            f"matches={[m['name'] for m in self.matches]})"
        )


class PromptScanner:
    """Scans text for prompt injection patterns.

    Args:
        threshold:       Severity level at which to raise InjectionRiskError.
                         One of: "LOW", "MEDIUM", "HIGH", "CRITICAL".
                         Default is "MEDIUM" -- blocks medium and above.
        custom_patterns: Optional list of additional pattern dicts to add.
                         Each must have: name, pattern (regex str), weight (int).
        categories:      Optional set of categories to scan. Default is all
                         categories EXCEPT "pii". Pass {"pii"} or
                         CATEGORIES to include PII scanning.
        exclude_categories: Optional set of categories to exclude from scanning.
                         Cannot be used together with `categories`.
    """

    SEVERITY_ORDER = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # Default: scan everything except PII (PII is opt-in)
    DEFAULT_CATEGORIES = CATEGORIES - {"pii"}

    def __init__(
        self,
        threshold: str = "MEDIUM",
        custom_patterns: Optional[List[dict]] = None,
        categories: Optional[Set[str]] = None,
        exclude_categories: Optional[Set[str]] = None,
    ):
        import re

        if threshold not in self.SEVERITY_ORDER:
            raise ValueError(f"threshold must be one of {self.SEVERITY_ORDER}")
        if categories is not None and exclude_categories is not None:
            raise ValueError("Cannot specify both 'categories' and 'exclude_categories'")

        self.threshold = threshold

        # Determine active categories
        if categories is not None:
            self._active_categories = set(categories)
        elif exclude_categories is not None:
            self._active_categories = CATEGORIES - set(exclude_categories)
        else:
            self._active_categories = self.DEFAULT_CATEGORIES

        # Filter patterns by active categories
        self._patterns = [
            p for p in COMPILED_PATTERNS
            if p.get("category", "unknown") in self._active_categories
        ]

        if custom_patterns:
            for p in custom_patterns:
                self._patterns.append({
                    **p,
                    "_regex": re.compile(p["pattern"], re.IGNORECASE | re.UNICODE),
                })

    def scan(self, text: str) -> ScanResult:
        """Scan text and return a ScanResult. Never raises."""
        matches = []
        total_score = 0

        for p in self._patterns:
            if p["_regex"].search(text):
                matches.append({
                    "name": p["name"],
                    "category": p.get("category", "unknown"),
                    "weight": p["weight"],
                })
                total_score += p["weight"]

        severity = score_to_severity(total_score)
        return ScanResult(
            text=text[:200],
            risk_score=total_score,
            severity=severity,
            matches=matches,
        )

    def check(self, text: str) -> ScanResult:
        """Scan text and raise InjectionRiskError if severity >= threshold."""
        result = self.scan(text)
        if self._exceeds_threshold(result.severity):
            raise InjectionRiskError(
                severity=result.severity,
                risk_score=result.risk_score,
                matches=[m["name"] for m in result.matches],
                text=text,
            )
        return result

    def protect(self, arg_index: int = 0, arg_name: Optional[str] = None) -> Callable:
        """Decorator that scans a function argument before calling the function.

        Args:
            arg_index: Positional index of the argument to scan (default 0).
            arg_name:  Keyword argument name to scan. Takes priority over arg_index.

        Usage::

            scanner = PromptScanner()

            @scanner.protect(arg_name="prompt")
            def call_llm(prompt: str):
                return client.chat(prompt)
        """
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if arg_name and arg_name in kwargs:
                    text = kwargs[arg_name]
                elif args and arg_index < len(args):
                    text = args[arg_index]
                else:
                    text = ""

                self.check(str(text))
                return fn(*args, **kwargs)

            return wrapper
        return decorator

    def _exceeds_threshold(self, severity: str) -> bool:
        try:
            return (
                self.SEVERITY_ORDER.index(severity)
                >= self.SEVERITY_ORDER.index(self.threshold)
            )
        except ValueError:
            return False
