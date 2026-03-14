"""Output scanner for prompt-shield.

Scans LLM outputs for:
  - Hallucinated URLs (HEAD check — do they actually exist?)
  - System prompt leakage (did the LLM dump its instructions?)
  - Sensitive data leakage (API keys, passwords, secrets in output)
  - PII leakage (SSN, credit cards, emails in output)
  - Package hallucination (did the LLM invent a Python/npm package?)
  - Code safety (dangerous function calls in generated code)

Zero ML dependencies. Pure stdlib + urllib for URL checks.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse


@dataclass
class OutputFinding:
    """A single finding from output scanning."""
    scanner: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    evidence: str  # The actual content that triggered the finding
    weight: int = 5


@dataclass
class OutputScanResult:
    """Result of scanning LLM output."""
    text: str
    findings: List[OutputFinding] = field(default_factory=list)
    risk_score: int = 0
    severity: str = "SAFE"

    @property
    def is_safe(self) -> bool:
        return self.severity == "SAFE"

    def __repr__(self) -> str:
        return (
            f"OutputScanResult(severity={self.severity!r}, score={self.risk_score}, "
            f"findings={[f.scanner for f in self.findings]})"
        )


def _score_to_severity(score: int) -> str:
    if score == 0:
        return "SAFE"
    if score <= 3:
        return "LOW"
    if score <= 6:
        return "MEDIUM"
    if score <= 9:
        return "HIGH"
    return "CRITICAL"


# ─── URL patterns ────────────────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r'https?://[^\s<>\'")\]},;]+',
    re.IGNORECASE,
)


def _check_url_exists(url: str, timeout: float = 5.0) -> bool:
    """Check if a URL actually resolves. Returns True if reachable."""
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    import ssl

    try:
        # Create a non-verifying context for speed (we're checking existence, not security)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = Request(url, method="HEAD", headers={"User-Agent": "prompt-shield/0.3.0"})
        resp = urlopen(req, timeout=timeout, context=ctx)
        return resp.status < 500
    except HTTPError as e:
        # 4xx = page exists but forbidden/not found — 404 is hallucinated, 403 exists
        return e.code != 404
    except (URLError, OSError, ValueError):
        return False


# ─── Sensitive data patterns ─────────────────────────────────────────────────

_SENSITIVE_PATTERNS = [
    {
        "name": "api_key_generic",
        "pattern": re.compile(
            r'(?:api[_-]?key|apikey|api[_-]?secret|api[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})',
            re.IGNORECASE,
        ),
        "weight": 9,
        "message": "API key/secret exposed in output",
    },
    {
        "name": "openai_key",
        "pattern": re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'),
        "weight": 10,
        "message": "OpenAI API key in output",
    },
    {
        "name": "aws_key",
        "pattern": re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
        "weight": 10,
        "message": "AWS Access Key ID in output",
    },
    {
        "name": "github_token",
        "pattern": re.compile(r'\b(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}\b'),
        "weight": 10,
        "message": "GitHub token in output",
    },
    {
        "name": "stripe_key",
        "pattern": re.compile(r'\b[sr]k_(live|test)_[a-zA-Z0-9]{20,}\b'),
        "weight": 10,
        "message": "Stripe API key in output",
    },
    {
        "name": "slack_token",
        "pattern": re.compile(r'\bxox[bprs]-[a-zA-Z0-9\-]{10,}\b'),
        "weight": 9,
        "message": "Slack token in output",
    },
    {
        "name": "private_key",
        "pattern": re.compile(r'-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----'),
        "weight": 10,
        "message": "Private key in output",
    },
    {
        "name": "password_assignment",
        "pattern": re.compile(
            r'(?:password|passwd|pwd|secret)\s*[:=]\s*["\']?([^\s"\']{8,})',
            re.IGNORECASE,
        ),
        "weight": 8,
        "message": "Password/secret value in output",
    },
    {
        "name": "connection_string",
        "pattern": re.compile(
            r'(?:mongodb|postgres|mysql|redis|amqp)://[^\s<>"\']+:[^\s<>"\']+@',
            re.IGNORECASE,
        ),
        "weight": 9,
        "message": "Database connection string with credentials in output",
    },
    {
        "name": "jwt_token",
        "pattern": re.compile(r'\beyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\b'),
        "weight": 8,
        "message": "JWT token in output",
    },
]

# ─── PII patterns ────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    {
        "name": "ssn",
        "pattern": re.compile(r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b'),
        "weight": 9,
        "message": "US Social Security Number in output",
    },
    {
        "name": "credit_card",
        "pattern": re.compile(
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
        ),
        "weight": 9,
        "message": "Credit card number in output",
    },
    {
        "name": "email",
        "pattern": re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'),
        "weight": 3,
        "message": "Email address in output",
    },
]

# ─── System prompt leakage patterns ──────────────────────────────────────────

_SYSTEM_PROMPT_PATTERNS = [
    {
        "name": "system_prompt_header",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:system\s*(?:prompt|message|instructions?)|'
            r'you\s+are\s+(?:a|an)\s+(?:AI|assistant|helpful)|'
            r'your\s+(?:role|purpose|instructions?)\s+(?:is|are)\s+to)\s*[:\-]',
            re.IGNORECASE | re.MULTILINE,
        ),
        "weight": 9,
        "message": "System prompt content leaked in output",
    },
    {
        "name": "instruction_block",
        "pattern": re.compile(
            r'(?:here\s+(?:are|is)\s+my\s+(?:system\s+)?(?:prompt|instructions?)|'
            r'my\s+(?:system\s+)?(?:prompt|instructions?)\s+(?:is|are|says?)\s*[:\-]|'
            r'I\s+was\s+(?:given|told|instructed)\s+(?:to|the\s+following))',
            re.IGNORECASE,
        ),
        "weight": 8,
        "message": "LLM appears to be disclosing its instructions",
    },
    {
        "name": "xml_system_tag",
        "pattern": re.compile(
            r'<(?:system|instructions?|system[_-]prompt)>.*?</(?:system|instructions?|system[_-]prompt)>',
            re.IGNORECASE | re.DOTALL,
        ),
        "weight": 9,
        "message": "System prompt leaked in XML-style tags",
    },
]

# ─── Dangerous code patterns (for generated code) ────────────────────────────

_DANGEROUS_CODE_PATTERNS = [
    {
        "name": "code_eval_exec",
        "pattern": re.compile(r'\b(?:eval|exec)\s*\('),
        "weight": 7,
        "message": "eval/exec in generated code — potential code injection",
    },
    {
        "name": "code_subprocess_shell",
        "pattern": re.compile(r'subprocess\.\w+\([^)]*shell\s*=\s*True'),
        "weight": 8,
        "message": "subprocess with shell=True in generated code",
    },
    {
        "name": "code_os_system",
        "pattern": re.compile(r'\bos\.system\s*\('),
        "weight": 7,
        "message": "os.system() in generated code — use subprocess instead",
    },
    {
        "name": "code_pickle_load",
        "pattern": re.compile(r'pickle\.loads?\s*\('),
        "weight": 8,
        "message": "pickle.load() in generated code — deserialization vulnerability",
    },
    {
        "name": "code_yaml_unsafe",
        "pattern": re.compile(r'yaml\.(?:load|unsafe_load)\s*\([^)]*(?!Loader)'),
        "weight": 7,
        "message": "yaml.load() without safe Loader in generated code",
    },
]


class OutputScanner:
    """Scans LLM output for safety issues.

    Zero ML dependencies. Uses regex, HTTP checks, and AST parsing.

    Args:
        scanners:     Set of scanner names to enable. Default: all except "url" (network).
                      Options: "secrets", "pii", "system_prompt", "code", "url", "packages"
        system_prompt: Optional system prompt text. If provided, enables more accurate
                      system prompt leakage detection by checking for exact content matches.
        url_timeout:  Timeout in seconds for URL existence checks. Default: 5.0.
    """

    ALL_SCANNERS = {"secrets", "pii", "system_prompt", "code", "url", "packages"}
    # URL and packages require network — opt-in only
    DEFAULT_SCANNERS = {"secrets", "pii", "system_prompt", "code"}

    def __init__(
        self,
        scanners: Optional[Set[str]] = None,
        system_prompt: Optional[str] = None,
        url_timeout: float = 5.0,
    ):
        self._active_scanners = scanners or self.DEFAULT_SCANNERS
        self._system_prompt = system_prompt
        self._url_timeout = url_timeout

        # Pre-compute system prompt fragments for leakage detection
        self._system_prompt_fragments: List[str] = []
        if system_prompt and len(system_prompt) > 20:
            # Split into meaningful chunks and keep phrases 5+ words
            words = system_prompt.split()
            for i in range(0, len(words) - 4):
                fragment = " ".join(words[i:i + 5]).lower()
                if len(fragment) > 25:  # Skip short generic phrases
                    self._system_prompt_fragments.append(fragment)

    def scan(self, text: str) -> OutputScanResult:
        """Scan LLM output text. Returns OutputScanResult."""
        findings: List[OutputFinding] = []

        if "secrets" in self._active_scanners:
            findings.extend(self._scan_secrets(text))

        if "pii" in self._active_scanners:
            findings.extend(self._scan_pii(text))

        if "system_prompt" in self._active_scanners:
            findings.extend(self._scan_system_prompt_leakage(text))

        if "code" in self._active_scanners:
            findings.extend(self._scan_code_safety(text))

        if "url" in self._active_scanners:
            findings.extend(self._scan_urls(text))

        if "packages" in self._active_scanners:
            findings.extend(self._scan_packages(text))

        total_score = sum(f.weight for f in findings)
        severity = _score_to_severity(total_score)

        return OutputScanResult(
            text=text[:200],
            findings=findings,
            risk_score=total_score,
            severity=severity,
        )

    def _scan_secrets(self, text: str) -> List[OutputFinding]:
        findings = []
        for pat in _SENSITIVE_PATTERNS:
            match = pat["pattern"].search(text)
            if match:
                findings.append(OutputFinding(
                    scanner="secrets",
                    severity="CRITICAL" if pat["weight"] >= 9 else "HIGH",
                    message=pat["message"],
                    evidence=match.group(0)[:80],
                    weight=pat["weight"],
                ))
        return findings

    def _scan_pii(self, text: str) -> List[OutputFinding]:
        findings = []
        for pat in _PII_PATTERNS:
            match = pat["pattern"].search(text)
            if match:
                findings.append(OutputFinding(
                    scanner="pii",
                    severity="HIGH" if pat["weight"] >= 7 else "MEDIUM",
                    message=pat["message"],
                    evidence=match.group(0)[:40],
                    weight=pat["weight"],
                ))
        return findings

    def _scan_system_prompt_leakage(self, text: str) -> List[OutputFinding]:
        findings = []

        # Pattern-based detection
        for pat in _SYSTEM_PROMPT_PATTERNS:
            match = pat["pattern"].search(text)
            if match:
                findings.append(OutputFinding(
                    scanner="system_prompt",
                    severity="CRITICAL",
                    message=pat["message"],
                    evidence=match.group(0)[:100],
                    weight=pat["weight"],
                ))

        # Exact match detection (if system prompt provided)
        if self._system_prompt_fragments:
            text_lower = text.lower()
            matched_fragments = 0
            for fragment in self._system_prompt_fragments:
                if fragment in text_lower:
                    matched_fragments += 1

            # If 3+ unique fragments from the system prompt appear, it's likely leaked
            if matched_fragments >= 3:
                findings.append(OutputFinding(
                    scanner="system_prompt",
                    severity="CRITICAL",
                    message=f"System prompt content detected in output ({matched_fragments} matching fragments)",
                    evidence=f"{matched_fragments}/{len(self._system_prompt_fragments)} fragments matched",
                    weight=10,
                ))

        return findings

    def _scan_code_safety(self, text: str) -> List[OutputFinding]:
        """Scan code blocks in output for dangerous patterns."""
        findings = []

        # Extract code blocks (```...```) and inline code
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if not code_blocks:
            # Try scanning the whole text if it looks like code
            if any(kw in text for kw in ['import ', 'def ', 'class ', 'from ']):
                code_blocks = [text]

        for code in code_blocks:
            for pat in _DANGEROUS_CODE_PATTERNS:
                match = pat["pattern"].search(code)
                if match:
                    findings.append(OutputFinding(
                        scanner="code",
                        severity="HIGH" if pat["weight"] >= 8 else "MEDIUM",
                        message=pat["message"],
                        evidence=match.group(0)[:60],
                        weight=pat["weight"],
                    ))

        return findings

    def _scan_urls(self, text: str) -> List[OutputFinding]:
        """Check if URLs in the output actually exist (hallucination detection)."""
        findings = []
        urls = _URL_PATTERN.findall(text)

        # Deduplicate and limit to avoid excessive requests
        seen = set()
        for url in urls[:10]:
            # Clean trailing punctuation
            url = url.rstrip('.,;:!?)]}')
            parsed = urlparse(url)
            if not parsed.netloc or parsed.netloc in seen:
                continue
            seen.add(parsed.netloc)

            # Skip known-good domains
            if parsed.netloc in {
                'github.com', 'google.com', 'stackoverflow.com', 'wikipedia.org',
                'python.org', 'pypi.org', 'npmjs.com', 'docs.python.org',
                'developer.mozilla.org', 'w3.org', 'example.com', 'localhost',
            }:
                continue

            if not _check_url_exists(url, timeout=self._url_timeout):
                findings.append(OutputFinding(
                    scanner="url",
                    severity="MEDIUM",
                    message=f"Hallucinated URL (does not resolve): {parsed.netloc}",
                    evidence=url[:100],
                    weight=5,
                ))

        return findings

    def _scan_packages(self, text: str) -> List[OutputFinding]:
        """Check if Python packages mentioned in output actually exist on PyPI."""
        from urllib.request import urlopen
        from urllib.error import HTTPError, URLError

        findings = []

        # Find pip install commands and import statements in code blocks
        pip_pattern = re.compile(r'pip\s+install\s+([a-zA-Z0-9_\-]+)', re.IGNORECASE)
        import_pattern = re.compile(r'(?:^|\n)\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)')

        packages = set()
        for match in pip_pattern.finditer(text):
            pkg = match.group(1).strip()
            if pkg and pkg not in {'e', 'r', 'U'}:  # Skip flags
                packages.add(pkg)

        for match in import_pattern.finditer(text):
            pkg = match.group(1).strip()
            # Skip stdlib modules
            if pkg not in {
                'os', 'sys', 're', 'json', 'math', 'time', 'datetime', 'collections',
                'itertools', 'functools', 'pathlib', 'typing', 'dataclasses', 'abc',
                'logging', 'unittest', 'argparse', 'hashlib', 'hmac', 'secrets',
                'urllib', 'http', 'socket', 'ssl', 'email', 'io', 'csv', 'sqlite3',
                'threading', 'multiprocessing', 'subprocess', 'shutil', 'tempfile',
                'glob', 'fnmatch', 'copy', 'pprint', 'textwrap', 'string',
                'struct', 'codecs', 'base64', 'binascii', 'pickle', 'shelve',
                'marshal', 'xml', 'html', 'ast', 'dis', 'inspect', 'traceback',
                'warnings', 'contextlib', 'enum', 'numbers', 'decimal', 'fractions',
                'random', 'statistics', 'operator', 'signal', 'queue', 'heapq',
                'bisect', 'array', 'weakref', 'types', 'pdb', 'profile', 'timeit',
                'platform', 'uuid', 'ctypes', 'concurrent', 'asyncio', 'venv',
            }:
                packages.add(pkg.replace('_', '-'))

        # Check each package against PyPI (limit to 5 to avoid rate limiting)
        for pkg in list(packages)[:5]:
            try:
                url = f"https://pypi.org/pypi/{pkg}/json"
                urlopen(url, timeout=5)
            except HTTPError as e:
                if e.code == 404:
                    findings.append(OutputFinding(
                        scanner="packages",
                        severity="HIGH",
                        message=f"Hallucinated Python package: '{pkg}' does not exist on PyPI",
                        evidence=f"pip install {pkg}",
                        weight=7,
                    ))
            except (URLError, OSError):
                pass  # Network error — skip, don't flag

        return findings
