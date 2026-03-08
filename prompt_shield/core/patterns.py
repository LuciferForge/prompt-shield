"""Injection pattern registry for prompt-shield.

Each pattern has:
  name:     unique identifier
  pattern:  regex (case-insensitive unless noted)
  weight:   1-10, contributes to risk_score
  category: role_override | jailbreak | exfiltration | manipulation | encoding
            | multilingual | tool_use | pii | claude_code

Categories added in v0.2.0:
  multilingual  — injection attempts in non-English languages + cross-lingual
  tool_use      — dangerous shell/agent tool invocations (MCP, Claude Code)
  pii           — personally identifiable information & secrets detection
  claude_code   — Claude Code / LLM-specific injection vectors
"""

import re

PATTERNS = [
    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL 22 PATTERNS (v0.1.0) — DO NOT MODIFY
    # ══════════════════════════════════════════════════════════════════════════

    # ── Role override ────────────────────────────────────────────────────────
    {
        "name": "ignore_instructions",
        "pattern": r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
        "weight": 9,
        "category": "role_override",
    },
    {
        "name": "disregard_training",
        "pattern": r"disregard\s+(your\s+)?(training|guidelines|rules|constraints|prompt)",
        "weight": 8,
        "category": "role_override",
    },
    {
        "name": "you_are_now",
        "pattern": r"you\s+are\s+now\s+(?!an?\s+assistant|helpful|a\s+language)",
        "weight": 7,
        "category": "role_override",
    },
    {
        "name": "new_instructions",
        "pattern": r"(your\s+new\s+instructions?|from\s+now\s+on\s+you\s+(are|will|must))",
        "weight": 7,
        "category": "role_override",
    },
    {
        "name": "override_system",
        "pattern": r"(override|bypass|circumvent)\s+(the\s+)?(system|safety|rules?|filter|restriction)",
        "weight": 8,
        "category": "role_override",
    },

    # ── Jailbreak ────────────────────────────────────────────────────────────
    {
        "name": "dan_jailbreak",
        "pattern": r"\bDAN\b|\bdo\s+anything\s+now\b",
        "weight": 10,
        "category": "jailbreak",
    },
    {
        "name": "act_as",
        "pattern": r"\bact\s+as\s+(if\s+you\s+are|a|an)\b(?!\s*(helpful|friendly|assistant))",
        "weight": 6,
        "category": "jailbreak",
    },
    {
        "name": "pretend_you_are",
        "pattern": r"\bpretend\s+(you\s+are|to\s+be)\b",
        "weight": 6,
        "category": "jailbreak",
    },
    {
        "name": "no_restrictions",
        "pattern": r"(without\s+(any\s+)?(restrictions?|limits?|filters?|censorship)|unrestricted\s+mode)",
        "weight": 8,
        "category": "jailbreak",
    },
    {
        "name": "true_self",
        "pattern": r"(your\s+true\s+self|your\s+real\s+personality|without\s+your\s+training)",
        "weight": 7,
        "category": "jailbreak",
    },
    {
        "name": "developer_mode",
        "pattern": r"(developer\s+mode|jailbreak\s+mode|god\s+mode|unrestricted\s+mode)",
        "weight": 9,
        "category": "jailbreak",
    },

    # ── Data exfiltration ────────────────────────────────────────────────────
    {
        "name": "print_system_prompt",
        "pattern": r"(print|repeat|output|show|reveal|display|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions?|context|training\s+data)",
        "weight": 9,
        "category": "exfiltration",
    },
    {
        "name": "repeat_everything_above",
        "pattern": r"repeat\s+(everything|all(\s+text)?)\s+(above|before|prior)",
        "weight": 8,
        "category": "exfiltration",
    },
    {
        "name": "what_were_your_instructions",
        "pattern": r"what\s+(were|are)\s+your\s+(instructions?|rules?|guidelines?|prompt)",
        "weight": 7,
        "category": "exfiltration",
    },
    {
        "name": "summarize_above",
        "pattern": r"summarize\s+(the\s+)?(above|previous|prior|everything\s+above)",
        "weight": 4,
        "category": "exfiltration",
    },

    # ── Manipulation ─────────────────────────────────────────────────────────
    {
        "name": "trusted_source_claim",
        "pattern": r"(this\s+is\s+a?\s*trusted|from\s+a?\s*trusted|authorized\s+by|approved\s+by\s+anthropic|from\s+openai)",
        "weight": 8,
        "category": "manipulation",
    },
    {
        "name": "developer_wants",
        "pattern": r"(your\s+developer|your\s+creator|your\s+owner|anthropic\s+says?|openai\s+says?)\s+(wants?|requires?|says?|told\s+you)",
        "weight": 9,
        "category": "manipulation",
    },
    {
        "name": "for_research",
        "pattern": r"(for\s+(research|educational|academic|testing)\s+purposes?|this\s+is\s+just\s+a\s+test|hypothetically\s+speaking)",
        "weight": 3,
        "category": "manipulation",
    },
    {
        "name": "token_smuggling",
        "pattern": r"(t+o+k+e+n|i+g+n+o+r+e|s+y+s+t+e+m)\s*p+r+o+m+p+t",
        "weight": 7,
        "category": "manipulation",
    },

    # ── Encoding / obfuscation ────────────────────────────────────────────────
    {
        "name": "base64_injection",
        "pattern": r"(base64|b64)\s*(decode|encoded|string)",
        "weight": 6,
        "category": "encoding",
    },
    {
        "name": "unicode_smuggling",
        "pattern": r"[\u200b\u200c\u200d\u202a-\u202e\u2060-\u2064\ufeff]",
        "weight": 8,
        "category": "encoding",
    },
    {
        "name": "rot13_reference",
        "pattern": r"\brot\s*13\b",
        "weight": 5,
        "category": "encoding",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NEW PATTERNS (v0.2.0)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Multilingual injection ───────────────────────────────────────────────
    # Chinese (Simplified)
    {
        "name": "ml_zh_ignore_instructions",
        "pattern": r"忽略之前的指令|忽略(以前|之前|上面)的(指示|说明|提示)",
        "weight": 9,
        "category": "multilingual",
        "description": "Chinese: ignore previous instructions",
    },
    {
        "name": "ml_zh_forget_rules",
        "pattern": r"忘记你的规则|忘(掉|记)(你的|所有)(规则|限制|指令)",
        "weight": 8,
        "category": "multilingual",
        "description": "Chinese: forget your rules",
    },
    {
        "name": "ml_zh_new_role",
        "pattern": r"你现在是|从现在开始你(是|要|必须)",
        "weight": 7,
        "category": "multilingual",
        "description": "Chinese: you are now / from now on you are",
    },
    # Spanish
    {
        "name": "ml_es_ignore_instructions",
        "pattern": r"ignora\s+(las\s+)?(instrucciones|directivas)\s+(anteriores|previas)",
        "weight": 9,
        "category": "multilingual",
        "description": "Spanish: ignore previous instructions",
    },
    {
        "name": "ml_es_forget_rules",
        "pattern": r"olvida\s+(tus\s+)?(reglas|restricciones|instrucciones|limitaciones)",
        "weight": 8,
        "category": "multilingual",
        "description": "Spanish: forget your rules",
    },
    {
        "name": "ml_es_new_role",
        "pattern": r"ahora\s+eres|a\s+partir\s+de\s+ahora\s+(eres|debes|vas\s+a)",
        "weight": 7,
        "category": "multilingual",
        "description": "Spanish: you are now / from now on you are",
    },
    # French
    {
        "name": "ml_fr_ignore_instructions",
        "pattern": r"ignore[sz]?\s+(les\s+)?(instructions?|directives?)\s+(pr[eé]c[eé]dentes?|ant[eé]rieures?)",
        "weight": 9,
        "category": "multilingual",
        "description": "French: ignore previous instructions",
    },
    {
        "name": "ml_fr_forget_rules",
        "pattern": r"oublie[sz]?\s+(tes|vos|les)\s+(r[eè]gles?|restrictions?|instructions?|limites?)",
        "weight": 8,
        "category": "multilingual",
        "description": "French: forget your rules",
    },
    # German
    {
        "name": "ml_de_ignore_instructions",
        "pattern": r"ignorier(e|en)?\s+(die\s+)?(vorherigen?|fr[uü]heren?|obigen?)\s+(Anweisungen?|Instruktionen?|Regeln?)",
        "weight": 9,
        "category": "multilingual",
        "description": "German: ignore previous instructions",
    },
    {
        "name": "ml_de_forget_rules",
        "pattern": r"vergiss\s+(deine|alle)\s+(Regeln?|Einschr[aä]nkungen?|Anweisungen?)",
        "weight": 8,
        "category": "multilingual",
        "description": "German: forget your rules",
    },
    # Japanese
    {
        "name": "ml_ja_ignore_instructions",
        "pattern": r"以前の指示を無視(して|しろ|せよ)|前の(指示|命令|ルール)を(無視|忘れ)",
        "weight": 9,
        "category": "multilingual",
        "description": "Japanese: ignore previous instructions",
    },
    {
        "name": "ml_ja_forget_rules",
        "pattern": r"ルールを忘れて|ルールを(無視|破棄)(して|しろ)|制約を(忘れ|無視)",
        "weight": 8,
        "category": "multilingual",
        "description": "Japanese: forget rules",
    },
    # Hindi
    {
        "name": "ml_hi_ignore_instructions",
        "pattern": r"पिछले\s+निर्देशों\s+को\s+अनदेखा\s+करें|पिछली\s+हिदायतों?\s+को\s+(भूल|अनदेखा)\s+(जाओ|करो|करें)",
        "weight": 9,
        "category": "multilingual",
        "description": "Hindi: ignore previous instructions",
    },
    {
        "name": "ml_hi_forget_rules",
        "pattern": r"अपने\s+नियम(ों)?\s+को\s+(भूल|अनदेखा)|सभी\s+नियम(ों)?\s+को\s+(भूल|तोड़)",
        "weight": 8,
        "category": "multilingual",
        "description": "Hindi: forget your rules",
    },
    # Cross-lingual detection: English injection keywords embedded in non-Latin text
    {
        "name": "ml_cross_lingual_inject",
        "pattern": r"(?:[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0900-\u097f\u0600-\u06ff\u0400-\u04ff]{3,}\s*)"
                   r"(?:ignore|forget|override|bypass|system\s*prompt|new\s*instructions?)"
                   r"(?:\s*[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0900-\u097f\u0600-\u06ff\u0400-\u04ff]{2,})",
        "weight": 8,
        "category": "multilingual",
        "description": "Cross-lingual: English injection keywords sandwiched in non-Latin script",
    },

    # ── Tool-use / agentic patterns ──────────────────────────────────────────
    {
        "name": "tool_shell_destructive",
        "pattern": r"\brm\s+-[rf]{1,2}\b|\brm\s+--no-preserve-root\b",
        "weight": 10,
        "category": "tool_use",
        "description": "Destructive rm commands",
    },
    {
        "name": "tool_shell_pipe_exec",
        "pattern": r"(curl|wget)\s+[^\|]*\|\s*(ba)?sh\b",
        "weight": 10,
        "category": "tool_use",
        "description": "Pipe remote content to shell execution",
    },
    {
        "name": "tool_shell_chmod_dangerous",
        "pattern": r"\bchmod\s+(777|[+]s|u[+]s|g[+]s)\b",
        "weight": 8,
        "category": "tool_use",
        "description": "Dangerous chmod: world-writable or setuid",
    },
    {
        "name": "tool_shell_command_chain",
        "pattern": r";\s*(rm|curl|wget|nc|ncat|bash|sh|python|perl|ruby)\b",
        "weight": 7,
        "category": "tool_use",
        "description": "Command chaining with dangerous executables",
    },
    {
        "name": "tool_data_exfil_curl",
        "pattern": r"\bcurl\b[^|]*\s+(-d|--data|--data-binary|-F|--form)\s",
        "weight": 9,
        "category": "tool_use",
        "description": "Curl sending data to external URL",
    },
    {
        "name": "tool_data_exfil_netcat",
        "pattern": r"\b(nc|ncat|netcat)\s+(-e\s+)?[\w.-]+\s+\d{1,5}\b",
        "weight": 9,
        "category": "tool_use",
        "description": "Netcat connection to external host",
    },
    {
        "name": "tool_sensitive_file_etc",
        "pattern": r"(?:/etc/(?:passwd|shadow|sudoers|hosts)|/etc/ssh/)",
        "weight": 8,
        "category": "tool_use",
        "description": "Access to sensitive system files",
    },
    {
        "name": "tool_sensitive_file_keys",
        "pattern": r"(?:~/\.ssh/|\.ssh/id_|\.ssh/authorized_keys|\.gnupg/|\.aws/credentials)",
        "weight": 9,
        "category": "tool_use",
        "description": "Access to SSH keys, GPG keys, or AWS credentials",
    },
    {
        "name": "tool_sensitive_file_env",
        "pattern": r"(?:^|\s|/)(\.env|\.env\.local|\.env\.production|credentials\.json|secrets\.ya?ml)\b",
        "weight": 8,
        "category": "tool_use",
        "description": "Access to environment/secrets files",
    },
    {
        "name": "tool_privilege_escalation",
        "pattern": r"\b(sudo\s+(?!apt|dnf|yum|brew)|su\s+root|su\s+-\s)\b",
        "weight": 8,
        "category": "tool_use",
        "description": "Privilege escalation (sudo for non-package-manager, su root)",
    },
    {
        "name": "tool_crypto_private_key",
        "pattern": r"(-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----|openssl\s+(genrsa|genpkey|req\s+-newkey))",
        "weight": 9,
        "category": "tool_use",
        "description": "Private key content or generation",
    },
    {
        "name": "tool_reverse_shell",
        "pattern": r"(bash\s+-i\s+>(&|&\s*)/dev/tcp/|/dev/tcp/[\d.]+/\d+|mkfifo\s+/tmp/)",
        "weight": 10,
        "category": "tool_use",
        "description": "Reverse shell patterns",
    },
    {
        "name": "tool_wget_exec",
        "pattern": r"\bwget\b[^;|&]*-O\s*-\s*\|\s*(ba)?sh\b",
        "weight": 10,
        "category": "tool_use",
        "description": "Wget piped to shell",
    },
    {
        "name": "tool_python_exec",
        "pattern": r"python[23]?\s+-c\s+['\"].*(?:import\s+(?:os|subprocess|shutil)|exec|eval)\b",
        "weight": 7,
        "category": "tool_use",
        "description": "Python one-liner with dangerous imports/builtins",
    },

    # ── PII detection ────────────────────────────────────────────────────────
    {
        "name": "pii_email_address",
        "pattern": r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        "weight": 3,
        "category": "pii",
        "description": "Email address",
    },
    {
        "name": "pii_credit_card",
        "pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "weight": 8,
        "category": "pii",
        "description": "Credit card number (Visa, MC, Amex, Discover formats)",
    },
    {
        "name": "pii_ssn",
        "pattern": r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        "weight": 9,
        "category": "pii",
        "description": "US Social Security Number (with basic validity checks)",
    },
    {
        "name": "pii_phone_international",
        "pattern": r"(?<!\d)\+?1?\s*[-.(]?\s*[2-9]\d{2}\s*[-.)]\s*\d{3}\s*[-.]?\s*\d{4}(?!\d)",
        "weight": 2,
        "category": "pii",
        "description": "Phone number (US/international formats)",
    },
    {
        "name": "pii_api_key_openai",
        "pattern": r"\bsk-[a-zA-Z0-9]{20,}\b",
        "weight": 9,
        "category": "pii",
        "description": "OpenAI API key pattern (sk-...)",
    },
    {
        "name": "pii_api_key_aws",
        "pattern": r"\bAKIA[0-9A-Z]{16}\b",
        "weight": 9,
        "category": "pii",
        "description": "AWS Access Key ID",
    },
    {
        "name": "pii_api_key_github",
        "pattern": r"\b(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}\b",
        "weight": 9,
        "category": "pii",
        "description": "GitHub personal access token or OAuth token",
    },
    {
        "name": "pii_api_key_stripe",
        "pattern": r"\b[sr]k_(live|test)_[a-zA-Z0-9]{20,}\b",
        "weight": 9,
        "category": "pii",
        "description": "Stripe API key",
    },
    {
        "name": "pii_api_key_slack",
        "pattern": r"\bxox[bprs]-[a-zA-Z0-9\-]{10,}\b",
        "weight": 9,
        "category": "pii",
        "description": "Slack bot/user/app token",
    },
    {
        "name": "pii_private_key_pem",
        "pattern": r"-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
        "weight": 10,
        "category": "pii",
        "description": "PEM-encoded private key",
    },

    # ── Claude Code-specific patterns ────────────────────────────────────────
    {
        "name": "cc_html_comment_injection",
        "pattern": r"<!--\s*(system|instruction|prompt|role|admin)\s*:.*?-->",
        "weight": 8,
        "category": "claude_code",
        "description": "Injection hidden in HTML comments",
    },
    {
        "name": "cc_authority_impersonation",
        "pattern": r"(ADMIN\s+MESSAGE|SYSTEM\s+(OVERRIDE|MESSAGE|NOTICE)|MESSAGE\s+FROM\s+(ANTHROPIC|OPENAI|THE\s+DEVELOPER))",
        "weight": 9,
        "category": "claude_code",
        "description": "Impersonating authority (admin/system messages)",
    },
    {
        "name": "cc_zero_width_chars",
        "pattern": r"[\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064\ufeff]{2,}",
        "weight": 9,
        "category": "claude_code",
        "description": "Multiple zero-width characters (smuggling attempt)",
    },
    {
        "name": "cc_config_file_injection",
        "pattern": r"(modify|edit|write|overwrite|change|update)\s+(the\s+)?(CLAUDE\.md|\.claude|\.cursorrules|\.github/copilot)",
        "weight": 8,
        "category": "claude_code",
        "description": "Attempts to modify AI assistant config files",
    },
    {
        "name": "cc_markdown_injection",
        "pattern": r"\[.*?\]\((?:javascript|data|vbscript):",
        "weight": 8,
        "category": "claude_code",
        "description": "Markdown link with dangerous protocol (XSS vector)",
    },
    {
        "name": "cc_prompt_delimiter_fake",
        "pattern": r"(```system|<\|system\|>|<\|im_start\|>system|\[INST\]|\[/INST\]|<<SYS>>)",
        "weight": 9,
        "category": "claude_code",
        "description": "Fake prompt delimiters to inject system-level instructions",
    },
    {
        "name": "cc_invisible_text_direction",
        "pattern": r"[\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]{2,}",
        "weight": 8,
        "category": "claude_code",
        "description": "Multiple Unicode bidirectional control characters",
    },
    {
        "name": "cc_tool_result_fake",
        "pattern": r"(<tool_result>|<function_result>|<output>.*</output>)",
        "weight": 8,
        "category": "claude_code",
        "description": "Fake tool/function result tags to inject trusted content",
    },
]


def _compile_patterns():
    compiled = []
    for p in PATTERNS:
        compiled.append({
            **p,
            "_regex": re.compile(p["pattern"], re.IGNORECASE | re.UNICODE),
        })
    return compiled


COMPILED_PATTERNS = _compile_patterns()


# Categories available for filtering
CATEGORIES = {
    "role_override", "jailbreak", "exfiltration", "manipulation", "encoding",
    "multilingual", "tool_use", "pii", "claude_code",
}

# PII category patterns are opt-in — users may want to detect PII
# without blocking. The scanner supports category-based filtering.
PII_CATEGORY = "pii"


def score_to_severity(score: int) -> str:
    if score == 0:
        return "SAFE"
    if score <= 3:
        return "LOW"
    if score <= 6:
        return "MEDIUM"
    if score <= 9:
        return "HIGH"
    return "CRITICAL"
