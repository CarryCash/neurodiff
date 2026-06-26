# NeuroDiff

**NeuroDiff** is an open-source, semantic git diff CLI tool designed specifically for developers using AI code assistants. When an AI generates hundreds of lines of code, standard line-by-line diff tools cause cognitive fatigue. NeuroDiff parses the AST, analyzes architectural impact, detects security vulnerabilities, finds duplicated code, and runs a deep LLM analysis to give you a single "Safe to Merge" verdict.

## Why NeuroDiff?
AI assistants inject code fast. It is extremely difficult to track *what logically changed*, if it violates your architecture layers, or if it hallucinates existing functions instead of reusing them. NeuroDiff solves this by combining static analysis with a final LLM reasoning step.

## Installation

The recommended way to install Python CLI tools globally without polluting your system or dealing with virtual environments is using `pipx`.

```bash
# If you don't have pipx, install it first:
# Windows (PowerShell): scoop install pipx
# macOS: brew install pipx
# Linux: sudo apt install pipx

# Install NeuroDiff globally
pipx install path/to/neurodiff/folder

# Now the `neurodiff` command is available everywhere!
```

## Setup API Keys

NeuroDiff uses LLMs for its final deep reasoning phase. You can securely save your API key (it gets stored in `~/.neurodiff/config.json`).

```bash
neurodiff config set-api-key gemini
# or
neurodiff config set-api-key claude
```
*(The input is hidden for security).*

## Workflow: Validating AI-Generated Code

1. You ask your AI editor (Cursor, Copilot, Cline) to implement a feature.
2. The AI writes 500 lines of code across 8 files.
3. **Before you commit or merge**, you want to know what it did:

```bash
# Analyze the current uncommitted changes vs HEAD
neurodiff analyze HEAD "" --llm
```

The output will show:
- **Semantic Changes:** "Added 3 functions, modified 1 class."
- **Security:** "Detected hardcoded secret in auth.py."
- **Architecture:** "Layer Violation: core imports engines."
- **Duplication:** "ProcessPayment is 95% similar to an existing function."
- **LLM Verdict:** "⛔ NOT SAFE TO MERGE. Circular dependency detected."

## Commands

- `neurodiff analyze <BASE> <HEAD> [--llm]`: Run the full analysis suite.
- `neurodiff init-arch-rules`: Generate a default architecture rules file.
- `neurodiff config set-api-key <provider>`: Securely store your API key.

## Caches
NeuroDiff ignores and suppresses typical python caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`) by default through its `.gitignore` to keep your workspace clean.