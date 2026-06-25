# NeuroDiff - Semantic Git Diff Analysis CLI Tool

## Complete Project Specification

### Overview

NeuroDiff is a complete open-source CLI tool for semantic git diff analysis. It goes beyond traditional git diffs to provide intelligent analysis of code changes at the semantic level, detecting functional changes, security vulnerabilities, and code duplication patterns.

### Project Goals

- **Semantic Analysis**: Extract meaningful code changes (functions, classes, imports)
- **Security Analysis**: Detect hardcoded secrets, injection vulnerabilities, and other security issues
- **Code Duplication**: Identify similar code patterns across the codebase
- **Risk Assessment**: Provide comprehensive risk scoring and actionable insights

---

## Architecture Overview

### Core Components

#### 1. **Git Parser** (`core/git_parser.py`)
- **Purpose**: Extract and parse diffs from Git repositories
- **Key Class**: `GitParser`
  - Takes repo path as input
  - Extracts file diffs between two Git refs (commits, branches, tags)
  - Detects language from file extension
  - Returns `FileDiff` objects containing:
    - `path`: File path relative to repo root
    - `language`: Detected language (python, javascript, typescript, java, etc.)
    - `content_before`: Full file content before changes
    - `content_after`: Full file content after changes
    - `raw_diff`: Raw unified diff output from git

#### 2. **Semantic Events** (`core/semantic_events.py`)
- **Purpose**: Define semantic code change events
- **Dataclasses**:
  - `FunctionAdded`: New function introduced
  - `FunctionModified`: Existing function changed (tracks complexity changes)
  - `FunctionRemoved`: Function deleted
  - `ClassAdded`: New class introduced
  - `ClassModified`: Existing class changed (tracks method changes)
  - `ImportAdded`: New import statement
  - `ImportRemoved`: Removed import statement
- **Union Type**: `SemanticEvent` - union of all event types
- **Exception**: `NeuroDiffError` - base exception for all NeuroDiff errors

#### 3. **AST Engine** (`core/ast_engine.py`)
- **Purpose**: Analyze code at the AST level to extract semantic events
- **Key Class**: `ASTEngine`
  - Uses tree-sitter for language-agnostic parsing
  - Supports: Python, JavaScript, TypeScript
  - Extracts functions, classes, and imports
  - Calculates cyclomatic complexity for functions
  
- **Cyclomatic Complexity Calculation**:
  - Formula: `1 + count(if, elif, else, for, while, except, and, or)`
  - Detects decision points in control flow
  - Used to track complexity changes in modified functions

#### 4. **Security Engine** (`engines/security_engine.py`)
- **Purpose**: Analyze code for security vulnerabilities
- **Integration**:
  - Primary: Semgrep integration (subprocess-based)
  - Fallback: Regex pattern matching for common issues
- **Patterns Detected**:
  - Hardcoded secrets (passwords, API keys, tokens)
  - SQL injection vulnerabilities
  - Command injection vulnerabilities
  - Hardcoded IP addresses
- **Output**: `SecurityFinding` dataclass with:
  - `rule_id`: Identifier for the finding
  - `title`: Human-readable title
  - `severity`: CRITICAL, HIGH, MEDIUM, LOW, INFO
  - `file_path`, `line`: Location of issue
  - `code_snippet`: Code excerpt showing the problem
  - `recommendation`: Suggested fix

#### 5. **Duplication Engine** (`engines/duplication_engine.py`)
- **Purpose**: Detect similar/duplicate code patterns
- **Implementation**:
  - Uses ChromaDB (vector database) at `~/.neurodiff/chroma_db/`
  - Embeddings via sentence-transformers (all-MiniLM-L6-v2 model)
  - Cosine similarity threshold: 0.80
- **Output**: `DuplicationFinding` dataclass with:
  - `source_file`, `target_file`: Files being compared
  - `similarity`: Float 0-1 indicating similarity
  - `severity`: HIGH if similarity > 0.95, MEDIUM otherwise

#### 6. **Reporter** (`output/reporter.py`)
- **Purpose**: Generate rich-formatted analysis reports
- **Output Sections**:
  - Header panel (repo path, refs, file count)
  - Semantic Summary table (event counts by type)
  - File Details (icons: ➕ added, ✏️ modified, ❌ removed)
  - Security Findings table (sorted by severity)
  - Duplication Warnings table
  - Risk Score badge (LOW, MEDIUM, HIGH, CRITICAL)
- **Framework**: Rich library for terminal rendering

#### 7. **CLI** (`cli/main.py`)
- **Purpose**: Command-line interface for analysis
- **Framework**: Typer
- **Commands**:
  - `neurodiff analyze <base_ref> <head_ref>` - Main command
    - `--repo-path, -r`: Path to repository (default: current dir)
    - `--format, -f`: Output format (terminal|json, default: terminal)
    - `--lang, -l`: Language filter (python|javascript|typescript|auto, default: auto)
- **Output Modes**:
  - Terminal: Rich-formatted report
  - JSON: Structured JSON output of all findings

---

## Directory Structure

```
neurodiff/
├── cli/
│   ├── __init__.py
│   └── main.py              # Typer CLI implementation
├── core/
│   ├── __init__.py
│   ├── semantic_events.py   # Event dataclasses & NeuroDiffError
│   ├── git_parser.py        # GitParser & FileDiff
│   └── ast_engine.py        # ASTEngine with tree-sitter
├── engines/
│   ├── __init__.py
│   ├── security_engine.py   # SecurityEngine & SecurityFinding
│   └── duplication_engine.py # DuplicationEngine & DuplicationFinding
├── output/
│   ├── __init__.py
│   └── reporter.py          # Rich-based Reporter
├── tests/
│   ├── __init__.py
│   ├── test_ast_engine.py
│   ├── test_security_engine.py
│   ├── test_duplication_engine.py
│   └── test_git_parser.py
├── pyproject.toml           # Project configuration & dependencies
└── COPILOT_INSTRUCTIONS.md  # This file
```

---

## Implementation Details

### Dependencies

**Core Dependencies**:
- `typer>=0.12` - CLI framework
- `rich>=13` - Terminal output formatting
- `tree-sitter>=0.21` - AST parsing
- `tree-sitter-languages>=1.10` - Language support for tree-sitter

**Analysis Dependencies**:
- `chromadb>=0.5` - Vector database for duplication detection
- `sentence-transformers>=3.0` - Embedding model (all-MiniLM-L6-v2)
- `semgrep>=1.70` - Security analysis

**Development**:
- `pytest>=7.0` - Testing framework

### Key Design Principles

1. **Full Type Hints**: All functions have complete type annotations
2. **Graceful Degradation**: 
   - Semgrep unavailable? Fall back to regex patterns
   - ChromaDB unavailable? Skip duplication analysis
   - Tree-sitter parsing fails? Return empty events
3. **No Global State**: All data passed via function arguments/returns
4. **pathlib.Path**: Use throughout instead of string paths
5. **Error Handling**: Centralized via `NeuroDiffError` exception

### Cyclomatic Complexity

Calculated using a simple token-based counter that increments for:
- Control flow keywords: `if`, `elif`, `for`, `while`, `except`
- Logical operators: `and`, `or`
- Base value: 1 (always present)
- Formula: `1 + count(decision_points)`

### Security Analysis

**Semgrep Integration**:
- Executed as subprocess: `semgrep --json <file>`
- Graceful degradation if unavailable
- Results parsed from JSON output
- Severity mapping: ERROR→CRITICAL, WARNING→HIGH, INFO→INFO

**Fallback Patterns**:
- Hardcoded secrets: `(password|secret|api[_-]?key|token) = "[^"]+"`
- SQL injection: `execute|query.*\+`
- Command injection: `subprocess|os\.system.*\+`
- Hardcoded IPs: `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`

### Code Duplication

**ChromaDB Setup**:
- Database location: `~/.neurodiff/chroma_db/`
- Collection name: `code_snippets`
- Space: `cosine` (for cosine similarity)

**Embedding Model**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Similarity calculation: 1 - cosine_distance
- Threshold: 0.80 (80% similarity)

### Test Coverage

**test_ast_engine.py**:
- Cyclomatic complexity calculation
- Function added/modified/removed detection
- Import detection
- Unsupported language handling
- Empty code handling

**test_security_engine.py**:
- Pattern matching for secrets, SQL injection, command injection
- Severity mapping
- Empty and clean code handling
- Finding structure validation

**test_duplication_engine.py**:
- Similarity threshold validation
- ChromaDB initialization
- Empty and single snippet handling
- Duplicate detection
- Snapshot storage

**test_git_parser.py**:
- Repository validation
- Language detection
- Diff extraction
- FileDiff structure validation

---

## Usage Examples

### Basic Analysis
```bash
neurodiff analyze main feature/new-feature
```

### With Repository Path
```bash
neurodiff analyze HEAD~5 HEAD --repo-path /path/to/repo
```

### JSON Output
```bash
neurodiff analyze v1.0 v2.0 --format json
```

### Language-Specific Analysis
```bash
neurodiff analyze main develop --lang python
```

### Full Example Command
```bash
neurodiff analyze \
  --repo-path /home/user/project \
  --format terminal \
  --lang auto \
  main feature/enhancement
```

---

## Error Handling Strategy

1. **Invalid Repository**: Raise `NeuroDiffError` on non-Git directory
2. **Git Command Failures**: Catch `CalledProcessError`, wrap in `NeuroDiffError`
3. **Tree-sitter Parse Failures**: Log warning, return empty events (graceful)
4. **Semgrep Unavailable**: Use regex fallback patterns
5. **ChromaDB Unavailable**: Skip duplication analysis, continue with others
6. **File I/O Errors**: Gracefully skip problematic files

---

## Development Notes

### Adding New Semantic Event Types
1. Add dataclass to `core/semantic_events.py`
2. Add to `SemanticEvent` union type
3. Add extraction method to `ASTEngine`
4. Add test in `tests/test_ast_engine.py`
5. Update `cli/main.py` JSON serialization

### Adding New Security Patterns
1. Add regex pattern to `FALLBACK_PATTERNS` in `SecurityEngine`
2. Set appropriate severity level
3. Add test in `tests/test_security_engine.py`

### Extending Language Support
1. Verify tree-sitter-languages has the language
2. Add to `AST_ENGINE.SUPPORTED_LANGUAGES`
3. Add language-specific node type mappings
4. Add test fixtures with sample code

---

## Performance Considerations

- **Large Diffs**: Process files incrementally, don't load all at once
- **Tree-sitter**: Parsing is fast; caching not needed
- **ChromaDB**: Persistence allows incremental updates
- **Semgrep**: Can be slow on large files; consider timeout
- **Embedding**: Model is lightweight (384-dim vectors)

---

## Future Enhancements

1. **Multi-language Support**: Expand beyond Python/JS/TS
2. **Custom Rules**: User-defined semantic event extraction
3. **Report Formats**: HTML, Markdown, PDF outputs
4. **CI/CD Integration**: Exit codes for CI pipelines
5. **Configuration File**: `.neurodiff.toml` for project settings
6. **Performance Metrics**: Track analysis time, event counts over time
7. **Database Comparison**: Compare current against previous runs

---

## Contributing Guidelines

1. Maintain full type hints
2. Add tests for new features
3. Use `from __future__ import annotations` in all files
4. Follow black/isort formatting
5. Handle errors gracefully with appropriate exceptions
6. Update this specification for major changes

---

## Version History

- **0.1.0**: Initial release
  - Core semantic analysis
  - Security analysis with semgrep fallback
  - Code duplication detection
  - CLI interface with multiple output formats
