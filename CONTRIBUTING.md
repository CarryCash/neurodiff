# Contributing to NeuroDiff

First off, thank you for considering contributing to NeuroDiff!

## Running Tests
We use `pytest`. To run the test suite:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Adding a New Engine
NeuroDiff has a pluggable architecture. To add a new engine:
1. Create `neurodiff/engines/your_engine.py`
2. Implement your logic returning a list of `YourEngineFinding` dataclasses
3. Update `neurodiff/output/reporter.py` to format your findings
4. Update `neurodiff/cli/main.py` to execute your engine in the pipeline

## Code Style Guidelines
- Use **Black** for formatting.
- Follow **PEP 8**.
- Use **Type hints** for all function signatures.
- Write docstrings for all modules, classes, and public functions using Google format.
