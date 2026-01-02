# Contributing to DeepSafaitic

Thank you for your interest in contributing to DeepSafaitic!

## Project Structure

```
DeepSafaitic/
├── src/deepsafaitic/    # Core package modules
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Test files
├── data/                # Data directories (gitignored)
├── outputs/             # Output directories (gitignored)
└── docs/                # Documentation
```

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install in development mode: `pip install -e .`

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all functions and classes

## Testing

Run tests before submitting:
```bash
python -m pytest tests/
```

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request with a clear description

