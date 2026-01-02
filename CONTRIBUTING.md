# Contributing to DeepSafaitic

Thank you for your interest in contributing to DeepSafaitic! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/DeepSafaitic.git`
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install in development mode: `pip install -e .`
3. Run tests to ensure everything works: `python -m pytest tests/`

## Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where possible
- Add docstrings to all functions and classes (Google style)
- Maximum line length: 100 characters
- Use meaningful variable and function names

## Testing

- Write tests for new features
- Ensure all tests pass: `python -m pytest tests/`
- Aim for good test coverage

## Submitting Changes

1. **Update Documentation**: Update README.md or relevant docs if needed
2. **Write Tests**: Add tests for new functionality
3. **Commit Messages**: Use clear, descriptive commit messages
4. **Pull Request**: 
   - Fill out the PR template
   - Reference any related issues
   - Add screenshots for UI changes
   - Ensure CI checks pass

## Project Structure

```
DeepSafaitic/
├── src/deepsafaitic/    # Core package modules
├── scripts/             # Executable scripts
│   ├── utils/           # Utility scripts
│   └── legacy/           # Deprecated scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Test files
├── data/                # Data directories (gitignored)
├── outputs/             # Output directories (gitignored)
└── docs/                # Documentation
```

## Questions?

Feel free to open an issue for questions or discussions about contributions.

