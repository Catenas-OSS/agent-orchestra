# Contributing to Agent Orchestra

First off, thank you for considering contributing to Agent Orchestra! It's people like you that make open source such a great community.

We welcome any and all contributions, from bug reports to feature requests and code contributions.

## How to Contribute

-   **Reporting Bugs:** If you find a bug, please open an issue on our [GitHub Issues](https://github.com/agent-orchestra/agent-orchestra/issues) page. Please include as much detail as possible, including steps to reproduce the bug.
-   **Suggesting Enhancements:** If you have an idea for a new feature or an improvement to an existing one, please open an issue to discuss it. This allows us to coordinate our efforts and avoid duplicating work.
-   **Pull Requests:** If you're ready to contribute code, please submit a pull request. We follow a standard pull request process.

## Getting Started

To get started with development, you'll need to set up your environment.

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/agent-orchestra.git
    cd agent-orchestra
    ```
3.  **Install the dependencies**, including the development dependencies:
    ```bash
    pip install -e .[dev]
    ```

## Coding Standards

We strive to maintain a high standard of code quality. Please follow these guidelines when contributing code:

-   **Style:** We follow the [Black](https://github.com/psf/black) code style. Please format your code with Black before submitting a pull request.
-   **Typing:** We use type hints extensively. Please add type hints to all new code and ensure that your code passes `mypy` checks.
-   **Documentation:** All new features should be documented with docstrings. We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings.
-   **Testing:** All new features should be accompanied by tests. We use `pytest` for testing.

### Code Quality Checks

Before submitting a pull request, please run the following checks locally:

```bash
# Type checking
- mypy src/

# Linting
- ruff check src/

# Tests
- pytest
```

## Pull Request Process

1.  **Ensure all tests pass** and that your code is formatted and linted correctly.
2.  **Create a pull request** from your fork to the `main` branch of the main repository.
3.  **Provide a clear description** of the changes you've made and why you've made them.
4.  **We will review your pull request** as soon as possible. We may ask for changes or provide feedback.

## Code of Conduct

We have a [Code of Conduct](CODE_OF_CONDUCT.md) that we expect all contributors to follow. Please make sure you are familiar with it.

Thank you for your contributions!
