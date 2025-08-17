# my-package Documentation

## Overview

`my-package` is a minimal Python package demonstrating modern Python packaging best practices with a clean src layout.

## Installation

Install from PyPI:

```bash
pip install my-package
```

Install for development:

```bash
git clone https://github.com/username/my-package.git
cd my-package
pip install -e .[dev]
```

## Quick Usage

```python
from my_package import hello
from my_package.utils import slugify
from my_package.subpackage import add, safe_div

# Basic greeting
print(hello("World"))  # Hello, World!

# Text utilities  
print(slugify("Hello World!"))  # hello-world

# Math helpers
print(add(2, 3))  # 5
print(safe_div(10, 2))  # 5.0
print(safe_div(10, 0))  # None
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=my_package
```

## Building and Publishing

Build the package:

```bash
python -m build
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```