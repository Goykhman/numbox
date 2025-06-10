# numbox

A toolbox of low-level utilities for working with [numba](https://numba.pydata.org/).

Documentation available at [numbox](https://goykhman.github.io/numbox)

## Tools

- **Any**: A lightweight structure that wraps any value into the same type leveraging a variant of the type erasure technique.
- **Node** A lightweight JIT-compatible graph node with type-erased dependencies in a uniform dynamically-sized vector (numba List). 
- **proxy**: Create a proxy for a decorated function with specified signatures, enabling efficient JIT compilation and caching.
- **Work**: JIT-compatible unit of calculation work with dependencies, inner states, and custom calculation.

## Installation

```bash
pip install numbox