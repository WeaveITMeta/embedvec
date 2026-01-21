# Contributing to embedvec

Thank you for your interest in contributing to embedvec!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/embedvec.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Run benchmarks: `cargo bench`
7. Submit a pull request

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/embedvec/embedvec.git
cd embedvec
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build Python bindings (requires maturin)
pip install maturin
cd embedvec-py
maturin develop
```

## Code Style

- Follow Rust standard formatting: `cargo fmt`
- Run clippy: `cargo clippy`
- Add documentation for public APIs
- Include tests for new functionality

## Testing Requirements

Before submitting a PR:

1. All existing tests pass: `cargo test`
2. New features have tests
3. Benchmarks don't regress significantly
4. Documentation is updated

## Pull Request Process

1. Update README.md if needed
2. Update CHANGELOG.md
3. Ensure CI passes
4. Request review from maintainers

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Include Rust/Python version
- Include OS information

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
