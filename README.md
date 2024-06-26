# python-project-template

This repository provides a template for Python 3.12 projects based on [Poetry](https://github.com/python-poetry/poetry).

* [click](https://github.com/pallets/click) is used for building command line interfaces
* [loguru](https://github.com/Delgan/loguru) is used for structured logging
* [mypy](https://mypy.readthedocs.io/en/stable) is used for type checking
* [pytest](https://docs.pytest.org/en/8.2.x) is used for writing tests
* [ruff](https://docs.astral.sh/ruff) is used for linting and formatting

The repository also provides a [GitHub Actions](https://docs.github.com/en/actions) workflow that automatically builds, lints, type checks, and tests the project on every commit, pull request, and direct invocation of the workflow.

Finally, this repository provides a [DevContainer](https://containers.dev) to allow you to get started with development without installing anything to your machine (other than Docker and VSCode).
To open this project up inside a DevContainer, bring up the command palette in VSCode (CTRL-Shift-P or Command-Shift-P) and select "rebuild and reopen project in DevContainer".

## Usage

To setup the virtual environment and install your project, run:

```shell
make install
```

To run the linter:

```shell
make lint
```

To run the type checker:

```shell
make type
```

To run the test suite:

```shell
make test
```

To lint, type check, and test in a single command, run:

```shell
make check
```
