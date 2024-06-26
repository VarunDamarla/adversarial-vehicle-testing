from __future__ import annotations

__all__ = ("cli",)

import sys

import click
from loguru import logger

from adversarial_vehicle_testing.example import (
    double,
    triple,
)


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level}:</level> {message}",
        level="TRACE",
        colorize=True,
    )


@click.group()
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="enables verbose logging.",
)
def cli(verbose: bool) -> None:
    if verbose:
        setup_logging()


@cli.command("double")
@click.argument("x", type=int)
def do_double(x: int) -> None:
    y = double(x)
    print(y)


@cli.command("triple")
@click.argument("x", type=int)
def do_triple(x: int) -> None:
    y = triple(x)
    print(y)
