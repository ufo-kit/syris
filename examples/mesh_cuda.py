# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
# SPDX-License-Identifier: LGPL-3.0-or-later

"""CUDA entry point for the shared mesh projection example."""

from .mesh import main as _main
from .mesh import parse_args as _parse_args


def main(argv=None):
    return _main(argv, backend="cuda")


def parse_args(argv=None):
    return _parse_args(argv, backend="cuda")


if __name__ == "__main__":
    main()
