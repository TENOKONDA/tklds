"""Compatibility shim for legacy packaging tools.

All project metadata, dependencies, package discovery rules, and version
information are defined in ``pyproject.toml``.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
