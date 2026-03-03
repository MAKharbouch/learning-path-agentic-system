"""Pytest configuration for custom test reporting."""

import pytest


def pytest_runtest_makereport(item, call):
    """Add test description from docstrings to the test report."""
    if call.excinfo is not None:
        return

    # Get the docstring from the test function
    docstring = item.obj.__doc__
    if docstring:
        # Store the description for custom reporting
        item._pytestDescription = docstring.strip()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "description: Add a description to test results"
    )
