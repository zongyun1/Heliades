"""Test layout: core/ (Helios core port), plantarch/ (plantarchitecture port), sim/ (deformable simulator),
legacy/ (the planttwin prototype), test_parity.py (every parity case against the oracle).

Tests marked ``slow`` (large plants, long growth, the heavy parity cases) are skipped unless ``--runslow``.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
