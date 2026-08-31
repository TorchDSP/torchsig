# tests/conftest.py
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--test-mode",
        action="store",
        default="fast",
        choices=("fast", "full"),
        help="fast skips selected slow tests; full runs everything",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_no_gpu: skip in fast mode when no GPU is available",
    )
    config.addinivalue_line(
        "markers",
        "slow: skip in fast mode regardless of GPU availability",
    )


def pytest_collection_modifyitems(session, config, items):
    test_mode = config.getoption("--test-mode")
    has_gpu = torch.cuda.is_available()

    if test_mode != "full":
        skip_slow = pytest.mark.skip(reason="skipped in fast test mode")
        skip_slow_no_gpu = pytest.mark.skip(reason="skipped in fast test mode because no GPU is available")

        for item in items:
            if item.get_closest_marker("slow") is not None:
                item.add_marker(skip_slow)

            if item.get_closest_marker("slow_no_gpu") is not None and not has_gpu:
                item.add_marker(skip_slow_no_gpu)

    # Define the priority order for test files
    priority_order = ["tests/signals", "tests/transforms", "tests/utils", "tests/datasets", "tests/geo"]
    # Individual test files
    #     "test_datasets.py",
    #     "test_datamodules.py",
    # ]
    # test_order = {
    #     # Directories
    #     "signals/",
    #     "transforms": 1,
    #     "utils/": 2,
    #     "datasets/": 3,

    #     # Individual test files
    #     "test_datasets.py": 3.1,
    #     "test_datamodules.py": 3.2,

    #     # Default priority for other files
    #     "default": 100
    # }

    rootdir = Path(config.rootdir)

    def get_priority(item):
        """Get priority for a test item based on its path"""
        rel_path = Path(item.fspath).relative_to(rootdir)

        # Check if the path starts with any of our directories
        for i, folder in enumerate(priority_order):
            if str(rel_path).startswith(folder):
                return i

        # Default priority
        return len(priority_order)

    # Sort the test items based on priority
    items.sort(key=get_priority)
