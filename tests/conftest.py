# tests/conftest.py
import pytest

def pytest_collection_modifyitems(items):
    """Reorder test items to ensure tests run in the desired order"""
    # Define the priority order for test files
    test_order = {
        # Directories
        "signals/": 0,
        "transforms": 1,
        "utils/": 2,
        "datasets/": 3,

        # Individual test files
        "test_datasets.py": 3.1,
        "test_datamodules.py": 3.2,

        # Default priority for other files
        "default": 100
    }

    def get_priority(item):
        """Get priority for a test item based on its path"""
        nodeid = item.nodeid.split('::')[0]  # Get the file path

        # Check if the path starts with any of our directories
        for path, priority in test_order.items():
            if path != "default" and nodeid.startswith(path):
                return priority

        return test_order["default"]

    # Sort the test items based on priority
    items.sort(key=get_priority)