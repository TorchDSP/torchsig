"""
Utility functions for dealing with the Signal type
"""

from typing import List


def check_signal_class(name: str, possible_names: List[str]) -> bool:
    """
    Check if the provided signal name matches any of the possible signal names.

    Args:
        name (str): The signal name to check.
        possible_names (List[str]): A list of possible signal names to compare against.

    Returns:
        bool: True if the signal name matches any of the possible names, otherwise False.
    
    """
    is_type_signal = [n in name for n in possible_names]
    return any(is_type_signal)