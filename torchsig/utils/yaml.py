"""YAML utilities
"""

import yaml


def custom_representer(dumper, value):
    """Custom representer for YAML to handle sequences (lists).

    This function customizes how lists are represented in the YAML output, using 
    flow style for sequences (inline lists).

    Args:
        dumper (yaml.Dumper): The YAML dumper responsible for serializing the data.
        value (list): The list to be represented in YAML.

    Returns:
        yaml.Dumper: The dumper with the custom representation for the list.
    """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)

def write_dict_to_yaml(filename: str, info_dict: dict) -> None:
    """Writes a dictionary to a YAML file with customized settings.

    This function writes the provided `info_dict` to a YAML file. It customizes 
    the representation of lists by using the `custom_representer`, and it uses 
    specific formatting options (e.g., no sorting of keys, custom line width).

    Args:
        filename (str): The name of the YAML file to which the dictionary will be written.
        info_dict (dict): The dictionary to be written to the YAML file.

    Returns:
        None: This function does not return any value.
    """
    yaml.add_representer(list, custom_representer)

    with open(filename, 'w+') as file:
        yaml.dump(info_dict, file, default_flow_style=False, sort_keys=False, width=200)