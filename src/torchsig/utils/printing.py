""" Contains Helpful methods for properly implementing `__str__` and `__repr__` methods of classes
"""

# TorchSig
from torchsig.utils.random import Seedable

# Built-In
from typing import Any, List
import textwrap


seedable_remove = [
    'children',
    'rng_seed',
    'np_rng',
    'random_generator',
    'torch_rng',
    'parent'
]

def generate_repr_str(class_object: Any, exclude_params: List[str] = []) -> str:
    """Generates a string representation of the class object, excluding specified parameters.

    This function creates a human-readable string representation of the given class object, 
    including its class name and parameters. It excludes any parameters specified in the 
    `exclude_params` list. If the class object is an instance of `Seedable`, certain 
    attributes related to seeding are handled specifically.

    Args:
        class_object (Any): The class object to generate the string representation for.
        exclude_params (List[str], optional): A list of parameter names to exclude from 
                                              the string representation. Defaults to an empty list.

    Returns:
        str: A formatted string representation of the class object with parameters.

    Raises:
        AttributeError: If the class object does not have a `__dict__` attribute or 
                        any other required attributes for the operation.

    Example:
        >>> class Example:
        >>>     def __init__(self, param1, param2):
        >>>         self.param1 = param1
        >>>         self.param2 = param2
        >>> e = Example(1, 2)
        >>> generate_repr_str(e)
        'Example(param1=1,param2=2)'

    Notes:
        - If the class object is an instance of `Seedable`, the `seed` and `parent` 
          attributes will be added back into the string representation.
    """
    class_dict = class_object.__dict__.copy()
    
    # remove any exclude params
    for r in exclude_params:
        if r in class_dict:
            class_dict.pop(r)

    if isinstance(class_object, Seedable):
        # remove Seedable params
        for r in seedable_remove:
            class_dict.pop(r)
        
        # add back in (last)
        #class_dict['seed'] = class_object.rng_seed
        # class_dict['parent'] = class_object.parent.__repr__()

    params = [f"{k}={v}" for k,v in class_dict.items()]
    params_str = ",".join(params)

    return f"{class_object.__class__.__name__}({params_str})"


### DatasetMetadata

def dataset_metadata_str(
    dataset_metadata,
    max_width: int = 100,
    first_col_width: int = 29,
    array_width_indent_offset: int = 2
) -> str:
    """Custom string representation for the class.

    This method returns a formatted string that provides a detailed summary of 
    the object’s key attributes, including signal parameters, dataset configuration, 
    and transform details. It uses `textwrap.fill` to format long attributes such as 
    lists or arrays into a neatly wrapped format for easier readability.

    The string includes information on the dataset's configuration, signal characteristics, 
    transformations, and other attributes in a human-readable way. The result is intended 
    to provide a concise yet comprehensive overview of the object's state, useful for 
    debugging, logging, or displaying object details.

    Args:
        dataset_metadata (Any): The dataset metadata object to generate a string for.
        max_width (int, optional): Maximum width of the output string. Defaults to 100.
        first_col_width (int, optional): Width of the first column in the output string. Defaults to 29.
        array_width_indent_offset (int, optional): Indentation offset for array-like attributes. Defaults to 2.

    Returns:
        str: A formatted string that represents the object’s attributes in a readable format.

    Example Output:
        ```
        MyClass
        ----------------------------------------------------------------------------------------------------
        num_iq_samples_dataset            1000        
        fft_size                          512       
        sample_rate                       1000.0    
        num_signals_min                   1         
        num_signals_max                   5         
        num_signals_distribution          [0.2, 0.3, 0.5]       
        snr_db_min                        5.0       
        snr_db_max                        30.0      
        signal_duration_min               0.001
        signal_duration_max               0.01
        signal_bandwidth_min              10
        signal_bandwidth_max              100
        signal_center_freq_min            -10
        signal_center_freq_max            10    
        class_list                        [Class1, Class2, Class3]    
        class_distribution                [0.3, 0.4, 0.3]       
        seed                               42         
        ```
    """
    # second_col_width = max_width - first_col_width
    array_width_indent = first_col_width + array_width_indent_offset


    num_signals_distribution_str = textwrap.fill(
        f"{None if dataset_metadata.num_signals_distribution is None else dataset_metadata.num_signals_distribution.tolist()}",
        width = max_width,
        initial_indent= f"{' ' * first_col_width}",
        subsequent_indent= f"{' ' * array_width_indent}",
    )[first_col_width:]

    class_distribution_str = textwrap.fill(
        f"{None if dataset_metadata.class_distribution is None else dataset_metadata.class_distribution.tolist()}",
        width = max_width,
        initial_indent= f"{' ' * first_col_width}",
        subsequent_indent= f"{' ' * array_width_indent}",
    )[first_col_width:]

    class_list_str = textwrap.fill(
        f"{dataset_metadata.class_list}",
        width = max_width,
        initial_indent= f"{' ' * first_col_width}",
        subsequent_indent= f"{' ' * array_width_indent}",
    )[first_col_width:]

    return (
        f"\n{dataset_metadata.__class__.__name__}\n"
        f"{'-' * max_width}\n"
        f"{'num_iq_samples_dataset':<29} {dataset_metadata.num_iq_samples_dataset:<10}\n"
        f"{'fft_size':<29} {dataset_metadata.fft_size}\n"
        f"{'sample_rate':<29} {dataset_metadata.sample_rate}\n" 
        f"{'num_signals_min':<29} {dataset_metadata.num_signals_min}\n"
        f"{'num_signals_max':<29} {dataset_metadata.num_signals_max}\n"
        f"{'num_signals_distribution':<29} {num_signals_distribution_str}\n" 
        f"{'snr_db_min':<29} {dataset_metadata.snr_db_min}\n" 
        f"{'snr_db_max':<29} {dataset_metadata.snr_db_max}\n" 
        f"{'signal_duration_min':<29} {dataset_metadata.signal_duration_min}\n" 
        f"{'signal_duration_max':<29} {dataset_metadata.signal_duration_max}\n" 
        f"{'signal_bandwidth_min':<29} {dataset_metadata.signal_bandwidth_min}\n" 
        f"{'signal_bandwidth_max':<29} {dataset_metadata.signal_bandwidth_max}\n" 
        f"{'signal_center_freq_min':<29} {dataset_metadata.signal_center_freq_min}\n" 
        f"{'signal_center_freq_max':<29} {dataset_metadata.signal_center_freq_max}\n"
        f"{'class_list':<29} {class_list_str}\n" 
        f"{'class_distribution':<29} {class_distribution_str}\n" 
        ####f"{'seed':<29} {dataset_metadata.rng_seed}\n"   
    )

def dataset_metadata_repr(
    dataset_metadata
) -> str:
    """Return a string representation of the object for debugging and inspection.

    This method generates a string that provides a concise yet detailed summary 
    of the object’s state, useful for debugging or interacting with the object 
    in an interactive environment (e.g., REPL, Jupyter notebooks).

    The `__repr__` method is intended to give an unambiguous, readable string that 
    represents the object. The returned string includes key attributes and their 
    values, formatted in a way that can be interpreted back as code, i.e., it aims to 
    provide a string that could be used to recreate the object (though not necessarily 
    identical, as it is for debugging purposes).

    Returns:
        str: A detailed, formatted string that represents the object’s state, showing 
            key attributes and their current values.
    """
    return (
        f"{dataset_metadata.__class__.__name__}"
        f"("
            f"num_iq_samples_dataset={dataset_metadata.num_iq_samples_dataset},"
            f"fft_size={dataset_metadata.fft_size},"
            f"num_signals_max={dataset_metadata.num_signals_max},"
            f"sample_rate={dataset_metadata.sample_rate}," 
            f"num_signals_min={dataset_metadata.num_signals_min},"
            f"num_signals_distribution={None if dataset_metadata.num_signals_distribution is None else dataset_metadata.num_signals_distribution.tolist()}," 
            f"snr_db_min={dataset_metadata.snr_db_min}," 
            f"snr_db_max={dataset_metadata.snr_db_max}," 
            f"signal_duration_min={dataset_metadata.signal_duration_min}," 
            f"signal_duration_max={dataset_metadata.signal_duration_max}," 
            f"signal_bandwidth_min={dataset_metadata.signal_bandwidth_min}," 
            f"signal_bandwidth_max={dataset_metadata.signal_bandwidth_max}," 
            f"signal_center_freq_min={dataset_metadata.signal_center_freq_min}," 
            f"signal_center_freq_max={dataset_metadata.signal_center_freq_max}," 
            f"class_list={dataset_metadata.class_list}," 
            f"class_distribution={None if dataset_metadata.class_distribution is None else dataset_metadata.class_distribution.tolist()}"
        f")"
    )
