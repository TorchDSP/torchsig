import json
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from torchsig.utils.types import *

SIGMF_DTYPE_MAP: Dict[str, np.dtype] = {
    "cf64_le": np.dtype("<f8"),
    "cf64_be": np.dtype(">f8"),
    "cf32_le": np.dtype("<f4"),
    "cf32_be": np.dtype(">f4"),
    "ci32_le": np.dtype("<i4"),
    "ci32_be": np.dtype(">i4"),
    "ci16_le": np.dtype("<i2"),
    "ci16_be": np.dtype(">i2"),
    "ci8_le": np.dtype("<i1"),
    "ci8_be": np.dtype(">i1"),
    "cu32_le": np.dtype("<u4"),
    "cu32_be": np.dtype(">u4"),
    "cu16_le": np.dtype("<u2"),
    "cu16_be": np.dtype(">u2"),
    "cu8_le": np.dtype("<u1"),
    "cu8_be": np.dtype(">u1"),
    "rf64_le": np.dtype("<f8"),
    "rf64_be": np.dtype(">f8"),
    "rf32_le": np.dtype("<f4"),
    "rf32_be": np.dtype(">f4"),
    "ri32_le": np.dtype("<i4"),
    "ri32_be": np.dtype(">i4"),
    "ri16_le": np.dtype("<i2"),
    "ri16_be": np.dtype(">i2"),
    "ri8_le": np.dtype("<i1"),
    "ri8_be": np.dtype(">i1"),
    "ru32_le": np.dtype("<u4"),
    "ru32_be": np.dtype(">u4"),
    "ru16_le": np.dtype("<u2"),
    "ru16_be": np.dtype(">u2"),
    "ru8_le": np.dtype("<u1"),
    "ru8_be": np.dtype(">u1"),
}


def save_index(index: List[Tuple[Any, SignalCapture]], pkl_file_path: str):
    with open(pkl_file_path, "wb") as pkl_file:
        pickle.dump(index, pkl_file)


def indexer_from_pickle(pkl_file_path: str) -> List[Tuple[Any, SignalCapture]]:
    """
    Args:
        pkl_file_path: Absolute file path of pkl file to load index from

    Returns:
        index: tuple of target, meta-data pairs

    """
    with open(pkl_file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)


def indexer_from_folders_sigmf(root: str) -> List[Tuple[Any, SignalCapture]]:
    """An indexer where classes are delineated by folders

    Notes:
        Assumes data is stored as follows:
            root/class_x/xxx.sigmf-data
            root/class_x/xxx.sigmf-meta

            root/class_y/yxx.sigmf-data
            root/class_y/yxx.sigmf-meta

    Args:
        root:

    Returns:
        index: tuple of target, meta-data pairs

    """
    # go through directories and find files
    non_empty_dirs = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    non_empty_dirs = [d for d in non_empty_dirs if os.listdir(os.path.join(root, d))]

    # Identify all files associated with each class
    index = []
    for dir_idx, dir_name in enumerate(non_empty_dirs):
        class_dir = os.path.join(root, dir_name)

        # Find files with sigmf-data at the end and make a list
        proper_sigmf_files = list(
            filter(
                lambda x: x.split(".")[-1] in {"sigmf-data"}
                and os.path.isfile(os.path.join(class_dir, x)),
                os.listdir(os.path.join(root, dir_name)),
            )
        )

        # Go through each file and create and index
        for f in proper_sigmf_files:
            for signal_file in _parse_sigmf_captures(os.path.join(class_dir, f)):
                index.append((dir_name, signal_file))

    return index


def _parse_sigmf_captures(absolute_file_path: str) -> List[SignalCapture]:
    """
    Args:
        absolute_file_path: absolute file path of sigmf-data file for which to create Captures

    Returns:
        signal_files:

    """
    meta_file_name = "{}{}".format(
        absolute_file_path.split("sigmf-data")[0], "sigmf-meta"
    )
    meta = json.load(open(meta_file_name, "r"))
    item_type = SIGMF_DTYPE_MAP[meta["global"]["core:datatype"]]
    sample_size = item_type.itemsize * (
        2 if "c" in meta["global"]["core:datatype"] else 1
    )
    total_num_samples = os.path.getsize(absolute_file_path) // sample_size

    # It's quite common for there to be only a single "capture" in sigMF
    meta = create_signal_metadata(
        sample_rate=meta["global"]["core:sample_rate"],
    )
    if len(meta["captures"]) == 1:
        has_matching_note = (
            meta["annotations"][0]["core:sample_start"]
            == meta["captures"][0]["core:sample_start"]
        )
        has_frequency_info = "core:freq_upper_edge" in meta["annotations"][0]

        if has_matching_note and has_frequency_info:
            meta["upper_freq"] = meta["annotations"][0]["core:freq_upper_edge"]
            meta["lower_freq"] = meta["annotations"][0]["core:freq_lower_edge"]

        return [
            SignalCapture(
                absolute_path=absolute_file_path,
                num_bytes=sample_size * total_num_samples,
                byte_offset=sample_size * meta["captures"][0]["core:sample_start"],
                item_type=item_type,
                is_complex=True if "c" in meta["global"]["core:datatype"] else False,
                metadata=meta,
            )
        ]

    # If there's more than one, we construct a list of captures
    signal_files = []
    for capture_idx, capture in enumerate(meta["captures"]):
        capture_start_idx = meta["captures"][capture_idx]["core:sample_start"]
        annotation_start_idx = meta["annotations"][capture_idx]["core:sample_start"]
        has_matching_note = capture_start_idx == annotation_start_idx
        has_frequency_info = "core:freq_upper_edge" in meta["annotations"][capture_idx]

        if has_matching_note and has_frequency_info:
            meta["upper_freq"] = meta["annotations"][capture_idx][
                "core:freq_upper_edge"
            ]
            meta["lower_freq"] = meta["annotations"][capture_idx][
                "core:freq_lower_edge"
            ]

        samples_in_capture = int(total_num_samples - capture_start_idx)
        if capture_idx < len(meta["captures"]) - 1:
            samples_in_capture = (
                meta["captures"][capture_idx + 1]["core:sample_start"]
                - capture_start_idx
            )

        signal_files.append(
            SignalCapture(
                absolute_path=absolute_file_path,
                num_bytes=sample_size * samples_in_capture,
                byte_offset=sample_size * capture_start_idx,
                item_type=item_type,
                is_complex=True if "c" in meta["global"]["core:datatype"] else False,
                metadata=deepcopy(meta),
            )
        )
    return signal_files
