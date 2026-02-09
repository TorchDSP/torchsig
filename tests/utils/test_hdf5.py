import pytest
from typing import Any, Sequence, Optional, List, Tuple, Dict
import numpy as np
import h5py

# --- Test Data Setup ---
# Define complex tuple-based targets for reusability
YOLO_1 = (41, 0.2279052734375, 0.25370860860317335, 0.199951171875, 0.078125)
YOLO_2 = (0, 0.6279296875, 0.3705115805038923, 0.13525390625, 0.390625)
YOLO_3 = (25, 0.7984619140625, 0.05374288372813274, 0.181396484375, 0.10748576735626561)
YOLO_4 = (49, 0.2596435546875, 0.9226088171111424, 0.120361328125, 0.140625)
YOLO_5 = (40, 0.4500732421875, 0.8010194135042737, 0.199951171875, 0.140625)
YOLO_6 = (26, 0.3477783203125, 0.5554004366512442, 0.162353515625, 0.265625)
YOLO_7 = (51, 0.3206787109375, 0.644611440615514, 0.170654296875, 0.328125)
YOLO_8 = (20, 0.8094482421875, 0.18160910573387345, 0.154541015625, 0.36321821136774707)

# --- Test Cases based on the provided table ---
TEST_CASES_test_find_first_target = [
    # id, targets, expected_output
    ("single_str", ("ofdm-300", [], [], []), "ofdm-300"),
    ("multi_str", (["ofdm-300", "ook", "16psk"], ["am-dsb-sc", "ofdm-256", "32psk"], ["am-lsb", "16msk"], []), "ofdm-300"),
    ("single_int", (41, [], [], []), 41),
    ("multi_int", ([41, 0, 25], [49, 40, 26], [51, 20], []), 41),
    ("multi_int_with_zero", ([0, 41, 25], [49, 40, 26]), 0),
    ("single_composite_tuple_name_index", (("ofdm-300", 41), [], [], []), ("ofdm-300", 41)),
    (
        "multi_composite_tuple_name_index",
        (
            [("ofdm-300", 41), ("ook", 0), ("16psk", 25)],
            [("am-dsb-sc", 49)],
            [],
        ),
        ("ofdm-300", 41),
    ),
    ("single_yolo", (YOLO_1, [], [], []), YOLO_1),
    ("multi_yolo", ([YOLO_1, YOLO_2, YOLO_3], [YOLO_4, YOLO_5, YOLO_6], [YOLO_7, YOLO_8], []), YOLO_1),
    ("single_composite_tuple_name_yolo", (("ofdm-300", YOLO_1), [], [], []), ("ofdm-300", YOLO_1)),
    (
        "multi_composite_tuple_name_yolo",
        (
            [("ofdm-300", YOLO_1), ("ook", YOLO_2)],
            [("am-dsb-sc", YOLO_4)],
        ),
        ("ofdm-300", YOLO_1),
    ),
    ("empty_first_then_valid_str", ([], "ofdm-300", []), "ofdm-300"),
    ("empty_first_then_valid_list", ([], ["ofdm-300", "ook"], []), "ofdm-300"),
    ("all_empty", ([], [], (), []), None),
    ("empty_tuple_then_valid", ((), "ofdm-300"), "ofdm-300"),
    ("empty_batch", (), None),
]


YOLO_TUPLE = (41, 0.227, 0.253, 0.199, 0.078)
YOLO_DTYPE = np.dtype([("f0", np.int32), ("f1", np.float32), ("f2", np.float32), ("f3", np.float32), ("f4", np.float32)])
YOLO_PADDING = (-1, -1.0, -1.0, -1.0, -1.0)

COMPLEX_TUPLE = ("ofdm", YOLO_TUPLE)
COMPLEX_DTYPE = np.dtype([("f0", "<S32"), ("f1", YOLO_DTYPE)])
COMPLEX_PADDING = (b"-1", YOLO_PADDING)


# --- Test Cases ---
TEST_CASES_test_get_target_properties = [
    # id, target, expected_dtype, expected_shape, expected_padding
    ("base_string", "a_string", "<S32", (), b"-1"),
    ("base_int", 42, np.int32, (), -1),
    ("base_float", 3.14, np.float32, (), -1.0),
    ("uniform_list_int", [1, 2, 3], np.int32, (3,), (-1, -1, -1)),
    ("uniform_list_str", ["a", "b"], "<S32", (2,), (b"-1", b"-1")),
    ("uniform_nested_list", [[1.0, 2.0], [3.0, 4.0]], np.float32, (2, 2), ((-1.0, -1.0), (-1.0, -1.0))),
    ("struct_yolo_tuple", YOLO_TUPLE, YOLO_DTYPE, (), YOLO_PADDING),
    ("struct_complex_tuple", COMPLEX_TUPLE, COMPLEX_DTYPE, (), COMPLEX_PADDING),
    ("struct_list_of_yolos", [YOLO_TUPLE, YOLO_TUPLE], YOLO_DTYPE, (2,), (YOLO_PADDING, YOLO_PADDING)),
    # --- Error Cases ---
    ("error_empty_list", [], None, None, pytest.raises(ValueError)),
    ("error_empty_tuple", (), None, None, pytest.raises(ValueError)),
    ("error_unsupported_type", {"a": 1}, None, None, pytest.raises(TypeError)),
    # ("error_non_uniform_struct", [("a",), ("b", "c")], None, None, pytest.raises(TypeError)),
]
