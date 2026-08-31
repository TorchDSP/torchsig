import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from torchsig.utils.file_handlers import WAVReader


# ==============================================================================
# MOCK HELPER
# ==============================================================================
def create_mock_dataset(root: Path, num_files: int, elements_per_file: int, num_samples: int):
    """Helper to create a valid TorchSig-style WAV dataset on disk."""
    root.mkdir(parents=True, exist_ok=True)

    # 1. Create WAV files
    for f_idx in range(num_files):
        sub_dir = root / f"mod_{f_idx % 2}"
        sub_dir.mkdir(exist_ok=True)
        wav_path = sub_dir / f"file_{f_idx}.wav"

        total_frames = elements_per_file * num_samples
        data = np.zeros((total_frames, 2), dtype="float32")
        for e in range(elements_per_file):
            start = e * num_samples
            end = (e + 1) * num_samples
            # We use values > 1.0 to test that the reader doesn't clip
            data[start:end, 0] = f_idx + 0.1  # File 0: 0.1, File 1: 1.1
            data[start:end, 1] = f_idx + 0.2  # File 0: 0.2, File 1: 1.2

        # Use subtype='FLOAT' to prevent clipping values at +/- 1.0
        sf.write(wav_path, data, 48000, subtype="FLOAT")

    # 2. Create metadata.csv
    total_elements = num_files * elements_per_file
    with open(root / "metadata.csv", "w", encoding="utf-8") as f:
        f.writelines(f"{i},label_{i},0,48000\n" for i in range(total_elements))

    # 3. Create info.json
    info = {"num_iq_samples": num_samples, "elements_per_file": elements_per_file, "size": total_elements, "class_list": ["mod_0", "mod_1"], "sample_rate": 48000}
    with open(root / "info.json", "w") as f:
        json.dump(info, f)


# ==============================================================================
# TESTS
# ==============================================================================


def test_init_inference(tmp_path):
    """Test that num_iq_samples is inferred correctly when JSON is missing.
    If 2 files have 2 elements each, and total frames per file is 200,
    num_iq_samples MUST be 100.
    """
    num_files, elements, samples = 2, 2, 100
    create_mock_dataset(tmp_path, num_files, elements, samples)

    # REMOVE JSON to force inference
    os.remove(tmp_path / "info.json")

    # VERIFICATION: Ensure CSV actually has the right number of lines before initializing
    with open(tmp_path / "metadata.csv") as f:
        lines = f.readlines()
        assert len(lines) == (num_files * elements), f"CSV should have {num_files * elements} lines, found {len(lines)}"

    reader = WAVReader(tmp_path)

    # Debugging prints if it fails again
    print(f"Dataset Size: {reader.dataset_size}")
    print(f"Files Found: {len(reader.wav_files)}")
    print(f"Elements Per File: {reader.elements_per_file}")

    assert reader.num_iq_samples == samples, f"Expected {samples}, got {reader.num_iq_samples}"


def test_read_single_element(tmp_path):
    """Verify that read(idx) retrieves the correct complex sample from the correct file."""
    num_files, elements, samples = 2, 3, 100
    create_mock_dataset(tmp_path, num_files, elements, samples)

    reader = WAVReader(tmp_path)

    # Element 4: File 0 (0,1,2), File 1 (3,4,5) -> should be in File 1
    sig = reader.read(4)

    assert sig.data.shape == (samples,)
    # File 1 has I=1.1, Q=1.2
    np.testing.assert_allclose(sig.data, 1.1 + 1.2j)


def test_recursive_rglob(tmp_path):
    """Test that files in subdirectories are found."""
    create_mock_dataset(tmp_path, 2, 1, 100)
    reader = WAVReader(tmp_path)
    assert len(reader.wav_files) == 2
