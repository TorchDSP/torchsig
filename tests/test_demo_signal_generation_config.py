"""Smoke test for the signal-generation configuration demonstration."""

import yaml

from examples.scripts.demo_signal_generation_config import main


def test_demo_signal_generation_config(tmp_path, capsys):
    output_dir = tmp_path / "demo"

    main(["--output-dir", str(output_dir)])

    assert "Generated alpha values: [0.35, 0.35, 0.35, 0.35, 0.35]" in capsys.readouterr().out
    artifact = yaml.safe_load((output_dir / "generated_dataset" / "dataset_info.yaml").read_text())
    assert artifact["experiment_config"]["signals"]["qpsk"]["parameters"]["alpha_rolloff"]["value"] == 0.35
