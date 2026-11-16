"""Tests for optional file-handler metrics and validation policy."""

from pathlib import Path

from torchsig.utils.file_handlers import FileReader
from torchsig.utils.file_handlers.metrics import ValidationPolicy


class _Reader(FileReader):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.values = [b"value"]

    def read(self, idx: int) -> bytes:
        return self.values[self.validate_index(idx)]

    def __len__(self) -> int:
        return len(self.values)


def test_metrics_are_disabled_by_default_and_do_not_change_values(tmp_path: Path) -> None:
    reader = _Reader(tmp_path)

    assert reader.read(0) == b"value"
    assert reader.metrics.snapshot()["records_read"] == 0
    reader.enable_metrics()
    assert reader.read(0) == b"value"
    assert reader.metrics.snapshot()["records_read"] == 1


def test_validation_policy_is_checked_centrally(tmp_path: Path) -> None:
    reader = _Reader(tmp_path)

    reader.set_validation_policy("lazy")
    assert reader.validation_policy is ValidationPolicy.LAZY
