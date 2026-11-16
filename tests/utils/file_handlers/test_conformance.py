"""Cross-backend conformance checks driven by the central registry."""

from pathlib import Path

import pytest

from torchsig.utils.file_handlers import FileHandlerIndexError, FileReader
from torchsig.utils.file_handlers.registry import file_handler_registry


class _Reader(FileReader):
    def __init__(self, root: Path, values: list[object]) -> None:
        super().__init__(root)
        self.values = values
        self.closed = False

    def read(self, idx: int) -> object:
        return self.values[self.validate_index(idx)]

    def __len__(self) -> int:
        return len(self.values)

    def teardown(self) -> None:
        self.closed = True


def test_common_reader_index_and_lifecycle_contract(tmp_path: Path) -> None:
    reader = _Reader(tmp_path, ["value"])

    assert reader.read(0) == "value"
    with pytest.raises(FileHandlerIndexError, match=str(tmp_path)):
        reader.read(1)
    with pytest.raises(TypeError, match="integer"):
        reader.read(True)
    with reader:
        pass
    assert reader.closed
    reader.close()


@pytest.mark.parametrize("spec", file_handler_registry.specs.values(), ids=lambda spec: spec.name)
def test_registered_backends_obey_declared_class_contract(spec) -> None:
    assert issubclass(spec.reader, FileReader)
    assert spec.reader.read is not FileReader.read
    assert spec.reader.__len__ is not FileReader.__len__
    if spec.writer is not None:
        assert not getattr(spec.writer.write, "__isabstractmethod__", False)


def test_public_exception_category_remains_index_error() -> None:
    error = FileHandlerIndexError("bad index")
    assert isinstance(error, IndexError)


def test_read_many_fallback_preserves_order_duplicates_and_empty(tmp_path: Path) -> None:
    reader = _Reader(tmp_path, ["zero", "one", "two"])

    assert reader.read_many([]) == []
    assert reader.read_many([2, 0, 2]) == ["two", "zero", "two"]
    assert reader.read_many(iter([1, 0])) == ["one", "zero"]


@pytest.mark.parametrize("indices", [[-1], [1], [True], "0", None])
def test_read_many_fallback_rejects_invalid_collections(tmp_path: Path, indices) -> None:
    reader = _Reader(tmp_path, ["zero"])

    with pytest.raises((FileHandlerIndexError, TypeError)):
        reader.read_many(indices)
