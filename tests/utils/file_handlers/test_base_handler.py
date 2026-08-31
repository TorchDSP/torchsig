"""Tests for file handler base and utility classes."""

from unittest.mock import Mock

import pytest

from torchsig.utils.file_handlers.base_handler import (
    BaseFileHandler,
    FileReader,
    FileWriter,
    reset_folder,
)


def test_reset_folder_creates_missing_directory(tmp_path):
    path = tmp_path / "new" / "nested"

    reset_folder(str(path))

    assert path.exists()
    assert path.is_dir()


def test_reset_folder_deletes_existing_directory_contents(tmp_path):
    path = tmp_path / "dataset"
    path.mkdir()
    (path / "old_file.txt").write_text("old data")
    (path / "subdir").mkdir()
    (path / "subdir" / "nested.txt").write_text("nested data")

    reset_folder(str(path))

    assert path.exists()
    assert path.is_dir()
    assert list(path.iterdir()) == []


def test_reset_folder_raises_when_path_is_file(tmp_path):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("content")

    with pytest.raises(ValueError, match="Path is not a directory"):
        reset_folder(str(file_path))


class TrackingFileWriter(FileWriter):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)
        self.setup_called = False
        self.teardown_called = False
        self.written_batches = []

    def _setup(self):
        self.setup_called = True
        (self.root / "setup_marker.txt").write_text("setup complete")

    def teardown(self):
        self.teardown_called = True

    def write(self, batch_idx, data):
        self.written_batches.append((batch_idx, data))

    def __len__(self):
        return len(self.written_batches)


def test_file_writer_resolves_root_path(tmp_path):
    writer = FileWriter(str(tmp_path))

    assert writer.root == tmp_path.resolve()


def test_file_writer_setup_resets_root_and_calls_setup_hook(tmp_path):
    root = tmp_path / "writer_root"
    root.mkdir()
    (root / "stale.txt").write_text("stale")

    writer = TrackingFileWriter(str(root))

    writer.setup()

    assert root.exists()
    assert not (root / "stale.txt").exists()
    assert writer.setup_called is True
    assert (root / "setup_marker.txt").read_text() == "setup complete"


def test_file_writer_exists_reflects_root_existence(tmp_path):
    root = tmp_path / "writer_root"
    writer = FileWriter(str(root))

    assert writer.exists() is False

    root.mkdir()

    assert writer.exists() is True


def test_file_writer_context_manager_calls_setup_and_teardown(tmp_path):
    root = tmp_path / "writer_root"

    with TrackingFileWriter(str(root)) as writer:
        assert writer.setup_called is True
        assert root.exists()
        assert (root / "setup_marker.txt").exists()
        assert writer.teardown_called is False

    assert writer.teardown_called is True


def test_file_writer_context_manager_does_not_suppress_exceptions(tmp_path):
    root = tmp_path / "writer_root"

    with pytest.raises(RuntimeError, match="boom"), TrackingFileWriter(str(root)):
        raise RuntimeError("boom")


def test_file_writer_write_is_not_implemented(tmp_path):
    writer = FileWriter(str(tmp_path))

    with pytest.raises(NotImplementedError):
        writer.write(0, {"data": 1})


def test_file_writer_len_is_not_implemented(tmp_path):
    writer = FileWriter(str(tmp_path))

    with pytest.raises(NotImplementedError):
        len(writer)


def test_file_writer_str_returns_class_name(tmp_path):
    writer = FileWriter(str(tmp_path))

    assert str(writer) == "FileWriter"


def test_file_writer_repr_uses_generate_repr_str(tmp_path, monkeypatch):
    writer = FileWriter(str(tmp_path))
    mock_generate_repr = Mock(return_value="mock repr")

    monkeypatch.setattr(
        "torchsig.utils.file_handlers.base_handler.generate_repr_str",
        mock_generate_repr,
    )

    assert repr(writer) == "mock repr"
    mock_generate_repr.assert_called_once_with(writer)


class TrackingFileReader(FileReader):
    def __init__(self, root: str, **kwargs):
        super().__init__(root, **kwargs)
        self.items = kwargs.get("items", ["a", "b"])

    def read(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)


def test_file_reader_initializes_root_and_dataset_info_path(tmp_path):
    reader = FileReader(str(tmp_path))

    assert reader.root == tmp_path.resolve()
    assert reader.dataset_info_filepath == tmp_path.resolve() / "dataset_info.yaml"


def test_file_reader_read_is_not_implemented(tmp_path):
    reader = FileReader(str(tmp_path))

    with pytest.raises(NotImplementedError):
        reader.read(0)


def test_file_reader_len_is_not_implemented(tmp_path):
    reader = FileReader(str(tmp_path))

    with pytest.raises(NotImplementedError):
        len(reader)


def test_file_reader_str_returns_class_name(tmp_path):
    reader = FileReader(str(tmp_path))

    assert str(reader) == "FileReader"


def test_file_reader_repr_uses_generate_repr_str(tmp_path, monkeypatch):
    reader = FileReader(str(tmp_path))
    mock_generate_repr = Mock(return_value="mock reader repr")

    monkeypatch.setattr(
        "torchsig.utils.file_handlers.base_handler.generate_repr_str",
        mock_generate_repr,
    )

    assert repr(reader) == "mock reader repr"
    mock_generate_repr.assert_called_once_with(reader)


def test_base_file_handler_create_reader_uses_reader_class(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseFileHandler, "reader_class", TrackingFileReader)

    handler = BaseFileHandler.create_handler(
        mode="r",
        root=str(tmp_path),
        items=["x", "y", "z"],
    )

    assert isinstance(handler, TrackingFileReader)
    assert handler.root == tmp_path.resolve()
    assert len(handler) == 3
    assert handler.read(1) == "y"


def test_base_file_handler_create_writer_uses_writer_class(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseFileHandler, "writer_class", TrackingFileWriter)

    handler = BaseFileHandler.create_handler(mode="w", root=str(tmp_path))

    assert isinstance(handler, TrackingFileWriter)
    assert handler.root == tmp_path.resolve()


def test_base_file_handler_create_handler_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="Invalid File Handler mode: invalid"):
        BaseFileHandler.create_handler(mode="invalid", root=str(tmp_path))


def test_base_file_handler_str_returns_class_name():
    handler = BaseFileHandler()

    assert str(handler) == "BaseFileHandler"


def test_base_file_handler_repr_uses_generate_repr_str(monkeypatch):
    handler = BaseFileHandler()
    mock_generate_repr = Mock(return_value="mock handler repr")

    monkeypatch.setattr(
        "torchsig.utils.file_handlers.base_handler.generate_repr_str",
        mock_generate_repr,
    )

    assert repr(handler) == "mock handler repr"
    mock_generate_repr.assert_called_once_with(handler)
