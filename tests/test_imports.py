import subprocess
import sys
from importlib import import_module
from types import ModuleType

import pytest

import torchsig
from torchsig import _lazy


def run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )


def test_top_level_import_does_not_import_subpackages():
    result = run_python("import sys, torchsig; unexpected = sorted(name for name in sys.modules if name.startswith('torchsig.') and name != 'torchsig._lazy'); assert not unexpected, unexpected")

    assert result.returncode == 0, result.stderr


def test_lazy_module_is_loaded_on_access_and_cached():
    result = run_python(
        "import sys, torchsig; assert 'torchsig.signals' not in sys.modules; first = torchsig.signals; assert first is sys.modules['torchsig.signals']; assert first is torchsig.signals"
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        ("torchsig", {"datasets", "geo", "signals", "transforms", "utils"}),
        ("torchsig.datasets", {"datamodules", "dataset_utils", "datasets", "TorchSigIterableDataset"}),
        ("torchsig.signals", {"builder", "signal_lists", "signal_types", "signal_utils"}),
        ("torchsig.transforms", {"base_transforms", "functional", "impairments", "metadata_transforms", "transforms"}),
        ("torchsig.utils", {"dsp", "file_handlers", "writer", "yaml"}),
        ("torchsig.utils.file_handlers", {"BaseFileHandler", "HDF5Reader", "base_handler", "hdf5"}),
    ],
)
def test_dir_and_all_include_lazy_exports(module_name, expected_names):
    module = import_module(module_name)

    assert expected_names <= set(module.__all__)
    assert set(module.__all__) <= set(dir(module))


def test_unknown_lazy_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'not_an_export'"):
        _ = torchsig.not_an_export


def test_lazy_helper_caches_export(monkeypatch):
    imported_module = ModuleType("example.target")
    imported_module.Exported = object()
    module_globals = {"__name__": "example"}
    imports = []

    def fake_import(module_name):
        imports.append(module_name)
        return imported_module

    monkeypatch.setattr(_lazy, "import_module", fake_import)

    value = _lazy.lazy_getattr("Exported", module_globals, {"Exported": ("example.target", "Exported")})

    assert value is imported_module.Exported
    assert module_globals["Exported"] is value
    assert imports == ["example.target"]


def test_lazy_helper_does_not_mask_import_errors(monkeypatch):
    def fail_import(_module_name):
        raise ImportError("dependency failed")

    monkeypatch.setattr(_lazy, "import_module", fail_import)

    with pytest.raises(ImportError, match="dependency failed"):
        _lazy.lazy_getattr("Exported", {"__name__": "example"}, {"Exported": ("example.target", None)})
