"""Tests for the portable dataset split manifest."""

from copy import deepcopy

import pytest
import yaml

from torchsig.datasets.split_manifest import (
    DatasetSplitManifest,
    ManifestProvenance,
    MembershipReference,
    SplitManifestEntry,
    UnsupportedSplitManifestVersionError,
    dump_split_manifest,
    load_split_manifest,
    save_split_manifest,
)

BACKEND = "torchsig.utils.file_handlers.hdf5.HDF5Reader"


def _entry(name, *, path=None, length=10, seed=1, membership=None, dataset_id=None):
    return SplitManifestEntry(
        name=name,
        dataset_id=dataset_id or f"example-{name}",
        path=name if path is None else path,
        storage_backend=BACKEND,
        length=length,
        generation_seed=seed,
        membership=membership,
    )


def _physical_manifest():
    return DatasetSplitManifest(
        collection_id="example",
        layout="physical",
        provenance=ManifestProvenance(producer="torchsig", producer_version="3.0.0"),
        train=_entry("train", length=80),
        validation=_entry("val", length=10),
        test=_entry("test", length=10),
    )


def _shared_manifest():
    def shared_entry(role, length):
        return _entry(
            role,
            path=".",
            length=length,
            seed=7,
            dataset_id="example-full",
            membership=MembershipReference(path=f"membership/{role}.npy", format="numpy-int64-v1"),
        )

    return DatasetSplitManifest(
        collection_id="example",
        layout="shared",
        provenance=ManifestProvenance(producer="torchsig", producer_version="3.0.0"),
        train=shared_entry("train", 80),
        validation=shared_entry("val", 10),
        test=shared_entry("test", 10),
    )


@pytest.mark.parametrize("manifest_factory", [_physical_manifest, _shared_manifest])
def test_manifest_round_trip_is_typed_and_deterministic(tmp_path, manifest_factory):
    manifest = manifest_factory()
    path = tmp_path / "split_manifest.yaml"

    save_split_manifest(manifest, path)

    assert load_split_manifest(path) == manifest
    assert path.read_text(encoding="utf-8") == dump_split_manifest(manifest)
    assert dump_split_manifest(load_split_manifest(path)) == dump_split_manifest(manifest)


def test_serialized_field_and_split_order_is_stable():
    text = dump_split_manifest(_physical_manifest())

    assert text.index("schema_version:") < text.index("collection_id:") < text.index("layout:")
    assert text.index("  train:") < text.index("  validation:") < text.index("  test:")


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute",
        "../outside",
        "train/../../outside",
        "train\\dataset",
        "C:/dataset",
        "https://example.invalid/dataset",
        "train//dataset",
        "train/./dataset",
    ],
)
def test_split_entry_rejects_unsafe_or_noncanonical_paths(path):
    with pytest.raises((TypeError, ValueError), match="path"):
        _entry("train", path=path)


@pytest.mark.parametrize("path", [".", "../membership.npy", "/membership.npy", "membership\\train.npy"])
def test_membership_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="membership.path"):
        MembershipReference(path=path, format="numpy-int64-v1")


def test_physical_layout_rejects_membership_and_duplicate_roots():
    manifest = _physical_manifest()
    membership = MembershipReference(path="membership/train.npy", format="numpy-int64-v1")

    with pytest.raises(ValueError, match="cannot have membership"):
        DatasetSplitManifest(
            collection_id=manifest.collection_id,
            layout="physical",
            provenance=manifest.provenance,
            train=_entry("train", membership=membership),
            validation=manifest.validation,
            test=manifest.test,
        )
    with pytest.raises(ValueError, match="paths must be distinct"):
        DatasetSplitManifest(
            collection_id=manifest.collection_id,
            layout="physical",
            provenance=manifest.provenance,
            train=_entry("train", path="same"),
            validation=_entry("val", path="same"),
            test=manifest.test,
        )


def test_shared_layout_requires_membership_and_one_backing_dataset():
    manifest = _shared_manifest()

    with pytest.raises(ValueError, match="require membership"):
        DatasetSplitManifest(
            collection_id=manifest.collection_id,
            layout="shared",
            provenance=manifest.provenance,
            train=_entry("train", path=".", dataset_id="example-full", seed=7),
            validation=manifest.validation,
            test=manifest.test,
        )
    with pytest.raises(ValueError, match="same backing dataset"):
        DatasetSplitManifest(
            collection_id=manifest.collection_id,
            layout="shared",
            provenance=manifest.provenance,
            train=manifest.train,
            validation=_entry(
                "val",
                path="other",
                dataset_id="other",
                seed=7,
                membership=MembershipReference(path="membership/val.npy", format="numpy-int64-v1"),
            ),
            test=manifest.test,
        )


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (lambda value: value.pop("collection_id"), ValueError, "missing required fields"),
        (lambda value: value.update({"unknown": True}), ValueError, "unknown fields"),
        (lambda value: value.update({"layout": "automatic"}), ValueError, "layout must be"),
        (lambda value: value["splits"].pop("validation"), ValueError, "missing required fields"),
        (lambda value: value["splits"]["train"].update({"length": -1}), ValueError, "cannot be negative"),
        (lambda value: value["splits"]["train"].update({"generation_seed": True}), TypeError, "generation_seed"),
        (lambda value: value.update({"provenance": []}), TypeError, "provenance must be a mapping"),
    ],
)
def test_parser_rejects_malformed_manifests(mutation, exception, message):
    value = deepcopy(_physical_manifest().to_dict())
    mutation(value)

    with pytest.raises(exception, match=message):
        DatasetSplitManifest.from_dict(value)


@pytest.mark.parametrize("version", [0, 2, 999])
def test_parser_rejects_unsupported_schema_versions(version):
    value = _physical_manifest().to_dict()
    value["schema_version"] = version

    with pytest.raises(UnsupportedSplitManifestVersionError, match=str(version)):
        DatasetSplitManifest.from_dict(value)


def test_loader_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "split_manifest.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(TypeError, match="root must be a mapping"):
        load_split_manifest(path)
