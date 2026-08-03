from __future__ import annotations

from pathlib import Path

from src.dataset.metadata import AudioMetadata
from src.fusion.cache import FusionCache


class _DummySerializer:
    def save_npz(self, fused, path):
        raise AssertionError("save should not be called for this test")

    def load_npz(self, path):
        raise AssertionError("load should not be called for this test")


class _DummyPipeline:
    def run(self, path):
        raise AssertionError("pipeline should not run for this test")


class _DummyExtractor:
    def extract(self, waveform):
        raise AssertionError("extractor should not run for this test")


class _DummyVecBuilder:
    def build(self, features):
        raise AssertionError("vector builder should not run for this test")


class _DummyEncoder:
    def encode(self, **kwargs):
        raise AssertionError("encoder should not run for this test")


class _DummyFusion:
    def build(self, **kwargs):
        raise AssertionError("fusion should not run for this test")


def _make_cache():
    return FusionCache(
        cache_root=Path("/tmp/fusion-cache"),
        pipeline=_DummyPipeline(),
        extractor=_DummyExtractor(),
        vec_builder=_DummyVecBuilder(),
        encoder=_DummyEncoder(),
        fusion=_DummyFusion(),
    )


def test_upload_records_bypass_cache(monkeypatch):
    cache = _make_cache()
    cache._serializer = _DummySerializer()

    uploaded = AudioMetadata(
        machine_type="pump",
        machine_id="id_00",
        label="normal",
        filename="uploaded.wav",
        relative_path=Path("/tmp/uploaded.wav"),
        absolute_path=Path("/tmp/uploaded.wav"),
        is_uploaded=True,
    )

    expected = object()

    def fake_compute(rec):
        assert rec is uploaded
        return expected

    monkeypatch.setattr(cache, "exists", lambda rec: (_ for _ in ()).throw(AssertionError("exists should not be called")))
    monkeypatch.setattr(cache, "load", lambda rec: (_ for _ in ()).throw(AssertionError("load should not be called")))
    monkeypatch.setattr(cache, "save", lambda fused, rec: (_ for _ in ()).throw(AssertionError("save should not be called")))
    monkeypatch.setattr(cache, "_compute", fake_compute)

    assert cache.load_or_create(uploaded) is expected


def test_dataset_records_still_use_disk_cache(monkeypatch):
    cache = _make_cache()
    cache._serializer = _DummySerializer()

    dataset_record = AudioMetadata(
        machine_type="pump",
        machine_id="id_00",
        label="normal",
        filename="00000000.wav",
        relative_path=Path("data/raw/MIMII/pump/id_00/normal/00000000.wav"),
        absolute_path=Path("data/raw/MIMII/pump/id_00/normal/00000000.wav"),
    )

    expected = object()

    def fake_exists(rec):
        assert rec is dataset_record
        return True

    def fake_load(rec):
        assert rec is dataset_record
        return expected

    monkeypatch.setattr(cache, "exists", fake_exists)
    monkeypatch.setattr(cache, "load", fake_load)
    monkeypatch.setattr(cache, "_compute", lambda rec: (_ for _ in ()).throw(AssertionError("_compute should not be called")))

    assert cache.load_or_create(dataset_record) is expected
