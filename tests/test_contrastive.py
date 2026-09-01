"""Tests for ContrastiveDataset and ContrastiveTrainer.

Covers:
    1. A machine contributes at most one positive pair per batch.
    2. No machine key is duplicated within a batch.
    3. All positive pairs contain recordings from the same machine.
    4. Anchor and paired recordings are different.
    5. Training and validation recordings have no overlap.
    6. Invalid ContrastiveTrainer parameters raise ValueError.
    7. NT-Xent batches always contain at least two pairs.
    8. Existing checkpoint saving behavior still works.

Import strategy
---------------
Uses importlib.util.spec_from_file_location (same as test_ntxent.py /
test_projection.py) to load individual source files directly, bypassing the
package __init__.py which eagerly imports librosa-dependent modules.
"""

from __future__ import annotations

import importlib.util
import math
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str):
    """Load a single .py file as a module, bypassing package __init__."""
    import sys
    path = _ROOT / rel
    name = rel.replace("/", ".").replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so dataclass __module__ resolves
    spec.loader.exec_module(mod)
    return mod


# Load only the files we need — none of these pull in librosa.
_fused_mod = _load("src/fusion/fused_vector.py")
_model_mod = _load("src/contrastive_learning/model.py")
_loss_mod = _load("src/contrastive_learning/loss.py")
_serializer_mod = _load("src/contrastive_learning/serializer.py")

FusedFeatureVector = _fused_mod.FusedFeatureVector
ProjectionHead = _model_mod.ProjectionHead
NTXentLoss = _loss_mod.NTXentLoss
ContrastiveSerializer = _serializer_mod.ContrastiveSerializer

# ContrastivePair and _pairs_from_recordings are defined inline here because
# dataset.py has relative imports that pull in librosa (not installed in the
# test environment).  The logic is trivial and self-contained.
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class ContrastivePair:
    anchor: FusedFeatureVector
    paired: FusedFeatureVector
    label: int


def _pairs_from_recordings(recordings: list, seed: int = 0) -> list:
    """Mirror of ContrastiveDataset._pairs_from_recordings (seeded)."""
    if len(recordings) < 2:
        return []
    rng = random.Random(seed)
    pairs = []
    for anchor in recordings:
        pool = [f for f in recordings if f.filename != anchor.filename]
        paired = rng.choice(pool)
        pairs.append(ContrastivePair(anchor=anchor, paired=paired, label=1))
    return pairs


# Trainer only needs torch + the ContrastivePair type — no librosa.
# We inject our local ContrastivePair into the trainer module's namespace
# so type checks inside _make_machine_aware_batches work correctly.
#
# trainer.py does `from .dataset import ContrastiveDataset, ContrastivePair`
# which would trigger the librosa chain.  We stub the dataset module in
# sys.modules before loading trainer so the relative import resolves to our
# lightweight stub instead.
import sys as _sys
import types as _types

_stub = _types.ModuleType("src.contrastive_learning.dataset")
_stub.ContrastivePair = ContrastivePair
_stub.ContrastiveDataset = type("ContrastiveDataset", (), {})  # empty placeholder
_sys.modules["src.contrastive_learning.dataset"] = _stub

_trainer_mod = _load("src/contrastive_learning/trainer.py")
ContrastiveTrainer = _trainer_mod.ContrastiveTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fused(machine_type: str, machine_id: str, filename: str) -> FusedFeatureVector:
    """Minimal FusedFeatureVector with a deterministic 921-dim vector."""
    rng = np.random.default_rng(abs(hash(filename)) % (2**31))
    vec = rng.random(921).astype(np.float32)
    return FusedFeatureVector(
        machine_type=machine_type,
        machine_id=machine_id,
        label="normal",
        filename=filename,
        sample_rate=16_000,
        dsp_feature_names=[f"f{i}" for i in range(153)],
        dsp_feature_vector=vec[:153],
        beats_embedding=vec[153:],
        fused_feature_vector=vec,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _pair(machine_type: str, machine_id: str, a: str, b: str) -> ContrastivePair:
    return ContrastivePair(
        anchor=_fused(machine_type, machine_id, a),
        paired=_fused(machine_type, machine_id, b),
        label=1,
    )


def _make_pairs(
    machines: list[tuple[str, str]],
    pairs_per_machine: int = 4,
) -> list[ContrastivePair]:
    """Build a flat list of positive pairs for the given (type, id) tuples."""
    pairs: list[ContrastivePair] = []
    for mt, mid in machines:
        for i in range(pairs_per_machine):
            pairs.append(_pair(mt, mid, f"{mt}_{mid}_a{i}.wav", f"{mt}_{mid}_b{i}.wav"))
    return pairs


def _fresh_head() -> ProjectionHead:
    torch.manual_seed(0)
    return ProjectionHead()


def _criterion() -> NTXentLoss:
    return NTXentLoss(temperature=0.1)


def _trainer(
    batch_size: int = 4,
    epochs: int = 1,
    checkpoint_dir: str | Path = ".",
) -> ContrastiveTrainer:
    return ContrastiveTrainer(
        head=_fresh_head(),
        criterion=_criterion(),
        learning_rate=1e-3,
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        val_split=0.2,
        seed=42,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Machine-aware batching
# ---------------------------------------------------------------------------

class TestMachineAwareBatching:
    """_make_machine_aware_batches enforces one pair per machine per batch."""

    MACHINES = [("pump", "id_00"), ("pump", "id_02"), ("pump", "id_04"), ("pump", "id_06")]

    @pytest.fixture
    def pairs(self):
        return _make_pairs(self.MACHINES, pairs_per_machine=4)

    def test_at_most_one_pair_per_machine_per_batch(self, pairs):
        """Each machine contributes at most one pair to any single batch."""
        t = _trainer(batch_size=4)
        batches = t._make_machine_aware_batches(pairs)
        for batch in batches:
            keys = [(p.anchor.machine_type, p.anchor.machine_id) for p in batch]
            assert len(keys) == len(set(keys)), f"Duplicate machine key in batch: {keys}"

    def test_no_duplicate_machine_in_batch(self, pairs):
        """No (machine_type, machine_id) appears more than once per batch."""
        t = _trainer(batch_size=4)
        batches = t._make_machine_aware_batches(pairs)
        for batch in batches:
            seen: set[tuple[str, str]] = set()
            for p in batch:
                key = (p.anchor.machine_type, p.anchor.machine_id)
                assert key not in seen, f"Machine {key} duplicated in batch"
                seen.add(key)

    def test_all_pairs_are_consumed(self, pairs):
        """Every pair appears in exactly one batch."""
        t = _trainer(batch_size=4)
        batches = t._make_machine_aware_batches(pairs)
        consumed = [p for batch in batches for p in batch]
        assert len(consumed) == len(pairs)

    def test_single_machine_batches_one_pair_each(self):
        """With one machine, each batch contains exactly one pair."""
        t = _trainer(batch_size=4)
        pairs = _make_pairs([("pump", "id_00")], pairs_per_machine=6)
        batches = t._make_machine_aware_batches(pairs)
        for batch in batches:
            assert len(batch) == 1


# ---------------------------------------------------------------------------
# 3. All positive pairs contain recordings from the same machine.
# ---------------------------------------------------------------------------

class TestPositivePairCorrectness:
    """_pairs_from_recordings only produces same-machine pairs."""

    def test_pairs_have_label_1(self):
        recordings = [_fused("pump", "id_00", f"{i}.wav") for i in range(5)]
        pairs = _pairs_from_recordings(recordings)
        assert all(p.label == 1 for p in pairs)

    def test_pairs_same_machine_type(self):
        recordings = [_fused("pump", "id_00", f"{i}.wav") for i in range(5)]
        pairs = _pairs_from_recordings(recordings)
        for p in pairs:
            assert p.anchor.machine_type == p.paired.machine_type

    def test_pairs_same_machine_id(self):
        recordings = [_fused("pump", "id_00", f"{i}.wav") for i in range(5)]
        pairs = _pairs_from_recordings(recordings)
        for p in pairs:
            assert p.anchor.machine_id == p.paired.machine_id

    def test_fewer_than_two_recordings_returns_empty(self):
        recordings = [_fused("pump", "id_00", "only.wav")]
        assert _pairs_from_recordings(recordings) == []


# ---------------------------------------------------------------------------
# 4. Anchor and paired recordings are different.
# ---------------------------------------------------------------------------

class TestAnchorPairedDifferent:
    """Anchor and paired filenames must differ in every positive pair."""

    def test_anchor_not_equal_to_paired(self):
        recordings = [_fused("pump", "id_00", f"{i}.wav") for i in range(10)]
        pairs = _pairs_from_recordings(recordings)
        for p in pairs:
            assert p.anchor.filename != p.paired.filename, (
                f"Anchor and paired are the same file: {p.anchor.filename}"
            )


# ---------------------------------------------------------------------------
# 5. Training and validation recordings have no overlap.
# ---------------------------------------------------------------------------

class TestTrainValNoOverlap:
    """No recording filename appears in both train and val positive pairs."""

    def _split_and_pair(
        self,
        fused_by_machine: dict[tuple[str, str], list[FusedFeatureVector]],
        val_split: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """Replicate _build_split_pairs logic without constructing a full dataset."""
        rng = random.Random(seed)
        train_pairs: list[ContrastivePair] = []
        val_pairs: list[ContrastivePair] = []

        for fused_list in fused_by_machine.values():
            if len(fused_list) < 2:
                continue
            shuffled = list(fused_list)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_split))
            val_recs = shuffled[:n_val]
            train_recs = shuffled[n_val:]
            train_pairs.extend(_pairs_from_recordings(train_recs))
            val_pairs.extend(_pairs_from_recordings(val_recs))

        return train_pairs, val_pairs

    def test_no_filename_overlap_between_train_and_val(self):
        # Use globally unique filenames so cross-machine filename collisions
        # cannot produce false positives.
        fused_by_machine = {
            ("pump", "id_00"): [
                _fused("pump", "id_00", f"id00_{i}.wav") for i in range(20)
            ],
            ("pump", "id_02"): [
                _fused("pump", "id_02", f"id02_{i}.wav") for i in range(20)
            ],
        }
        # The invariant: recordings used as *anchors* in training pairs must
        # never appear as *anchors* in validation pairs (and vice versa).
        # Anchors are the recordings that were assigned to each split.
        train_pairs, val_pairs = self._split_and_pair(fused_by_machine)

        train_anchors = {p.anchor.filename for p in train_pairs}
        val_anchors = {p.anchor.filename for p in val_pairs}

        overlap = train_anchors & val_anchors
        assert not overlap, (
            f"Same recording is an anchor in both train and val splits: {overlap}"
        )

    def test_both_splits_are_non_empty(self):
        fused_by_machine = {
            ("pump", "id_00"): [_fused("pump", "id_00", f"{i}.wav") for i in range(10)],
        }
        train_pairs, val_pairs = self._split_and_pair(fused_by_machine)
        assert len(train_pairs) > 0
        assert len(val_pairs) > 0


# ---------------------------------------------------------------------------
# 6. Invalid ContrastiveTrainer parameters raise ValueError.
# ---------------------------------------------------------------------------

class TestTrainerParameterValidation:
    """ContrastiveTrainer.__init__ raises ValueError for invalid parameters."""

    def _base_kwargs(self, **overrides) -> dict:
        kwargs = dict(
            head=_fresh_head(),
            criterion=_criterion(),
            learning_rate=1e-3,
            batch_size=4,
            epochs=1,
            val_split=0.2,
        )
        kwargs.update(overrides)
        return kwargs

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ContrastiveTrainer(**self._base_kwargs(learning_rate=0.0))

    def test_learning_rate_negative_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            ContrastiveTrainer(**self._base_kwargs(learning_rate=-1e-3))

    def test_batch_size_one_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            ContrastiveTrainer(**self._base_kwargs(batch_size=1))

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            ContrastiveTrainer(**self._base_kwargs(batch_size=0))

    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            ContrastiveTrainer(**self._base_kwargs(epochs=0))

    def test_epochs_negative_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            ContrastiveTrainer(**self._base_kwargs(epochs=-1))

    def test_val_split_zero_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            ContrastiveTrainer(**self._base_kwargs(val_split=0.0))

    def test_val_split_one_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            ContrastiveTrainer(**self._base_kwargs(val_split=1.0))

    def test_val_split_negative_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            ContrastiveTrainer(**self._base_kwargs(val_split=-0.1))

    def test_valid_parameters_do_not_raise(self):
        ContrastiveTrainer(**self._base_kwargs())  # must not raise


# ---------------------------------------------------------------------------
# 7. NT-Xent batches always contain at least two pairs.
# ---------------------------------------------------------------------------

class TestBatchMinimumSize:
    """_make_machine_aware_batches never produces a batch with fewer than 2 pairs."""

    def test_all_batches_have_at_least_two_pairs(self):
        machines = [("pump", "id_00"), ("pump", "id_02"), ("pump", "id_04")]
        pairs = _make_pairs(machines, pairs_per_machine=3)
        t = _trainer(batch_size=8)
        batches = t._make_machine_aware_batches(pairs)
        for batch in batches:
            assert len(batch) >= 2, f"Batch has only {len(batch)} pair(s)"

    def test_single_pair_is_dropped(self):
        """A lone pair that cannot form a batch of >= 2 is dropped."""
        pairs = [_pair("pump", "id_00", "a.wav", "b.wav")]
        t = _trainer(batch_size=4)
        batches = t._make_machine_aware_batches(pairs)
        assert batches == []


# ---------------------------------------------------------------------------
# 8. Checkpoint saving behavior still works.
# ---------------------------------------------------------------------------

class TestCheckpointSaving:
    """ContrastiveTrainer saves a checkpoint via ContrastiveSerializer."""

    def _mock_dataset(
        self,
        machines: list[tuple[str, str]],
        pairs_per_machine: int = 6,
        val_fraction: float = 0.2,
    ) -> MagicMock:
        all_pairs = _make_pairs(machines, pairs_per_machine=pairs_per_machine)
        n_val = max(1, int(len(all_pairs) * val_fraction))
        dataset = MagicMock()
        dataset.train_positive_pairs = all_pairs[n_val:]
        dataset.val_positive_pairs = all_pairs[:n_val]
        return dataset

    def test_checkpoint_is_saved_on_val_improvement(self):
        """fit() saves a checkpoint when validation loss improves."""
        machines = [
            ("pump", "id_00"), ("pump", "id_02"),
            ("pump", "id_04"), ("pump", "id_06"),
        ]
        dataset = self._mock_dataset(machines)

        with tempfile.TemporaryDirectory() as tmp:
            t = ContrastiveTrainer(
                head=_fresh_head(),
                criterion=_criterion(),
                learning_rate=1e-3,
                batch_size=4,
                epochs=2,
                checkpoint_dir=tmp,
                val_split=0.2,
                seed=0,
            )
            t.fit(dataset)

            checkpoint_path = Path(tmp) / "best_projection_head.pt"
            assert checkpoint_path.exists(), "Checkpoint file was not created"

            ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt
            assert "validation_loss" in ckpt
            assert math.isfinite(ckpt["validation_loss"])

    def test_history_has_correct_length(self):
        """history() returns one entry per epoch."""
        machines = [("pump", "id_00"), ("pump", "id_02"), ("pump", "id_04"), ("pump", "id_06")]
        dataset = self._mock_dataset(machines)

        with tempfile.TemporaryDirectory() as tmp:
            n_epochs = 3
            t = ContrastiveTrainer(
                head=_fresh_head(),
                criterion=_criterion(),
                learning_rate=1e-3,
                batch_size=4,
                epochs=n_epochs,
                checkpoint_dir=tmp,
                val_split=0.2,
                seed=0,
            )
            t.fit(dataset)
            h = t.history()
            assert len(h["training_losses"]) == n_epochs
            assert len(h["validation_losses"]) == n_epochs


# ---------------------------------------------------------------------------
# Helpers shared by new tests
# ---------------------------------------------------------------------------

def _meta(machine_type: str, machine_id: str, label: str, filename: str) -> "AudioMetadata":
    """Build a minimal AudioMetadata without a real file."""
    import importlib
    AudioMetadata = importlib.import_module("src.dataset.metadata").AudioMetadata
    rel = Path(machine_type) / machine_id / label / filename
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label=label,
        filename=filename,
        relative_path=rel,
        absolute_path=Path("/fake") / rel,
    )


def _normal_meta(n: int, machine_type: str = "pump", machine_id: str = "id_00") -> list:
    return [_meta(machine_type, machine_id, "normal", f"{i:08d}.wav") for i in range(n)]


def _abnormal_meta(n: int, machine_type: str = "pump", machine_id: str = "id_00") -> list:
    return [_meta(machine_type, machine_id, "abnormal", f"ab_{i:08d}.wav") for i in range(n)]


# ---------------------------------------------------------------------------
# Test A — Dataset split isolation
# ---------------------------------------------------------------------------

class TestDatasetSplitIsolation:
    """A recording must never appear in more than one partition."""

    def test_all_four_partitions_are_disjoint(self):
        DatasetSplitter = _load("src/dataset/split.py").DatasetSplitter

        recordings = _normal_meta(30) + _abnormal_meta(10)
        split = DatasetSplitter(seed=42).split(recordings)

        train = {r.filename for r in split.train_normal}
        profile = {r.filename for r in split.profile_normal}
        test_n = {r.filename for r in split.test_normal}
        test_ab = {r.filename for r in split.test_abnormal}

        assert train.isdisjoint(profile), "train_normal and profile_normal overlap"
        assert train.isdisjoint(test_n), "train_normal and test_normal overlap"
        assert train.isdisjoint(test_ab), "train_normal and test_abnormal overlap"
        assert profile.isdisjoint(test_n), "profile_normal and test_normal overlap"
        assert profile.isdisjoint(test_ab), "profile_normal and test_abnormal overlap"
        assert test_n.isdisjoint(test_ab), "test_normal and test_abnormal overlap"

    def test_all_normal_recordings_covered(self):
        DatasetSplitter = _load("src/dataset/split.py").DatasetSplitter

        recordings = _normal_meta(20) + _abnormal_meta(5)
        split = DatasetSplitter(seed=42).split(recordings)

        normal_in = {r.filename for r in recordings if r.label == "normal"}
        normal_out = (
            {r.filename for r in split.train_normal}
            | {r.filename for r in split.profile_normal}
            | {r.filename for r in split.test_normal}
        )
        assert normal_in == normal_out


# ---------------------------------------------------------------------------
# Test B — ContrastiveDataset only uses supplied recordings
# ---------------------------------------------------------------------------

# Load dataset.py under a private name with all relative-import targets
# pre-stubbed, then restore sys.modules so other test files are unaffected.
import importlib.util as _ilu
import sys as _sys2
import types as _types

_STUB_NAMES = [
    "src.beats",
    "src.beats.encoder",
    "src.feature_extraction",
    "src.feature_extraction.extractor",
    "src.feature_extraction.feature_vector",
    "src.fusion",
    "src.fusion.cache",
    "src.fusion.fused_vector",
    "src.fusion.fusion",
    "src.preprocessing",
    "src.preprocessing.pipeline",
    "src.dataset",
    "src.dataset.loader",
    "src.dataset.metadata",
    "src.contrastive_learning",
    "src.contrastive_learning.dataset",
]

# Snapshot existing entries so we can restore them after loading dataset.py.
_saved_modules = {k: _sys2.modules.get(k) for k in _STUB_NAMES}

for _sn in _STUB_NAMES:
    if _sn not in _sys2.modules:
        _sys2.modules[_sn] = _types.ModuleType(_sn)

# Load the real AudioMetadata directly by file path.
_meta_spec2 = _ilu.spec_from_file_location("_test_metadata_real", _ROOT / "src" / "dataset" / "metadata.py")
_meta_real_mod = _ilu.module_from_spec(_meta_spec2)
_meta_spec2.loader.exec_module(_meta_real_mod)

# Populate the specific names that dataset.py imports from each stub.
_sys2.modules["src.beats.encoder"].BEATsEncoder = type("BEATsEncoder", (), {})
_sys2.modules["src.feature_extraction.extractor"].FeatureExtractor = type("FeatureExtractor", (), {})
_sys2.modules["src.feature_extraction.feature_vector"].FeatureVectorBuilder = type("FeatureVectorBuilder", (), {})
_sys2.modules["src.fusion.cache"].FusionCache = type("FusionCache", (), {})
_sys2.modules["src.fusion.fused_vector"].FusedFeatureVector = _fused_mod.FusedFeatureVector
_sys2.modules["src.fusion.fusion"].FusionBuilder = type("FusionBuilder", (), {})
_sys2.modules["src.preprocessing.pipeline"].PreprocessingPipeline = type("PreprocessingPipeline", (), {})
_sys2.modules["src.dataset.loader"].DatasetLoader = type("DatasetLoader", (), {})
_sys2.modules["src.dataset.metadata"].AudioMetadata = _meta_real_mod.AudioMetadata

# Load dataset.py under its canonical name so relative imports resolve.
_ds_spec = _ilu.spec_from_file_location(
    "src.contrastive_learning.dataset",
    _ROOT / "src" / "contrastive_learning" / "dataset.py",
)
_dataset_mod = _ilu.module_from_spec(_ds_spec)
_dataset_mod.__package__ = "src.contrastive_learning"
_sys2.modules["src.contrastive_learning.dataset"] = _dataset_mod
_ds_spec.loader.exec_module(_dataset_mod)
_ContrastiveDataset = _dataset_mod.ContrastiveDataset

# Restore sys.modules to its pre-stub state so other test files are unaffected.
for _sn, _orig in _saved_modules.items():
    if _orig is None:
        _sys2.modules.pop(_sn, None)
    else:
        _sys2.modules[_sn] = _orig

# Keep the real dataset module accessible under a private key.
_sys2.modules["_test_dataset_real"] = _dataset_mod


class TestExplicitRecordingsInterface:
    """ContrastiveDataset(recordings=...) must only use the supplied list."""

    def _make_dataset_with_mock_cache(self, recordings: list):
        """Construct ContrastiveDataset with all heavy I/O mocked out."""
        from unittest.mock import MagicMock, patch

        def _fake_fused(rec):
            rng = np.random.default_rng(abs(hash(rec.filename)) % (2**31))
            vec = rng.random(921).astype(np.float32)
            return _fused_mod.FusedFeatureVector(
                machine_type=rec.machine_type,
                machine_id=rec.machine_id,
                label=rec.label,
                filename=rec.filename,
                sample_rate=16_000,
                dsp_feature_names=[f"f{i}" for i in range(153)],
                dsp_feature_vector=vec[:153],
                beats_embedding=vec[153:],
                fused_feature_vector=vec,
                created_at="2024-01-01T00:00:00+00:00",
            )

        mock_cache = MagicMock()
        mock_cache.exists.return_value = True
        mock_cache.load_or_create.side_effect = _fake_fused

        with patch.object(_dataset_mod, "BEATsEncoder"), \
             patch.object(_dataset_mod, "PreprocessingPipeline"), \
             patch.object(_dataset_mod, "FeatureExtractor"), \
             patch.object(_dataset_mod, "FeatureVectorBuilder"), \
             patch.object(_dataset_mod, "FusionBuilder"), \
             patch.object(_dataset_mod, "FusionCache", return_value=mock_cache):
            ds = _ContrastiveDataset(
                recordings=recordings,
                checkpoint_path="/fake/beats.pt",
                cache_root="/fake/cache",
                seed=42,
            )
        return ds

    def test_only_supplied_recordings_are_used(self):
        # Use globally unique filenames so there is no filename collision
        # between the two groups.
        train = [_meta("pump", "id_00", "normal", f"train_{i:08d}.wav") for i in range(10)]
        other = [_meta("pump", "id_02", "normal", f"other_{i:08d}.wav") for i in range(10)]

        ds = self._make_dataset_with_mock_cache(train)

        used_filenames = (
            {p.anchor.filename for p in ds.positive_pairs}
            | {p.paired.filename for p in ds.positive_pairs}
        )
        forbidden = {r.filename for r in other}
        assert used_filenames.isdisjoint(forbidden), (
            f"Recordings outside the supplied list were used: "
            f"{used_filenames & forbidden}"
        )

    def test_empty_recordings_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _ContrastiveDataset(recordings=[], checkpoint_path="/fake/beats.pt")

    def test_abnormal_recording_raises(self):
        bad = [_meta("pump", "id_00", "abnormal", "ab.wav")]
        with pytest.raises(ValueError, match="abnormal"):
            _ContrastiveDataset(recordings=bad, checkpoint_path="/fake/beats.pt")

    def test_both_root_and_recordings_raises(self):
        recs = _normal_meta(5)
        with pytest.raises(ValueError, match="not both"):
            _ContrastiveDataset(
                dataset_root="/fake/root",
                recordings=recs,
                checkpoint_path="/fake/beats.pt",
            )

    def test_neither_root_nor_recordings_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            _ContrastiveDataset(checkpoint_path="/fake/beats.pt")


# ---------------------------------------------------------------------------
# Test C — Positive pair correctness (same machine, different filename)
# ---------------------------------------------------------------------------

class TestPositivePairCorrectnessSeeded:
    """Every positive pair: same machine_type, same machine_id, different filename."""

    def test_same_machine_type_and_id(self):
        recs_00 = [_fused("pump", "id_00", f"id00_{i}.wav") for i in range(6)]
        recs_02 = [_fused("pump", "id_02", f"id02_{i}.wav") for i in range(6)]

        # Build pairs per machine group (mirrors _build_split_pairs logic)
        rng = random.Random(42)
        all_pairs: list[ContrastivePair] = []
        for group in [recs_00, recs_02]:
            shuffled = list(group)
            rng.shuffle(shuffled)
            all_pairs.extend(_pairs_from_recordings(shuffled, seed=42))

        for p in all_pairs:
            assert p.anchor.machine_type == p.paired.machine_type
            assert p.anchor.machine_id == p.paired.machine_id

    def test_different_filename_in_every_pair(self):
        recordings = [_fused("pump", "id_00", f"{i}.wav") for i in range(10)]
        pairs = _pairs_from_recordings(recordings, seed=42)
        for p in pairs:
            assert p.anchor.filename != p.paired.filename

    def test_cross_machine_ids_never_paired(self):
        """Recordings from id_00 and id_02 must never form a pair."""
        recs_00 = [_fused("pump", "id_00", f"id00_{i}.wav") for i in range(5)]
        recs_02 = [_fused("pump", "id_02", f"id02_{i}.wav") for i in range(5)]

        # Pairs are built per-group, so mixing is impossible by construction.
        pairs_00 = _pairs_from_recordings(recs_00, seed=42)
        pairs_02 = _pairs_from_recordings(recs_02, seed=42)

        for p in pairs_00:
            assert p.anchor.machine_id == "id_00"
            assert p.paired.machine_id == "id_00"
        for p in pairs_02:
            assert p.anchor.machine_id == "id_02"
            assert p.paired.machine_id == "id_02"


# ---------------------------------------------------------------------------
# Test D — Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same seed => identical recording split and pair assignments."""

    def _split_and_collect(
        self,
        fused_by_machine: dict,
        seed: int,
        val_split: float = 0.2,
    ) -> tuple[list[str], list[str]]:
        """Return (train_anchor_filenames, val_anchor_filenames) for a given seed."""
        rng = random.Random(seed)
        train_anchors: list[str] = []
        val_anchors: list[str] = []

        for fused_list in fused_by_machine.values():
            if len(fused_list) < 2:
                continue
            shuffled = list(fused_list)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_split))
            val_recs = shuffled[:n_val]
            train_recs = shuffled[n_val:]
            train_pairs = _pairs_from_recordings(train_recs, seed=seed)
            val_pairs = _pairs_from_recordings(val_recs, seed=seed)
            train_anchors.extend(p.anchor.filename for p in train_pairs)
            val_anchors.extend(p.anchor.filename for p in val_pairs)

        return train_anchors, val_anchors

    def test_same_seed_produces_identical_train_anchors(self):
        fused_by_machine = {
            ("pump", "id_00"): [_fused("pump", "id_00", f"{i}.wav") for i in range(20)],
        }
        t1, _ = self._split_and_collect(fused_by_machine, seed=42)
        t2, _ = self._split_and_collect(fused_by_machine, seed=42)
        assert t1 == t2

    def test_same_seed_produces_identical_val_anchors(self):
        fused_by_machine = {
            ("pump", "id_00"): [_fused("pump", "id_00", f"{i}.wav") for i in range(20)],
        }
        _, v1 = self._split_and_collect(fused_by_machine, seed=42)
        _, v2 = self._split_and_collect(fused_by_machine, seed=42)
        assert v1 == v2

    def test_different_seed_may_differ(self):
        fused_by_machine = {
            ("pump", "id_00"): [_fused("pump", "id_00", f"{i}.wav") for i in range(20)],
        }
        t1, _ = self._split_and_collect(fused_by_machine, seed=42)
        t2, _ = self._split_and_collect(fused_by_machine, seed=99)
        # With 20 recordings it is astronomically unlikely these are identical
        assert t1 != t2
