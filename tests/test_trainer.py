"""Tests for ContrastiveTrainer (src/contrastive_learning/trainer.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import importlib.util
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load(rel: str):
    """Load a module directly from its .py file and register it in sys.modules."""
    path = _ROOT / rel
    name = rel.replace("/", ".").replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load leaf modules that have no heavy transitive dependencies
_fused_vector_mod = _load("src/fusion/fused_vector.py")
_model_mod = _load("src/contrastive_learning/model.py")
_loss_mod = _load("src/contrastive_learning/loss.py")
_serializer_mod = _load("src/contrastive_learning/serializer.py")

FusedFeatureVector = _fused_vector_mod.FusedFeatureVector
ProjectionHead = _model_mod.ProjectionHead
NTXentLoss = _loss_mod.NTXentLoss
ContrastiveSerializer = _serializer_mod.ContrastiveSerializer


# ---------------------------------------------------------------------------
# ContrastivePair — replicated here to avoid the librosa import chain that
# dataset.py triggers through its relative imports.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastivePair:
    anchor: FusedFeatureVector
    paired: FusedFeatureVector
    label: int


# Stub the dataset module so trainer.py's `from .dataset import ...` resolves
# without pulling in librosa.
_dataset_stub = types.ModuleType("src.contrastive_learning.dataset")
_dataset_stub.ContrastivePair = ContrastivePair
_dataset_stub.ContrastiveDataset = None  # trainer only uses it as a type hint
sys.modules["src.contrastive_learning.dataset"] = _dataset_stub

# Also stub the package __init__ so relative imports inside trainer.py work
_pkg_stub = types.ModuleType("src.contrastive_learning")
_pkg_stub.dataset = _dataset_stub
sys.modules.setdefault("src.contrastive_learning", _pkg_stub)

_trainer_mod = _load("src/contrastive_learning/trainer.py")
ContrastiveTrainer = _trainer_mod.ContrastiveTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

_DSP_DIM = 153
_BEATS_DIM = 768
_FUSED_DIM = 921
_DSP_NAMES = [f"feat_{i}" for i in range(_DSP_DIM)]


def _make_fused_vector(machine_type: str = "pump", machine_id: str = "id_00", idx: int = 0) -> FusedFeatureVector:
    dsp = _RNG.random(_DSP_DIM).astype(np.float32)
    beats = _RNG.random(_BEATS_DIM).astype(np.float32)
    fused = np.concatenate([dsp, beats]).astype(np.float32)
    return FusedFeatureVector(
        machine_type=machine_type,
        machine_id=machine_id,
        label="normal",
        filename=f"{machine_type}_{machine_id}_{idx:05d}.wav",
        sample_rate=16_000,
        dsp_feature_names=_DSP_NAMES,
        dsp_feature_vector=dsp,
        beats_embedding=beats,
        fused_feature_vector=fused,
    )


def _make_positive_pair(machine_type: str = "pump", machine_id: str = "id_00", idx: int = 0) -> ContrastivePair:
    return ContrastivePair(
        anchor=_make_fused_vector(machine_type, machine_id, idx * 2),
        paired=_make_fused_vector(machine_type, machine_id, idx * 2 + 1),
        label=1,
    )


def _make_pairs(n: int, machine_type: str = "pump", machine_id: str = "id_00") -> list[ContrastivePair]:
    return [_make_positive_pair(machine_type, machine_id, i) for i in range(n)]


def _make_multi_machine_pairs(
    n_per_machine: int = 3,
    n_machines: int = 4,
) -> list[ContrastivePair]:
    """Create pairs spread across multiple machine IDs so batches can be filled."""
    pairs: list[ContrastivePair] = []
    for m in range(n_machines):
        machine_id = f"id_{m:02d}"
        for i in range(n_per_machine):
            pairs.append(_make_positive_pair("pump", machine_id, m * 100 + i))
    return pairs


class FakeContrastiveDataset:
    def __init__(
        self,
        train_positive_pairs: list[ContrastivePair],
        val_positive_pairs: list[ContrastivePair],
    ) -> None:
        self.train_positive_pairs = train_positive_pairs
        self.val_positive_pairs = val_positive_pairs

    @property
    def positive_pairs(self) -> list[ContrastivePair]:
        return self.train_positive_pairs + self.val_positive_pairs


def _make_trainer(tmp_path: Path, epochs: int = 2, batch_size: int = 4) -> ContrastiveTrainer:
    torch.manual_seed(0)
    head = ProjectionHead()
    criterion = NTXentLoss(temperature=0.1)
    return ContrastiveTrainer(
        head=head,
        criterion=criterion,
        learning_rate=1e-3,
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=tmp_path,
    )


def _make_dataset(n_train: int = 10, n_val: int = 4) -> FakeContrastiveDataset:
    return FakeContrastiveDataset(
        train_positive_pairs=_make_pairs(n_train),
        val_positive_pairs=_make_pairs(n_val),
    )


def _make_multi_machine_dataset(
    n_per_machine: int = 3,
    n_machines: int = 4,
    n_val: int = 4,
) -> FakeContrastiveDataset:
    """Dataset with train pairs spread across multiple machines (needed for non-empty batches)."""
    return FakeContrastiveDataset(
        train_positive_pairs=_make_multi_machine_pairs(n_per_machine, n_machines),
        val_positive_pairs=_make_multi_machine_pairs(2, 2),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainerInitialization:
    def test_initialization_succeeds(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer is not None

    def test_invalid_learning_rate_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ContrastiveTrainer(ProjectionHead(), NTXentLoss(), learning_rate=0.0, checkpoint_dir=tmp_path)

    def test_invalid_batch_size_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ContrastiveTrainer(ProjectionHead(), NTXentLoss(), batch_size=1, checkpoint_dir=tmp_path)

    def test_invalid_epochs_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ContrastiveTrainer(ProjectionHead(), NTXentLoss(), epochs=0, checkpoint_dir=tmp_path)

    def test_invalid_val_split_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ContrastiveTrainer(ProjectionHead(), NTXentLoss(), val_split=0.0, checkpoint_dir=tmp_path)


class TestTrainingCompletes:
    def test_fit_completes(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.fit(_make_dataset())

    def test_history_keys(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.fit(_make_dataset())
        h = trainer.history()
        assert "training_losses" in h
        assert "validation_losses" in h

    def test_history_length_equals_epochs(self, tmp_path):
        epochs = 3
        trainer = _make_trainer(tmp_path, epochs=epochs)
        trainer.fit(_make_dataset())
        h = trainer.history()
        assert len(h["training_losses"]) == epochs
        assert len(h["validation_losses"]) == epochs


class TestLossValues:
    @pytest.fixture(scope="class")
    @classmethod
    def trained_history(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("loss")
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit(_make_multi_machine_dataset())
        return trainer.history()

    def test_training_losses_are_finite(self, trained_history):
        assert all(np.isfinite(v) for v in trained_history["training_losses"])

    def test_validation_losses_are_finite(self, trained_history):
        assert all(np.isfinite(v) for v in trained_history["validation_losses"])


class TestCheckpoint:
    @pytest.fixture(scope="class")
    @classmethod
    def checkpoint_path(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("ckpt")
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit(_make_multi_machine_dataset())
        return tmp / "best_projection_head.pt"

    def test_checkpoint_file_exists(self, checkpoint_path):
        assert checkpoint_path.exists()

    def test_checkpoint_required_keys(self, checkpoint_path):
        ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
        assert "model_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "validation_loss" in ckpt

    def test_checkpoint_optimizer_state_exists(self, checkpoint_path):
        ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
        assert "optimizer_state_dict" in ckpt
        assert ckpt["optimizer_state_dict"] is not None

    def test_checkpoint_epoch_is_valid(self, checkpoint_path):
        ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
        assert isinstance(ckpt["epoch"], int)
        assert ckpt["epoch"] >= 1

    def test_checkpoint_validation_loss_is_finite(self, checkpoint_path):
        ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
        assert np.isfinite(ckpt["validation_loss"])


class TestEdgeCases:
    def test_too_few_train_pairs_raises(self, tmp_path):
        dataset = FakeContrastiveDataset(
            train_positive_pairs=_make_pairs(1),
            val_positive_pairs=_make_pairs(4),
        )
        trainer = _make_trainer(tmp_path)
        with pytest.raises(ValueError):
            trainer.fit(dataset)

    def test_too_few_val_pairs_raises(self, tmp_path):
        dataset = FakeContrastiveDataset(
            train_positive_pairs=_make_pairs(10),
            val_positive_pairs=_make_pairs(1),
        )
        trainer = _make_trainer(tmp_path)
        with pytest.raises(ValueError):
            trainer.fit(dataset)


class TestMakeBatches:
    """Tests for ContrastiveTrainer._make_machine_aware_batches.

    The batcher groups by (machine_type, machine_id) and takes at most one
    pair per machine per batch slot.  Tests therefore use multiple machine IDs
    so that batches can be filled beyond a single pair.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def trainer(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("batches")
        return _make_trainer(tmp, batch_size=4)

    def test_batches_respect_batch_size(self, trainer):
        # 4 machines × 3 pairs each = 12 pairs; batch_size=4
        pairs = _make_multi_machine_pairs(n_per_machine=3, n_machines=4)
        batches = trainer._make_machine_aware_batches(pairs)
        for batch in batches:
            assert len(batch) <= 4

    def test_all_pairs_covered(self, trainer):
        # 4 machines × 2 pairs each = 8 pairs; all should appear in batches
        pairs = _make_multi_machine_pairs(n_per_machine=2, n_machines=4)
        batches = trainer._make_machine_aware_batches(pairs)
        total = sum(len(b) for b in batches)
        assert total == len(pairs)

    def test_final_batch_with_one_pair_is_dropped(self, trainer):
        # 5 machines × 1 pair each = 5 pairs, batch_size=4
        # → first batch has 4 pairs, second has 1 (dropped)
        pairs = _make_multi_machine_pairs(n_per_machine=1, n_machines=5)
        batches = trainer._make_machine_aware_batches(pairs)
        for batch in batches:
            assert len(batch) >= 2

    def test_final_batch_with_two_pairs_is_kept(self, trainer):
        # 6 machines × 1 pair each = 6 pairs, batch_size=4
        # → first batch 4, second batch 2 (kept)
        pairs = _make_multi_machine_pairs(n_per_machine=1, n_machines=6)
        batches = trainer._make_machine_aware_batches(pairs)
        assert len(batches) == 2
        assert len(batches[-1]) == 2


class TestMultipleEpochs:
    def test_history_has_one_entry_per_epoch(self, tmp_path):
        for epochs in (1, 3, 5):
            trainer = _make_trainer(tmp_path / str(epochs), epochs=epochs)
            trainer.fit(_make_dataset())
            h = trainer.history()
            assert len(h["training_losses"]) == epochs
            assert len(h["validation_losses"]) == epochs


class TestModelActuallyTrains:
    def test_parameters_change_after_training(self, tmp_path):
        torch.manual_seed(0)
        head = ProjectionHead()
        before = head.net[0].weight.detach().clone()

        trainer = ContrastiveTrainer(
            head=head,
            criterion=NTXentLoss(temperature=0.1),
            learning_rate=1e-3,
            batch_size=4,
            epochs=1,
            checkpoint_dir=tmp_path,
        )
        trainer.fit(_make_multi_machine_dataset())

        after = head.net[0].weight.detach()
        assert not torch.equal(before, after), "Weights did not change after training"
