# API Reference — Part 2

Covers public classes in `src/beats`, `src/fusion`, and `src/contrastive_learning`.

---

## Table of Contents

- [src/beats](#srcbeats)
  - [BEATsEmbedding](#beatsembedding)
  - [BEATsEncoder](#beatsencoder)
- [src/fusion](#srcfusion)
  - [FusedFeatureVector](#fusedfeaturevector)
  - [FusionBuilder](#fusionbuilder)
  - [FusionCache](#fusioncache)
  - [FusedVectorSerializer](#fusedvectorserializer)
- [src/contrastive_learning](#srccontrastive_learning)
  - [ContrastivePair](#contrastivepair)
  - [ContrastiveDataset](#contrastivedataset)
  - [NTXentLoss](#ntxentloss)
  - [ProjectionHead](#projectionhead)
  - [EpochResult](#epochresult)
  - [ContrastiveTrainer](#contrastivetrainer)
  - [ContrastiveInference](#contrastiveinference)
  - [ContrastiveSerializer](#contrastiveserializer)

---

## src/beats

### BEATsEmbedding

**File**: `src/beats/embedding.py`

**Purpose**: Frozen dataclass holding the 768-dimensional embedding produced by the BEATs encoder for a single audio recording, along with its provenance metadata.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `vector` | `ndarray` shape `(768,)` | L2-normalized embedding vector |
| `embedding_dim` | `int` | Dimensionality of the vector (always 768) |
| `filename` | `str` | Source audio filename |
| `machine_type` | `str` | Machine type label (e.g. `"pump"`) |
| `machine_id` | `str` | Machine ID label (e.g. `"id_00"`) |
| `sample_rate` | `int` | Sample rate of the source waveform |
| `created_at` | `str` | ISO 8601 timestamp of creation |

**Public Methods**

---

#### `to_dict`

Serializes the embedding to a JSON-compatible dictionary.

**Arguments**: None

**Returns**: `dict` — all fields with `vector` converted to a Python list.

---

#### `from_dict`

Reconstructs a `BEATsEmbedding` from a dictionary.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` | Dictionary previously produced by `to_dict()` |

**Returns**: `BEATsEmbedding`

---

**Example**

```python
from src.beats.embedding import BEATsEmbedding

emb = BEATsEmbedding(
    vector=my_array,
    embedding_dim=768,
    filename="00000000.wav",
    machine_type="pump",
    machine_id="id_00",
    sample_rate=16000,
    created_at="2024-01-01T00:00:00",
)

d = emb.to_dict()
restored = BEATsEmbedding.from_dict(d)
```

---

### BEATsEncoder

**File**: `src/beats/encoder.py`

**Purpose**: Wraps the pretrained, frozen BEATs model. Accepts a raw waveform and returns a `BEATsEmbedding`. The model weights are never updated during any training step.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str \| Path` | Path to `BEATs_iter3_plus_AS2M.pt` |

**Raises**

- `FileNotFoundError` — checkpoint file does not exist at the given path.
- `RuntimeError` — checkpoint cannot be loaded or the model fails to initialize.

**Public Methods**

---

#### `encode`

Encodes a single waveform into a 768-dimensional embedding.

**Arguments**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `waveform` | `ndarray` | — | 1-D mono waveform array |
| `sample_rate` | `int` | — | Sample rate of the waveform |
| `filename` | `str` | `""` | Optional source filename stored in the result |

**Returns**: `BEATsEmbedding` with `vector` shape `(768,)`.

**Raises**

- `ValueError` — waveform is empty or has an unsupported shape.
- `RuntimeError` — inference fails inside the BEATs model.

---

#### `embedding_dim` *(property)*

**Returns**: `int` — always `768`.

---

**Example**

```python
from src.beats.encoder import BEATsEncoder

encoder = BEATsEncoder("models/beats/BEATs_iter3_plus_AS2M.pt")
embedding = encoder.encode(waveform, sample_rate=16000, filename="00000000.wav")
print(embedding.vector.shape)  # (768,)
print(encoder.embedding_dim)   # 768
```

---

## src/fusion

### FusedFeatureVector

**File**: `src/fusion/fused_vector.py`

**Purpose**: Frozen dataclass holding the 921-dimensional fusion vector (DSP ⊕ BEATs) for a single recording, along with the component vectors and provenance metadata.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `machine_type` | `str` | Machine type (e.g. `"pump"`) |
| `machine_id` | `str` | Machine ID (e.g. `"id_00"`) |
| `label` | `str` | Recording label (`"normal"` or `"abnormal"`) |
| `filename` | `str` | Source audio filename |
| `sample_rate` | `int` | Sample rate of the source waveform |
| `dsp_feature_names` | `list[str]` | Ordered names of the 153 DSP features |
| `dsp_feature_vector` | `ndarray` shape `(D,)` | DSP feature vector (D = 153) |
| `beats_embedding` | `ndarray` shape `(768,)` | BEATs embedding vector |
| `fused_feature_vector` | `ndarray` shape `(D+768,)` | Concatenated fusion vector (921-dim) |
| `created_at` | `str` | ISO 8601 timestamp of creation |

**Public Methods**

---

#### `to_dict`

Serializes the fused vector to a JSON-compatible dictionary.

**Arguments**: None

**Returns**: `dict` — all fields with NumPy arrays converted to Python lists.

---

#### `from_dict`

Reconstructs a `FusedFeatureVector` from a dictionary.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` | Dictionary previously produced by `to_dict()` |

**Returns**: `FusedFeatureVector`

---

**Example**

```python
from src.fusion.fused_vector import FusedFeatureVector

fused = FusedFeatureVector(
    machine_type="pump",
    machine_id="id_00",
    label="normal",
    filename="00000000.wav",
    sample_rate=16000,
    dsp_feature_names=names,
    dsp_feature_vector=dsp_vec,
    beats_embedding=beats_vec,
    fused_feature_vector=fused_vec,
    created_at="2024-01-01T00:00:00",
)

d = fused.to_dict()
restored = FusedFeatureVector.from_dict(d)
```

---

### FusionBuilder

**File**: `src/fusion/fusion.py`

**Purpose**: Stateless builder that concatenates a DSP feature vector and a BEATs embedding into a `FusedFeatureVector`. Validates inputs before concatenation.

**Constructor**

No parameters.

**Public Methods**

---

#### `build`

Concatenates DSP and BEATs vectors into a `FusedFeatureVector`.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `dsp_vector` | `ndarray` | DSP feature vector (153-dim) |
| `dsp_feature_names` | `list[str]` | Ordered feature names matching `dsp_vector` |
| `beats_embedding` | `BEATsEmbedding` | BEATs embedding object |
| `machine_type` | `str` | Machine type label |
| `machine_id` | `str` | Machine ID label |
| `label` | `str` | Recording label (`"normal"` or `"abnormal"`) |

**Returns**: `FusedFeatureVector` with `fused_feature_vector` shape `(921,)`.

**Raises**

- `ValueError` — either vector is empty, contains NaN or Inf, or `dsp_feature_names` length does not match `dsp_vector` length.

---

**Example**

```python
from src.fusion.fusion import FusionBuilder

builder = FusionBuilder()
fused = builder.build(
    dsp_vector=dsp_vec,
    dsp_feature_names=names,
    beats_embedding=beats_emb,
    machine_type="pump",
    machine_id="id_00",
    label="normal",
)
print(fused.fused_feature_vector.shape)  # (921,)
```

---

### FusionCache

**File**: `src/fusion/cache.py`

**Purpose**: Disk-backed cache for `FusedFeatureVector` objects. On a cache miss, computes the full preprocessing → DSP → BEATs → fusion pipeline and saves the result. On a cache hit, loads the pre-computed NPZ file directly, avoiding redundant computation.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `cache_root` | `str \| Path` | Root directory for cached NPZ files |
| `pipeline` | `PreprocessingPipeline` | Preprocessing pipeline instance |
| `extractor` | `FeatureExtractor` | DSP feature extractor instance |
| `vec_builder` | `FeatureVectorBuilder` | DSP vector builder instance |
| `encoder` | `BEATsEncoder` | Frozen BEATs encoder instance |
| `fusion` | `FusionBuilder` | Fusion builder instance |

**Public Methods**

---

#### `exists`

Checks whether a cached NPZ file exists for the given recording.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `rec` | `AudioMetadata` | Recording metadata |

**Returns**: `bool`

---

#### `save`

Saves a `FusedFeatureVector` to disk as an NPZ file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `fused` | `FusedFeatureVector` | Fused vector to persist |
| `rec` | `AudioMetadata` | Recording metadata used to derive the cache key |

**Returns**: `None`

---

#### `load`

Loads a `FusedFeatureVector` from a cached NPZ file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `rec` | `AudioMetadata` | Recording metadata used to derive the cache key |

**Returns**: `FusedFeatureVector`

**Raises**

- `FileNotFoundError` — no cached file exists for this recording.

---

#### `load_or_create`

Returns the cached `FusedFeatureVector` if it exists; otherwise computes, saves, and returns it.

**Arguments**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rec` | `AudioMetadata` | — | Recording metadata |
| `verbose` | `bool` | `False` | If `True`, prints cache hit/miss status |

**Returns**: `FusedFeatureVector`

---

**Example**

```python
from src.fusion.cache import FusionCache

cache = FusionCache(
    cache_root="data/fusion_cache",
    pipeline=preprocessing_pipeline,
    extractor=feature_extractor,
    vec_builder=vector_builder,
    encoder=beats_encoder,
    fusion=fusion_builder,
)

fused = cache.load_or_create(rec, verbose=True)
print(fused.fused_feature_vector.shape)  # (921,)
```

---

### FusedVectorSerializer

**File**: `src/fusion/serializer.py`

**Purpose**: Handles serialization and deserialization of `FusedFeatureVector` objects to and from JSON and NPZ formats.

**Constructor**

No parameters.

**Public Methods**

---

#### `save_json`

Saves a `FusedFeatureVector` to a JSON file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `fused` | `FusedFeatureVector` | Fused vector to serialize |
| `path` | `str \| Path` | Destination file path |

**Returns**: `None`

---

#### `load_json`

Loads a `FusedFeatureVector` from a JSON file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Source file path |

**Returns**: `FusedFeatureVector`

**Raises**

- `FileNotFoundError` — file does not exist at the given path.

---

#### `save_npz`

Saves a `FusedFeatureVector` to a compressed NPZ file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `fused` | `FusedFeatureVector` | Fused vector to serialize |
| `path` | `str \| Path` | Destination file path |

**Returns**: `None`

---

#### `load_npz`

Loads a `FusedFeatureVector` from a compressed NPZ file.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Source file path |

**Returns**: `FusedFeatureVector`

**Raises**

- `FileNotFoundError` — file does not exist at the given path.

---

**Example**

```python
from src.fusion.serializer import FusedVectorSerializer

serializer = FusedVectorSerializer()
serializer.save_npz(fused, "outputs/fused_pump_id00.npz")
restored = serializer.load_npz("outputs/fused_pump_id00.npz")
```

---

## src/contrastive_learning

### ContrastivePair

**File**: `src/contrastive_learning/dataset.py`

**Purpose**: Frozen dataclass representing a single contrastive pair used during training. Positive pairs share the same machine identity; negative pairs come from different machines.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `anchor` | `FusedFeatureVector` | Anchor recording |
| `paired` | `FusedFeatureVector` | Paired recording (positive or negative) |
| `label` | `int` | `1` for positive pair (same machine), `0` for negative pair |

No public methods beyond dataclass field access.

---

**Example**

```python
from src.contrastive_learning.dataset import ContrastivePair

pair = ContrastivePair(anchor=fused_a, paired=fused_b, label=1)
print(pair.label)  # 1
```

---

### ContrastiveDataset

**File**: `src/contrastive_learning/dataset.py`

**Purpose**: Builds and exposes all contrastive pairs from the MIMII dataset. Uses `FusionCache` internally to retrieve or compute fusion vectors. Supports filtering by machine type and machine ID.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_root` | `str \| Path` | — | Root directory of the MIMII dataset |
| `checkpoint_path` | `str \| Path \| None` | `None` | BEATs checkpoint path; defaults to project default |
| `cache_root` | `str \| Path \| None` | `None` | Fusion cache root; defaults to project default |
| `seed` | `int` | `42` | Random seed for pair sampling |
| `machine_type` | `str \| None` | `None` | Restrict to one machine type |
| `machine_id` | `str \| None` | `None` | Restrict to one machine ID |
| `max_recordings` | `int \| None` | `None` | Maximum normal recordings per machine ID |

**Public Methods**

---

#### `machine_types`

**Arguments**: None

**Returns**: `list[str]` — all machine types present in the dataset.

---

#### `machine_ids`

**Arguments**: None

**Returns**: `list[str]` — all machine IDs present in the dataset.

---

#### `normal_recording_count`

**Arguments**: None

**Returns**: `int` — total number of normal recordings loaded.

---

#### `positive_pairs` *(property)*

**Returns**: `list[ContrastivePair]` — all pairs where `label == 1` (same machine).

---

#### `negative_pairs` *(property)*

**Returns**: `list[ContrastivePair]` — all pairs where `label == 0` (different machines).

---

#### `all_pairs` *(property)*

**Returns**: `list[ContrastivePair]` — combined list of positive and negative pairs.

---

#### `__len__`

**Returns**: `int` — total number of pairs.

---

#### `__iter__`

**Returns**: Iterator over `ContrastivePair` objects.

---

**Example**

```python
from src.contrastive_learning.dataset import ContrastiveDataset

ds = ContrastiveDataset(
    dataset_root="data/raw/MIMII",
    machine_type="pump",
    max_recordings=200,
)

print(len(ds))
print(len(ds.positive_pairs), len(ds.negative_pairs))
for pair in ds:
    print(pair.label)
    break
```

---

### NTXentLoss

**File**: `src/contrastive_learning/loss.py`

**Purpose**: Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss, also known as InfoNCE. Treats each sample in a batch as an anchor; its augmented counterpart is the positive, all others are negatives.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | `0.1` | Temperature scaling factor τ |

**Public Methods**

---

#### `forward`

Computes NT-Xent loss over a batch of embedding pairs.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `embeddings_a` | `Tensor` shape `(N, 256)` | Anchor embeddings (L2-normalized) |
| `embeddings_b` | `Tensor` shape `(N, 256)` | Paired embeddings (L2-normalized) |

**Returns**: `Tensor` — scalar loss value.

**Raises**

- `ValueError` — shapes of `embeddings_a` and `embeddings_b` do not match, embedding dimension is not 256, batch size is less than 2, or inputs contain NaN or Inf.

---

**Example**

```python
from src.contrastive_learning.loss import NTXentLoss

criterion = NTXentLoss(temperature=0.1)
loss = criterion(embeddings_a, embeddings_b)
loss.backward()
```

---

### ProjectionHead

**File**: `src/contrastive_learning/model.py`

**Purpose**: Trainable projection network that maps a 921-dimensional fusion vector to a 256-dimensional L2-normalized learned fingerprint. Only this module is trained; all upstream components remain frozen.

Architecture: `Linear(921 → 512) → ReLU → Linear(512 → 256) → L2-norm`

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_dim` | `int` | `921` | Input dimension (fusion vector size) |
| `output_dim` | `int` | `256` | Output dimension (fingerprint size) |

**Public Methods**

---

#### `forward`

Projects a batch of fusion vectors to L2-normalized fingerprints.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `x` | `Tensor` shape `(N, 921)` | Batch of fusion vectors |

**Returns**: `Tensor` shape `(N, 256)` — L2-normalized fingerprints.

---

#### `save_weights`

Saves model state dict to disk.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Destination file path |

**Returns**: `None`

---

#### `load_weights`

Loads model state dict from disk.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Source file path |

**Returns**: `None`

**Raises**

- `FileNotFoundError` — file does not exist at the given path.

---

**Example**

```python
from src.contrastive_learning.model import ProjectionHead

head = ProjectionHead(input_dim=921, output_dim=256)
fingerprint = head(fusion_tensor)   # shape (N, 256), L2-normalized
head.save_weights("models/contrastive/best_projection_head.pt")
head.load_weights("models/contrastive/best_projection_head.pt")
```

---

### EpochResult

**File**: `src/contrastive_learning/trainer.py`

**Purpose**: Dataclass holding the training and validation loss recorded at the end of a single training epoch.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `epoch` | `int` | Epoch index (1-based) |
| `training_loss` | `float` | Mean NT-Xent loss over the training split |
| `validation_loss` | `float` | Mean NT-Xent loss over the validation split |

No public methods beyond dataclass field access.

---

**Example**

```python
result = trainer.history()["epochs"][0]
print(result.epoch, result.training_loss, result.validation_loss)
```

---

### ContrastiveTrainer

**File**: `src/contrastive_learning/trainer.py`

**Purpose**: Trains the `ProjectionHead` using NT-Xent loss on a `ContrastiveDataset`. Saves the best checkpoint (lowest validation loss) automatically.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `head` | `ProjectionHead` | — | Projection head to train |
| `criterion` | `NTXentLoss` | — | Loss function |
| `learning_rate` | `float` | `1e-3` | Adam optimizer learning rate |
| `batch_size` | `int` | `32` | Training batch size |
| `epochs` | `int` | `10` | Number of training epochs |
| `checkpoint_dir` | `str \| Path` | `"models/contrastive"` | Directory to save the best checkpoint |
| `val_split` | `float` | `0.1` | Fraction of pairs reserved for validation |
| `seed` | `int` | `42` | Random seed for train/val split |

**Public Methods**

---

#### `fit`

Runs the full training loop over the provided dataset.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `ContrastiveDataset` | Dataset of contrastive pairs |

**Returns**: `None`

---

#### `history`

Returns the recorded training history.

**Arguments**: None

**Returns**: `dict` with key `"epochs"` mapping to `list[EpochResult]`.

---

**Example**

```python
from src.contrastive_learning.trainer import ContrastiveTrainer
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead

head = ProjectionHead()
criterion = NTXentLoss(temperature=0.1)
trainer = ContrastiveTrainer(head=head, criterion=criterion, epochs=10)
trainer.fit(dataset)

for epoch_result in trainer.history()["epochs"]:
    print(epoch_result.epoch, epoch_result.training_loss, epoch_result.validation_loss)
```

---

### ContrastiveInference

**File**: `src/contrastive_learning/inference.py`

**Purpose**: Runs a trained `ProjectionHead` at inference time to produce a 256-dimensional L2-normalized learned fingerprint from a single `FusedFeatureVector`.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `projection_head` | `ProjectionHead \| None` | `None` | Pre-instantiated projection head; if `None`, a new one is created |
| `checkpoint_path` | `str \| Path \| None` | `None` | Path to a saved checkpoint; weights are loaded if provided |

**Public Methods**

---

#### `generate_fingerprint`

Produces a 256-dimensional learned fingerprint from a fusion vector.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `fused_vector` | `FusedFeatureVector` | Fusion vector for a single recording |

**Returns**: `ndarray` shape `(256,)` — L2-normalized learned fingerprint.

**Raises**

- `ValueError` — `fused_feature_vector` dimension is not 921, output dimension is not 256, or L2 norm of the output deviates from 1.0.

---

**Example**

```python
from src.contrastive_learning.inference import ContrastiveInference

inference = ContrastiveInference(
    checkpoint_path="models/contrastive/best_projection_head.pt"
)
fingerprint = inference.generate_fingerprint(fused_vector)
print(fingerprint.shape)  # (256,)
```

---

### ContrastiveSerializer

**File**: `src/contrastive_learning/serializer.py`

**Purpose**: Provides static methods for saving and loading `ProjectionHead` checkpoints. No instance is required.

**Constructor**

No parameters. All methods are static.

**Public Methods**

---

#### `save_checkpoint` *(static)*

Saves a model checkpoint to disk.

**Arguments**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Destination file path |
| `model_state_dict` | `dict` | — | `model.state_dict()` output |
| `epoch` | `int` | — | Epoch at which the checkpoint was saved |
| `validation_loss` | `float` | — | Validation loss at this checkpoint |
| `optimizer_state_dict` | `dict \| None` | `None` | Optional optimizer state |
| `config` | `dict \| None` | `None` | Optional training configuration metadata |

**Returns**: `None`

---

#### `load_checkpoint` *(static)*

Loads a checkpoint from disk.

**Arguments**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Source file path |

**Returns**: `dict` with keys `model_state_dict`, `epoch`, `validation_loss`, and optionally `optimizer_state_dict` and `config`.

**Raises**

- `FileNotFoundError` — file does not exist at the given path.

---

**Example**

```python
from src.contrastive_learning.serializer import ContrastiveSerializer

ContrastiveSerializer.save_checkpoint(
    path="models/contrastive/best_projection_head.pt",
    model_state_dict=head.state_dict(),
    epoch=10,
    validation_loss=0.42,
)

checkpoint = ContrastiveSerializer.load_checkpoint(
    "models/contrastive/best_projection_head.pt"
)
head.load_state_dict(checkpoint["model_state_dict"])
```
