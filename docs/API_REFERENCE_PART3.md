# API Reference — Part 3

Modules: `learned_profile`, `learned_drift`, `learned_health_index`, `pipeline`

---

## learned_profile

### LearnedFingerprintProfile

Dataclass holding the healthy learned fingerprint profile for one machine. Stores all 256-dim embeddings from normal recordings plus their per-dimension mean and std.

**Constructor**

```python
LearnedFingerprintProfile(
    machine_type: str,
    machine_id: str,
    embedding_dimension: int,
    embeddings: np.ndarray,      # shape (N, 256), float32
    mean_vector: np.ndarray,     # shape (256,), float32
    std_vector: np.ndarray,      # shape (256,), float32
    created_at: str = <utc_now>
)
```

**Public Methods**

- `to_dict()` — no args → `dict` (JSON-compatible; arrays as lists)
- `from_dict(data: dict)` — classmethod → `LearnedFingerprintProfile`; raises `KeyError` if required fields missing

**Example**

```python
profile = LearnedFingerprintProfile(
    machine_type="pump", machine_id="id_00",
    embedding_dimension=256,
    embeddings=matrix, mean_vector=mean, std_vector=std,
)
d = profile.to_dict()
restored = LearnedFingerprintProfile.from_dict(d)
```

---

### LearnedProfileBuilder

Builds a `LearnedFingerprintProfile` by running all normal recordings through the full pipeline: Preprocessing → DSP → BEATs → Fusion → ProjectionHead.

**Constructor**

```python
LearnedProfileBuilder(
    checkpoint_path: str | Path,
    beats_checkpoint: str | Path | None = None,  # defaults to models/beats/BEATs_iter3_plus_AS2M.pt
    cache_root: str | Path | None = None,         # defaults to data/fusion_cache
)
```

**Public Methods**

- `build(loader, machine_type, machine_id, max_recordings=None)`
  - `loader: DatasetLoader`
  - `machine_type: str`
  - `machine_id: str`
  - `max_recordings: int | None`
  - Returns `LearnedFingerprintProfile`
  - Raises `ValueError` if no normal recordings found or all fail

**Example**

```python
from src.dataset.loader import DatasetLoader
from src.learned_profile.builder import LearnedProfileBuilder

builder = LearnedProfileBuilder(checkpoint_path="models/contrastive/best_projection_head.pt")
loader = DatasetLoader("data/raw/MIMII")
profile = builder.build(loader, machine_type="pump", machine_id="id_00", max_recordings=200)
```

---

### LearnedProfileSerializer

Saves and loads `LearnedFingerprintProfile` to JSON or NPZ.

**Constructor**

```python
LearnedProfileSerializer()
```

**Public Methods**

- `save_json(profile: LearnedFingerprintProfile, path: str | Path)` → `None`
- `load_json(path: str | Path)` → `LearnedFingerprintProfile`; raises `FileNotFoundError`, `KeyError`
- `save_npz(profile: LearnedFingerprintProfile, path: str | Path)` → `None`
- `load_npz(path: str | Path)` → `LearnedFingerprintProfile`; raises `FileNotFoundError`

**Example**

```python
from src.learned_profile.serializer import LearnedProfileSerializer

s = LearnedProfileSerializer()
s.save_json(profile, "outputs/learned_profiles/pump_id_00.json")
profile = s.load_json("outputs/learned_profiles/pump_id_00.json")
```

---

## learned_drift

### LearnedDriftResult

Dataclass holding raw and normalized drift metrics for one recording compared against a healthy profile.

**Constructor**

```python
LearnedDriftResult(
    machine_type: str,
    machine_id: str,
    filename: str,
    euclidean_distance: float,
    manhattan_distance: float,
    cosine_similarity: float,
    norm_euclidean_distance: float,
    norm_manhattan_distance: float,
    norm_cosine_similarity: float,
    normalized_vector: np.ndarray,  # shape (256,), float32
    created_at: str = <utc_now>
)
```

**Public Methods**

- `to_dict()` → `dict`
- `from_dict(data: dict)` — classmethod → `LearnedDriftResult`; raises `KeyError`

**Example**

```python
d = result.to_dict()
restored = LearnedDriftResult.from_dict(d)
```

---

### LearnedDriftMetrics

Computes raw and z-score normalized drift metrics between a 256-dim embedding and a profile.

**Constructor**

```python
LearnedDriftMetrics()
```

**Public Methods**

- `compute(current_embedding: np.ndarray, profile: LearnedFingerprintProfile)`
  - Returns `tuple[float, float, float, float, float, float, np.ndarray]`:
    `(euclidean, manhattan, cosine, norm_euclidean, norm_manhattan, norm_cosine, normalized_vector)`
  - Raises `ValueError` if embedding dimension mismatches profile

**Example**

```python
from src.learned_drift.metrics import LearnedDriftMetrics

metrics = LearnedDriftMetrics()
euclid, manhat, cosine, n_euclid, n_manhat, n_cosine, z_vec = metrics.compute(embedding, profile)
```

---

### LearnedDriftAnalyzer

Runs the full pipeline for one recording and returns a `LearnedDriftResult`. Reuses `FusionCache` and `ContrastiveInference` internally.

**Constructor**

```python
LearnedDriftAnalyzer(
    checkpoint_path: str | Path,
    beats_checkpoint: str | Path | None = None,
    cache_root: str | Path | None = None,
)
```

**Public Methods**

- `analyze(record: AudioMetadata, profile: LearnedFingerprintProfile)` → `LearnedDriftResult`
  - Raises `ValueError` if `machine_type` or `machine_id` mismatches between record and profile

**Example**

```python
from src.learned_drift.analyzer import LearnedDriftAnalyzer

analyzer = LearnedDriftAnalyzer(checkpoint_path="models/contrastive/best_projection_head.pt")
result = analyzer.analyze(record, profile)
print(result.norm_euclidean_distance)
```

---

### LearnedDriftSerializer

Saves and loads `LearnedDriftResult` to JSON or NPZ.

**Constructor**

```python
LearnedDriftSerializer()
```

**Public Methods**

- `save_json(result: LearnedDriftResult, path: str | Path)` → `None`
- `load_json(path: str | Path)` → `LearnedDriftResult`; raises `FileNotFoundError`, `KeyError`
- `save_npz(result: LearnedDriftResult, path: str | Path)` → `None`
- `load_npz(path: str | Path)` → `LearnedDriftResult`; raises `FileNotFoundError`

**Example**

```python
from src.learned_drift.serializer import LearnedDriftSerializer

s = LearnedDriftSerializer()
s.save_json(result, "outputs/drift/pump_id_00_result.json")
result = s.load_json("outputs/drift/pump_id_00_result.json")
```

---

## learned_health_index

### LearnedHealthResult

Dataclass holding the health score, percentage, state, and normalized drift inputs for one recording.

**Constructor**

```python
LearnedHealthResult(
    machine_type: str,
    machine_id: str,
    filename: str,
    health_score: float,          # [0, 100]
    health_percentage: str,       # e.g. "82.5%"
    health_state: str,            # EXCELLENT | GOOD | WARNING | CRITICAL
    normalized_euclidean: float,
    normalized_manhattan: float,
    normalized_cosine: float,
    created_at: str = <utc_now>
)
```

**Public Methods**

- `to_dict()` → `dict`
- `from_dict(data: dict)` — classmethod → `LearnedHealthResult`; raises `KeyError`

**Example**

```python
d = result.to_dict()
restored = LearnedHealthResult.from_dict(d)
```

---

### LearnedHealthCalculator

Converts normalized drift metrics into a bounded health score using a machine-specific scale derived from the profile.

Score formula: `100 × (1 − norm_euclidean / (2 × profile_healthy_norm))`, clamped to `[0, 100]`.

**Constructor**

```python
LearnedHealthCalculator(
    thresholds: dict[str, float] | None = None
    # defaults: EXCELLENT≥90, GOOD≥75, WARNING≥50, else CRITICAL
)
```

**Public Methods**

- `calculate(normalized_euclidean, normalized_manhattan, normalized_cosine, profile_healthy_norm)`
  - All args `float`
  - Returns `tuple[float, str, str]`: `(health_score, health_percentage, health_state)`

**Example**

```python
from src.learned_health_index.calculator import LearnedHealthCalculator

calc = LearnedHealthCalculator()
score, pct, state = calc.calculate(
    normalized_euclidean=14.2,
    normalized_manhattan=180.0,
    normalized_cosine=0.95,
    profile_healthy_norm=13.75,
)
# score=48.7, pct="48.7%", state="CRITICAL"
```

---

### LearnedHealthAnalyzer

Computes the health index for one recording by delegating to `LearnedDriftAnalyzer` then `LearnedHealthCalculator`.

**Constructor**

```python
LearnedHealthAnalyzer(
    checkpoint_path: str | Path,
    beats_checkpoint: str | Path | None = None,
    cache_root: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
)
```

**Public Methods**

- `analyze(record: AudioMetadata, profile: LearnedFingerprintProfile)` → `LearnedHealthResult`

**Example**

```python
from src.learned_health_index.analyzer import LearnedHealthAnalyzer

analyzer = LearnedHealthAnalyzer(checkpoint_path="models/contrastive/best_projection_head.pt")
result = analyzer.analyze(record, profile)
print(result.health_score, result.health_state)
```

---

### LearnedHealthSerializer

Saves and loads `LearnedHealthResult` to JSON or NPZ.

**Constructor**

```python
LearnedHealthSerializer()
```

**Public Methods**

- `save_json(result: LearnedHealthResult, path: str | Path)` → `None`
- `load_json(path: str | Path)` → `LearnedHealthResult`; raises `FileNotFoundError`, `KeyError`
- `save_npz(result: LearnedHealthResult, path: str | Path)` → `None`
- `load_npz(path: str | Path)` → `LearnedHealthResult`; raises `FileNotFoundError`

**Example**

```python
from src.learned_health_index.serializer import LearnedHealthSerializer

s = LearnedHealthSerializer()
s.save_json(result, "outputs/health/pump_id_00.json")
result = s.load_json("outputs/health/pump_id_00.json")
```

---

## pipeline

### MachineHealthReport

Dataclass aggregating dimension metadata, raw drift, normalized drift, and health index for one recording.

**Constructor**

```python
MachineHealthReport(
    machine_type: str,
    machine_id: str,
    filename: str,
    dsp_dimension: int,
    beats_dimension: int,
    fusion_dimension: int,
    learned_dimension: int,
    euclidean_distance: float,
    manhattan_distance: float,
    cosine_similarity: float,
    normalized_euclidean_distance: float,
    normalized_manhattan_distance: float,
    normalized_cosine_similarity: float,
    health_score: float,
    health_percentage: str,
    health_state: str,
    created_at: str = <utc_now>
)
```

**Public Methods**

- `to_dict()` → `dict`
- `from_dict(data: dict)` — classmethod → `MachineHealthReport`; raises `KeyError`

**Example**

```python
d = report.to_dict()
restored = MachineHealthReport.from_dict(d)
```

---

### MachineHealthPipeline

Runs the full end-to-end health analysis pipeline for one recording. Accepts pre-constructed analyzers so BEATs, ProjectionHead, and FusionCache are loaded once and reused.

**Constructor**

```python
MachineHealthPipeline(
    profile: LearnedFingerprintProfile,
    drift_analyzer: LearnedDriftAnalyzer,
    health_analyzer: LearnedHealthAnalyzer,
)
```

**Public Methods**

- `analyze(record: AudioMetadata)` → `MachineHealthReport`

**Example**

```python
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.pipeline.pipeline import MachineHealthPipeline

ckpt = "models/contrastive/best_projection_head.pt"
drift_analyzer = LearnedDriftAnalyzer(ckpt)
health_analyzer = LearnedHealthAnalyzer(ckpt)

pipeline = MachineHealthPipeline(profile, drift_analyzer, health_analyzer)
report = pipeline.analyze(record)
print(report.health_score, report.health_state)
```
