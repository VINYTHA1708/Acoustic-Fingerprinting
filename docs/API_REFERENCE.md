# API Reference — Part 1

Covers three modules: `dataset`, `preprocessing`, and `feature_extraction`.

These modules form the first two stages of the acoustic fingerprinting pipeline
(SDD v4 §10 V1): raw file discovery → audio cleaning → DSP feature extraction.

---

## Table of Contents

- [dataset](#dataset)
  - [AudioMetadata](#audiometadata)
  - [DatasetLoader](#datasetloader)
- [preprocessing](#preprocessing)
  - [AudioLoader](#audioloader)
  - [AudioNormalizer](#audionormalizer)
  - [AudioResampler](#audioresampler)
  - [SpectrogramGenerator](#spectrogramgenerator)
  - [PreprocessingPipeline](#preprocessingpipeline)
- [feature\_extraction](#feature_extraction)
  - [MFCCExtractor](#mfccextractor)
  - [SpectralExtractor](#spectralextractor)
  - [TemporalExtractor](#temporalextractor)
  - [HarmonicExtractor](#harmonicextractor)
  - [FeatureExtractor](#featureextractor)
  - [FeatureVectorBuilder](#featurevectorbuilder)

---

## dataset

Scans a MIMII-style directory tree and exposes per-recording metadata through
a filterable index. No audio is loaded by this module — it works entirely with
file paths and directory structure.

Expected directory layout relative to the dataset root:

```
<machine_type>/<machine_id>/<label>/<filename>.wav
```

Example: `fan/id_00/normal/00000000.wav`

---

### AudioMetadata

**Module:** `src/dataset/metadata.py`

#### Purpose

Immutable dataclass holding the metadata extracted from a single audio file
path. It is the atomic unit returned by `DatasetLoader` and consumed by every
downstream module that needs to locate or identify a recording.

#### Constructor

`AudioMetadata` is a frozen dataclass. Instances are created internally by
`extract_metadata()` and returned through `DatasetLoader`. Direct construction
is possible but not the typical usage.

| Field | Type | Description |
|---|---|---|
| `machine_type` | `str` | Machine type parsed from the path (e.g. `"fan"`, `"pump"`). |
| `machine_id` | `str` | Machine identifier parsed from the path (e.g. `"id_00"`). |
| `label` | `str` | Recording condition — `"normal"` or `"abnormal"`. |
| `filename` | `str` | Bare filename including extension (e.g. `"00000000.wav"`). |
| `relative_path` | `Path` | Path relative to the dataset root. |
| `absolute_path` | `Path` | Fully resolved absolute path to the audio file. |

#### Public Methods

`AudioMetadata` is a frozen dataclass and exposes no additional public methods
beyond standard dataclass equality and hashing.

#### Example

```python
from src.dataset.metadata import AudioMetadata
from pathlib import Path

meta = AudioMetadata(
    machine_type="pump",
    machine_id="id_00",
    label="normal",
    filename="00000000.wav",
    relative_path=Path("pump/id_00/normal/00000000.wav"),
    absolute_path=Path("/data/raw/MIMII/pump/id_00/normal/00000000.wav"),
)

print(meta.machine_type)   # pump
print(meta.label)          # normal
```

---

### DatasetLoader

**Module:** `src/dataset/loader.py`

#### Purpose

Scans a dataset root directory on construction, parses every `.wav` file path
into an `AudioMetadata` record, and provides filtered views over the resulting
index. It is the single entry point for all dataset access in the pipeline.

#### Constructor

```python
DatasetLoader(root: str | Path)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root` | `str \| Path` | required | Path to the dataset root directory (e.g. `"data/raw/MIMII"`). |

**Raises**

- `FileNotFoundError` — if `root` does not exist.
- `NotADirectoryError` — if `root` is not a directory.

#### Public Methods

---

##### `get_all_files`

- **Purpose:** Return every loaded audio metadata record.
- **Arguments:** none
- **Returns:** `list[AudioMetadata]` — all valid records found under `root`.

---

##### `get_machine_types`

- **Purpose:** Return the sorted unique machine types present in the dataset.
- **Arguments:** none
- **Returns:** `list[str]` — e.g. `['fan', 'pump', 'slider', 'valve']`.

---

##### `get_machine_ids`

- **Purpose:** Return sorted unique machine IDs, optionally restricted to one machine type.
- **Arguments:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `machine_type` | `str \| None` | `None` | If provided, restrict to IDs belonging to this type. |

- **Returns:** `list[str]` — e.g. `['id_00', 'id_02', 'id_04', 'id_06']`.

---

##### `filter_by_machine`

- **Purpose:** Return all records for a given machine type.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `machine_type` | `str` | Machine type to filter on (e.g. `"fan"`). |

- **Returns:** `list[AudioMetadata]`

---

##### `filter_by_machine_id`

- **Purpose:** Return all records for a given machine ID across all machine types.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `machine_id` | `str` | Machine ID to filter on (e.g. `"id_00"`). |

- **Returns:** `list[AudioMetadata]`

---

##### `filter_by_label`

- **Purpose:** Return all records matching a label.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `label` | `str` | `"normal"` or `"abnormal"`. |

- **Returns:** `list[AudioMetadata]`

---

##### `summary`

- **Purpose:** Print a human-readable dataset summary to stdout. Includes root path, machine types, machine IDs, and normal/abnormal recording counts.
- **Arguments:** none
- **Returns:** `None`

---

#### Example

```python
from src.dataset.loader import DatasetLoader

loader = DatasetLoader("data/raw/MIMII")
loader.summary()

# All normal pump recordings
normal_pumps = [
    r for r in loader.filter_by_machine("pump")
    if r.label == "normal"
]

# All machine IDs for fan
fan_ids = loader.get_machine_ids("fan")
print(fan_ids)  # ['id_00', 'id_02', 'id_04', 'id_06']
```

---

## preprocessing

Converts a raw `.wav` file into a clean, fixed-format waveform and a log-Mel
spectrogram. Each class in this module has a single responsibility and can be
used independently or composed through `PreprocessingPipeline`.

Pipeline order (SDD v4 §10 V1 step 2):

```
Load (mono) → Resample → Normalize → Log-Mel Spectrogram
```

---

### AudioLoader

**Module:** `src/preprocessing/audio_loader.py`

#### Purpose

Loads a `.wav` file from disk using librosa and optionally converts it to mono.
It is the first step in the preprocessing chain.

#### Constructor

```python
AudioLoader(mono: bool = True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mono` | `bool` | `True` | If `True`, convert stereo to mono after loading. |

#### Public Methods

---

##### `load`

- **Purpose:** Load an audio file and return the waveform and its native sample rate.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wav` file. |

- **Returns:** `tuple[np.ndarray, int]` — `(waveform, sample_rate)` where `waveform` is a 1-D float32 array (mono) and `sample_rate` is the file's native rate in Hz.
- **Raises:**
  - `FileNotFoundError` — if the file does not exist.
  - `ValueError` — if the file is empty, the format is unsupported, or loading fails.

---

#### Example

```python
from src.preprocessing.audio_loader import AudioLoader

loader = AudioLoader(mono=True)
waveform, sr = loader.load("data/raw/MIMII/pump/id_00/normal/00000000.wav")
print(waveform.shape, sr)  # (160000,) 16000
```

---

### AudioNormalizer

**Module:** `src/preprocessing/normalizer.py`

#### Purpose

Peak-normalizes a waveform to the range `[-1, 1]`. Silent recordings (peak
amplitude below `1e-9`) are returned unchanged to avoid amplifying pure noise.

#### Constructor

```python
AudioNormalizer()
```

No parameters.

#### Public Methods

---

##### `normalize`

- **Purpose:** Peak-normalize a waveform to `[-1, 1]`.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |

- **Returns:** `np.ndarray` — normalized waveform with the same shape and dtype as the input. Returned as-is if the recording is silent.

---

#### Example

```python
from src.preprocessing.normalizer import AudioNormalizer
import numpy as np

normalizer = AudioNormalizer()
waveform = np.array([0.0, 0.5, -0.25], dtype=np.float32)
normalized = normalizer.normalize(waveform)
print(normalized.max())  # 1.0
```

---

### AudioResampler

**Module:** `src/preprocessing/resampler.py`

#### Purpose

Resamples a waveform to a configurable target sample rate using librosa. If the
input is already at the target rate, the waveform is returned unchanged with no
processing overhead.

#### Constructor

```python
AudioResampler(target_sr: int = 16_000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_sr` | `int` | `16_000` | Desired output sample rate in Hz. |

#### Public Methods

---

##### `resample`

- **Purpose:** Resample a waveform to the target sample rate.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |
| `sample_rate` | `int` | Native sample rate of `waveform` in Hz. |

- **Returns:** `tuple[np.ndarray, int]` — `(resampled_waveform, target_sr)`.

---

#### Example

```python
from src.preprocessing.resampler import AudioResampler
import numpy as np

resampler = AudioResampler(target_sr=16_000)
waveform = np.zeros(44100, dtype=np.float32)
resampled, sr = resampler.resample(waveform, sample_rate=44100)
print(sr)  # 16000
```

---

### SpectrogramGenerator

**Module:** `src/preprocessing/spectrogram.py`

#### Purpose

Computes a log-Mel spectrogram from a waveform. The output is used as the
shared input representation for both DSP feature extraction and the BEATs
encoder (SDD v4 §5).

#### Constructor

```python
SpectrogramGenerator(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sample_rate` | `int` | `16_000` | Expected sample rate of the input waveform in Hz. |
| `n_fft` | `int` | `1024` | FFT window size. |
| `hop_length` | `int` | `512` | Number of samples between frames. |
| `win_length` | `int` | `1024` | Window length in samples. |
| `n_mels` | `int` | `128` | Number of Mel filter banks. |
| `fmin` | `float` | `20.0` | Lowest frequency for the Mel filter bank in Hz. |
| `fmax` | `float \| None` | `None` | Highest frequency in Hz. Defaults to `sample_rate // 2` (Nyquist). |

#### Public Methods

---

##### `generate`

- **Purpose:** Compute the log-Mel spectrogram of a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform at the configured sample rate. |

- **Returns:** `np.ndarray` — 2-D float32 array of shape `(n_mels, time_frames)`.

---

#### Example

```python
from src.preprocessing.spectrogram import SpectrogramGenerator
import numpy as np

gen = SpectrogramGenerator(sample_rate=16_000, n_mels=128)
waveform = np.random.randn(16000).astype(np.float32)
spec = gen.generate(waveform)
print(spec.shape)  # (128, 32)
```

---

### PreprocessingPipeline

**Module:** `src/preprocessing/pipeline.py`

#### Purpose

Orchestrates the full preprocessing chain — load, resample, normalize,
spectrogram — in a single `run()` call. This is the primary entry point used
by all higher-level modules (fusion cache, drift analyzer, health analyzer).

Returns a `PreprocessingResult` TypedDict with keys `waveform`, `sample_rate`,
and `spectrogram`.

#### Constructor

```python
PreprocessingPipeline(
    target_sr: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_sr` | `int` | `16_000` | Target sample rate in Hz. |
| `n_fft` | `int` | `1024` | FFT window size passed to `SpectrogramGenerator`. |
| `hop_length` | `int` | `512` | Hop length passed to `SpectrogramGenerator`. |
| `win_length` | `int` | `1024` | Window length passed to `SpectrogramGenerator`. |
| `n_mels` | `int` | `128` | Number of Mel bands passed to `SpectrogramGenerator`. |
| `fmin` | `float` | `20.0` | Minimum frequency for the Mel filter bank in Hz. |
| `fmax` | `float \| None` | `None` | Maximum frequency in Hz. Defaults to `target_sr // 2`. |

#### Public Methods

---

##### `run`

- **Purpose:** Execute the full preprocessing pipeline on a single audio file.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wav` file to process. |

- **Returns:** `PreprocessingResult` — a TypedDict with:
  - `waveform` (`np.ndarray`) — 1-D float32 normalized waveform at `target_sr`.
  - `sample_rate` (`int`) — the target sample rate in Hz.
  - `spectrogram` (`np.ndarray`) — 2-D float32 log-Mel spectrogram of shape `(n_mels, time_frames)`.
- **Raises:**
  - `FileNotFoundError` — propagated from `AudioLoader` if the file does not exist.
  - `ValueError` — propagated from `AudioLoader` if the file is empty or unsupported.

---

#### Example

```python
from src.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(target_sr=16_000)
result = pipeline.run("data/raw/MIMII/pump/id_00/normal/00000000.wav")

print(result["waveform"].shape)      # (160000,)
print(result["sample_rate"])         # 16000
print(result["spectrogram"].shape)   # (128, 313)
```

---

## feature_extraction

Computes the 153-dimensional DSP feature vector from a preprocessed waveform.
All features are deterministic and human-interpretable — they form the
explainability backbone of the system (SDD v4 §4.2).

The full feature set is organized into four sub-extractors:

| Sub-extractor | Features produced |
|---|---|
| `MFCCExtractor` | MFCC, delta-MFCC, delta-delta-MFCC (mean + std per coefficient) |
| `SpectralExtractor` | Centroid, bandwidth, rolloff, flatness, ZCR, spectral contrast per band |
| `TemporalExtractor` | RMS energy (mean, std, max), short-time energy (mean, std), dynamic range |
| `HarmonicExtractor` | Harmonic energy, percussive energy, harmonic ratio |

`FeatureExtractor` orchestrates all four. `FeatureVectorBuilder` converts the
resulting named dict into a 1-D float32 numpy array.

---

### MFCCExtractor

**Module:** `src/feature_extraction/mfcc.py`

#### Purpose

Extracts MFCC, delta-MFCC, and delta-delta-MFCC statistics from a waveform.
With the default of 20 coefficients, produces `20 × 2 × 3 = 120` named float
values (mean and std for each coefficient across all three derivative orders).

#### Constructor

```python
MFCCExtractor(n_mfcc: int = 20, sample_rate: int = 16_000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_mfcc` | `int` | `20` | Number of MFCC coefficients. |
| `sample_rate` | `int` | `16_000` | Sample rate of the input waveform in Hz. |

#### Public Methods

---

##### `extract`

- **Purpose:** Extract MFCC-family statistics from a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |

- **Returns:** `dict[str, float]` — flat dict of `n_mfcc * 2 * 3` named float values. Keys follow the pattern `mfcc_<i>_mean`, `mfcc_<i>_std`, `mfcc_delta_<i>_mean`, etc.

---

#### Example

```python
from src.feature_extraction.mfcc import MFCCExtractor
import numpy as np

extractor = MFCCExtractor(n_mfcc=20, sample_rate=16_000)
waveform = np.random.randn(16000).astype(np.float32)
features = extractor.extract(waveform)
print(len(features))          # 120
print(list(features.keys())[:3])  # ['mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean']
```

---

### SpectralExtractor

**Module:** `src/feature_extraction/spectral.py`

#### Purpose

Extracts spectral shape features from a waveform: centroid, bandwidth, rolloff,
flatness, zero-crossing rate, and per-band spectral contrast. Each feature is
summarized as mean and std across time frames.

#### Constructor

```python
SpectralExtractor(
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_bands: int = 6,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sample_rate` | `int` | `16_000` | Sample rate of the input waveform in Hz. |
| `n_fft` | `int` | `1024` | FFT window size. |
| `hop_length` | `int` | `512` | Hop length in samples. |
| `n_bands` | `int` | `6` | Number of bands for spectral contrast. Produces `n_bands + 1` contrast bands. |

#### Public Methods

---

##### `extract`

- **Purpose:** Extract spectral statistics from a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |

- **Returns:** `dict[str, float]` — flat dict with mean and std for: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `zcr`, and `spectral_contrast_band<i>` for each of the `n_bands + 1` contrast bands.

---

#### Example

```python
from src.feature_extraction.spectral import SpectralExtractor
import numpy as np

extractor = SpectralExtractor(sample_rate=16_000)
waveform = np.random.randn(16000).astype(np.float32)
features = extractor.extract(waveform)
print(features["spectral_centroid_mean"])
print(features["spectral_rolloff_std"])
```

---

### TemporalExtractor

**Module:** `src/feature_extraction/temporal.py`

#### Purpose

Extracts temporal energy features from a waveform: RMS energy statistics,
short-time energy statistics, and dynamic range in dB. These features capture
the energy envelope of the recording over time.

#### Constructor

```python
TemporalExtractor(frame_length: int = 1024, hop_length: int = 512)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `frame_length` | `int` | `1024` | Frame length in samples. |
| `hop_length` | `int` | `512` | Hop length in samples. |

#### Public Methods

---

##### `extract`

- **Purpose:** Extract temporal energy statistics from a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |

- **Returns:** `dict[str, float]` — flat dict with six keys:
  - `rms_mean`, `rms_std`, `rms_max` — RMS energy statistics.
  - `ste_mean`, `ste_std` — short-time energy statistics.
  - `dynamic_range_db` — difference between max and min RMS in dB.

---

#### Example

```python
from src.feature_extraction.temporal import TemporalExtractor
import numpy as np

extractor = TemporalExtractor()
waveform = np.random.randn(16000).astype(np.float32)
features = extractor.extract(waveform)
print(features["rms_mean"])
print(features["dynamic_range_db"])
```

---

### HarmonicExtractor

**Module:** `src/feature_extraction/harmonic.py`

#### Purpose

Separates the harmonic and percussive components of a waveform using
librosa's HPSS (Harmonic-Percussive Source Separation) and computes
energy-based statistics for each component. The harmonic ratio is the
primary explainability feature for harmonic drift (SDD v4 §4.2).

#### Constructor

```python
HarmonicExtractor()
```

No parameters.

#### Public Methods

---

##### `extract`

- **Purpose:** Extract harmonic and percussive energy features from a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |

- **Returns:** `dict[str, float]` — flat dict with three keys:
  - `harmonic_energy` — mean squared amplitude of the harmonic component.
  - `percussive_energy` — mean squared amplitude of the percussive component.
  - `harmonic_ratio` — `harmonic_energy / (harmonic_energy + percussive_energy)`. Returns `0.0` for silent recordings.

---

#### Example

```python
from src.feature_extraction.harmonic import HarmonicExtractor
import numpy as np

extractor = HarmonicExtractor()
waveform = np.random.randn(16000).astype(np.float32)
features = extractor.extract(waveform)
print(features["harmonic_ratio"])   # e.g. 0.63
```

---

### FeatureExtractor

**Module:** `src/feature_extraction/extractor.py`

#### Purpose

Orchestrates all four DSP sub-extractors (`MFCCExtractor`, `SpectralExtractor`,
`TemporalExtractor`, `HarmonicExtractor`) and merges their outputs into a single
flat dictionary. Key order is deterministic: MFCC → spectral → temporal →
harmonic. This is the primary DSP entry point used by `FusionCache` and the
training pipeline.

#### Constructor

```python
FeatureExtractor(
    sample_rate: int = 16_000,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 512,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sample_rate` | `int` | `16_000` | Sample rate of the input waveform in Hz. |
| `n_mfcc` | `int` | `20` | Number of MFCC coefficients passed to `MFCCExtractor`. |
| `n_fft` | `int` | `1024` | FFT window size shared by spectral and temporal extractors. |
| `hop_length` | `int` | `512` | Hop length in samples shared by spectral and temporal extractors. |

#### Public Methods

---

##### `extract`

- **Purpose:** Extract the full DSP feature set from a waveform.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `waveform` | `np.ndarray` | 1-D float32 audio waveform. |
| `sample_rate` | `int \| None` | Ignored — present for API symmetry. Sample rate is fixed at construction time. |

- **Returns:** `dict[str, float]` — flat dict mapping feature name to float value. With default settings, produces 153 named features in deterministic order.

---

#### Example

```python
from src.feature_extraction.extractor import FeatureExtractor
import numpy as np

extractor = FeatureExtractor(sample_rate=16_000)
waveform = np.random.randn(16000).astype(np.float32)
features = extractor.extract(waveform)
print(len(features))   # 153
print(list(features.keys())[:4])
# ['mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean']
```

---

### FeatureVectorBuilder

**Module:** `src/feature_extraction/feature_vector.py`

#### Purpose

Converts the named feature dictionary produced by `FeatureExtractor` into a
deterministic, ordered 1-D float32 numpy array. The builder is stateless —
key order is preserved from the dict's insertion order (Python 3.7+). The
resulting vector is the DSP block of the 921-dimensional Fusion Fingerprint
(SDD v4 §4.1).

#### Constructor

```python
FeatureVectorBuilder()
```

No parameters.

#### Public Methods

---

##### `build`

- **Purpose:** Convert a feature dict to a 1-D float32 numpy vector.
- **Arguments:**

| Parameter | Type | Description |
|---|---|---|
| `features` | `dict[str, float]` | Flat dict of `{feature_name: float_value}` as returned by `FeatureExtractor`. |

- **Returns:** `tuple[np.ndarray, list[str]]` — a tuple of:
  - `vector` — 1-D float32 numpy array of length `len(features)`.
  - `names` — list of feature names in the same order as `vector`.
- **Raises:**
  - `ValueError` — if `features` is empty.

---

#### Example

```python
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
import numpy as np

waveform = np.random.randn(16000).astype(np.float32)

extractor = FeatureExtractor(sample_rate=16_000)
builder = FeatureVectorBuilder()

features = extractor.extract(waveform)
vector, names = builder.build(features)

print(vector.shape)   # (153,)
print(vector.dtype)   # float32
print(names[0])       # mfcc_0_mean
```
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
