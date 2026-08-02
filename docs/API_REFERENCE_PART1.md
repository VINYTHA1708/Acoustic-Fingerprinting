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
