# System Architecture

Complete inference pipeline from raw audio to machine health report.

```mermaid
flowchart TD
    A([".wav File"]):::io

    subgraph PRE["Preprocessing"]
        B["AudioLoader\nMono · 16 kHz"]
        C["AudioResampler"]
        D["AudioNormalizer"]
        E["SpectrogramGenerator\nLog-Mel"]
        B --> C --> D --> E
    end

    subgraph DSP["DSP Feature Extraction"]
        F["MFCCExtractor\n20 coeffs · mean + std"]
        G["SpectralExtractor\nCentroid · Rolloff"]
        H["TemporalExtractor\nRMS Energy"]
        I["HarmonicExtractor\nHarmonic Salience"]
        J["FeatureVectorBuilder\n153-dim vector"]
        F & G & H & I --> J
    end

    subgraph BEATS["BEATs Encoder  (frozen)"]
        K["BEATsEncoder\nBEATs_iter3_plus_AS2M.pt"]
        L["Mean Pool Frames\n768-dim embedding"]
        K --> L
    end

    subgraph FUSION["Fusion"]
        M["FusionBuilder\nDSP ⊕ BEATs"]
        N[("FusionCache\nNPZ on disk")]
        M <-->|"load / save"| N
    end

    subgraph CL["Contrastive Learning"]
        O["ContrastiveInference\nload checkpoint"]
        P["ProjectionHead\n921 → 512 → 256 · L2-norm"]
        O --> P
    end

    subgraph PROFILE["Learned Profile  (healthy)"]
        Q["LearnedFingerprintProfile\nmean · std · embeddings"]
    end

    subgraph DRIFT["Learned Drift Analysis"]
        R["LearnedDriftMetrics"]
        S["LearnedDriftResult\nRaw + Normalised metrics\nEuclidean · Manhattan · Cosine"]
        R --> S
    end

    subgraph HEALTH["Learned Health Index"]
        T["LearnedHealthCalculator\nz-score normalisation"]
        U["LearnedHealthResult\nscore · percentage · state"]
        T --> U
    end

    V(["PipelineResult\nDimensions · Drift · Health"]):::io

    A --> PRE
    PRE --> DSP
    PRE --> BEATS
    J -->|"153-dim"| M
    L -->|"768-dim"| M
    M -->|"921-dim fused vector"| O
    P -->|"256-dim embedding"| R
    Q -->|"mean · std"| R
    Q -->|"healthy norm"| T
    S -->|"norm_euclidean\nnorm_manhattan\nnorm_cosine"| T
    U --> V
    S --> V

    classDef io fill:#1e3a5f,stroke:#4a90d9,color:#e8f4fd,font-weight:bold
    classDef default fill:#1a1a2e,stroke:#4a4a6a,color:#c8c8e8
```

## Dimension Summary

| Stage | Output |
|---|---|
| DSP Feature Extraction | 153-dim float32 vector |
| BEATs Encoder | 768-dim float32 embedding |
| Fusion Vector | 921-dim float32 (DSP ⊕ BEATs) |
| ProjectionHead | 256-dim L2-normalised embedding |

## Health State Bands

| Score | State | Meaning |
|---|---|---|
| 90–100 | EXCELLENT | Within normal healthy variation |
| 75–89 | GOOD | Slightly outside typical healthy variation |
| 50–74 | WARNING | Statistically significant deviation |
| 0–49 | CRITICAL | Extreme outlier relative to healthy distribution |
