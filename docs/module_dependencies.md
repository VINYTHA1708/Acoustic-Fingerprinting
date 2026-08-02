# Module Dependencies

Directed dependency graph of the major source modules.
An arrow `A --> B` means module A imports from module B.

```mermaid
flowchart LR
    %% ── Leaf modules (no internal deps) ──────────────────────────────
    DS(["dataset\nDatasetLoader\nAudioMetadata"]):::leaf
    PRE(["preprocessing\nPreprocessingPipeline"]):::leaf
    FE(["feature_extraction\nFeatureExtractor\nFeatureVectorBuilder"]):::leaf
    BT(["beats\nBEATsEncoder"]):::leaf

    %% ── Fusion layer ─────────────────────────────────────────────────
    FB(["fusion\nFusionBuilder\nFusedFeatureVector"]):::mid
    FC(["fusion_cache\nFusionCache"]):::mid

    %% ── Contrastive learning ─────────────────────────────────────────
    CDS(["contrastive_learning\nContrastiveDataset"]):::mid
    PH(["contrastive_learning\nProjectionHead"]):::mid
    NTX(["contrastive_learning\nNTXentLoss"]):::mid
    CT(["contrastive_learning\nContrastiveTrainer"]):::mid
    CI(["contrastive_learning\nContrastiveInference"]):::mid
    CS(["contrastive_learning\nContrastiveSerializer"]):::mid

    %% ── Profile / Drift / Health ─────────────────────────────────────
    LP(["learned_profile\nLearnedProfileBuilder\nLearnedFingerprintProfile"]):::high
    LD(["learned_drift\nLearnedDriftAnalyzer\nLearnedDriftMetrics"]):::high
    LH(["learned_health_index\nLearnedHealthAnalyzer\nLearnedHealthCalculator"]):::high

    %% ── Pipeline & Benchmark ─────────────────────────────────────────
    PL(["pipeline\nInferencePipeline\nMachineHealthPipeline"]):::top
    BM(["benchmark\nPipelineBenchmark"]):::top

    %% ── FusionCache depends on all compute modules ───────────────────
    FC --> PRE
    FC --> FE
    FC --> BT
    FC --> FB
    FC --> DS

    %% ── ContrastiveDataset depends on FusionCache + Dataset ──────────
    CDS --> FC
    CDS --> DS

    %% ── Trainer depends on Dataset, Head, Loss, Serializer ───────────
    CT --> CDS
    CT --> PH
    CT --> NTX
    CT --> CS

    %% ── Inference depends on Head + Serializer ───────────────────────
    CI --> PH
    CI --> CS

    %% ── LearnedProfileBuilder depends on FusionCache + Inference ─────
    LP --> FC
    LP --> CI
    LP --> DS

    %% ── LearnedDriftAnalyzer depends on FusionCache + Inference ──────
    LD --> FC
    LD --> CI
    LD --> LP

    %% ── LearnedHealthAnalyzer depends on LearnedDriftAnalyzer ────────
    LH --> LD
    LH --> LP

    %% ── Pipeline depends on Drift + Health ───────────────────────────
    PL --> LD
    PL --> LH
    PL --> LP

    %% ── Benchmark depends on all compute modules directly ────────────
    BM --> PRE
    BM --> FE
    BM --> BT
    BM --> FB
    BM --> FC
    BM --> CI
    BM --> LD
    BM --> LH
    BM --> LP

    classDef leaf  fill:#1a3a1a,stroke:#4a8a4a,color:#c8e8c8
    classDef mid   fill:#1a2a3a,stroke:#4a6a8a,color:#c8d8e8
    classDef high  fill:#2a1a3a,stroke:#6a4a8a,color:#d8c8e8
    classDef top   fill:#3a1a1a,stroke:#8a4a4a,color:#e8c8c8
```

## Dependency Layers

| Layer | Modules | Depends on |
|---|---|---|
| **Leaf** | `preprocessing`, `feature_extraction`, `beats`, `dataset` | External libraries only |
| **Fusion** | `fusion`, `fusion_cache` | Leaf layer |
| **Contrastive** | `ProjectionHead`, `NTXentLoss`, `ContrastiveInference`, `ContrastiveTrainer`, `ContrastiveDataset`, `ContrastiveSerializer` | Fusion layer |
| **Profile / Drift / Health** | `learned_profile`, `learned_drift`, `learned_health_index` | Fusion + Contrastive layers |
| **Orchestration** | `pipeline`, `benchmark` | All layers above |
