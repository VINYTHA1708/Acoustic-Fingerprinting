# Training Pipeline

Contrastive training flow from raw normal recordings to a saved ProjectionHead checkpoint.

```mermaid
flowchart TD
    A([Normal Recordings\nMIMII dataset]):::io

    subgraph DATASET["DatasetLoader"]
        B["DatasetLoader\nscans MIMII directory tree"]
        C["AudioMetadata\nmachine_type · machine_id · label · path"]
        B --> C
    end

    subgraph CACHE["FusionCache"]
        D{"NPZ cached?"}
        E["PreprocessingPipeline\nmono · 16 kHz · normalise"]
        F["FeatureExtractor\nMFCC · Spectral · Temporal · Harmonic"]
        G["BEATsEncoder\nfrozen · 768-dim"]
        H["FusionBuilder\n921-dim fused vector"]
        I[("NPZ on disk\ndata/fusion_cache/")]
        D -->|"miss"| E --> F & G --> H --> I
        D -->|"hit"| I
    end

    subgraph CONTRASTIVE["ContrastiveDataset"]
        J["Positive Pair Builder\nsame machine · different recordings"]
        K["Negative Pair Builder\ndifferent machine_id or machine_type"]
        L["ContrastivePair\nanchor · paired · label"]
        J & K --> L
    end

    subgraph TRAIN["ContrastiveTrainer"]
        M["Train / Val Split\n80 % train · 20 % val"]
        N["Mini-batch Loop\nbatch_size = 32"]
        O["ProjectionHead\n921 → 512 → 256 · L2-norm"]
        P["NTXentLoss\nNT-Xent · temperature = 0.1"]
        Q["Adam Optimizer\nlr = 1e-3"]
        R{"val_loss\nimproved?"}
        M --> N --> O --> P --> Q --> R
        R -->|"yes"| S
        R -->|"no"| N
    end

    subgraph SAVE["ContrastiveSerializer"]
        S["save_checkpoint()\nmodel_state_dict · epoch · val_loss"]
        T([models/contrastive/\nbest_projection_head.pt]):::io
        S --> T
    end

    A --> DATASET
    C --> CACHE
    I --> CONTRASTIVE
    L --> TRAIN

    classDef io fill:#1e3a5f,stroke:#4a90d9,color:#e8f4fd,font-weight:bold
    classDef default fill:#1a1a2e,stroke:#4a4a6a,color:#c8c8e8
```

## Training Configuration

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `2` | Number of full passes over training pairs |
| `batch_size` | `32` | Pairs per mini-batch |
| `learning_rate` | `1e-3` | Adam learning rate |
| `temperature` | `0.1` | NT-Xent softmax temperature |
| `val_split` | `0.2` | Fraction of pairs reserved for validation |

## Pair Construction Rules

| Pair Type | Rule |
|---|---|
| Positive | Same `machine_type` and `machine_id`, different recording |
| Negative | Different `machine_id` or different `machine_type` |
