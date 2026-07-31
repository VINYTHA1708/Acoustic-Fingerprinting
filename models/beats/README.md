# models/beats/

This directory holds the pretrained BEATs model checkpoint.

## What belongs here

Download the pretrained checkpoint from the Microsoft UniLM repository:

    https://github.com/microsoft/unilm/tree/master/beats

Recommended checkpoint:

    BEATs_iter3_plus_AS2M.pt

Place the downloaded file directly in this directory:

    models/beats/BEATs_iter3_plus_AS2M.pt

## How it is used

src/beats/encoder.py loads the checkpoint via:

    torch.load("models/beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu")

The checkpoint path is passed to BEATsEncoder at construction time.

## Do not commit checkpoints to version control

Model checkpoints are large binary files and must not be committed to git.
The .gitignore should exclude *.pt files under models/.

## Version 2 dependency

This checkpoint is required for Version 2 (BEATs Integration).
The Version 1 DSP-only pipeline does not depend on this directory.
