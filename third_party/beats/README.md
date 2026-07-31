# third_party/beats/

This directory holds the official Microsoft BEATs source files.

## What belongs here

Clone or copy the following files from the Microsoft UniLM repository:

    https://github.com/microsoft/unilm/tree/master/beats

Required files:

    BEATs.py
    backbone.py
    modules.py
    tokenizers.py

## Why a third_party/ directory

BEATs is not available as a pip-installable package.
The source must be vendored locally so the project can import it without
depending on a live network connection or an unstable external path.

## How it is used

src/beats/encoder.py adds this directory to sys.path at import time:

    sys.path.insert(0, ".../third_party/beats")
    from BEATs import BEATs, BEATsConfig

This import is currently commented out (TODO marker in encoder.py) and will
be activated in Version 2 once the source files are placed here.

## Do not commit large binary files

Do not place the model checkpoint (.pt file) here.
Checkpoints belong in models/beats/.
