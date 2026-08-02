# Contributing

Thank you for your interest in contributing to the Acoustic Fingerprinting project.

---

## Repository Setup

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/Acoustic-Fingerprinting.git
cd Acoustic-Fingerprinting
```

Add the upstream remote so you can pull in future changes:

```bash
git remote add upstream https://github.com/VINYTHA1708/Acoustic-Fingerprinting.git
```

---

## Virtual Environment

Create and activate a virtual environment before installing anything:

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## Installing Dependencies

Upgrade pip, then install all project requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

The BEATs encoder requires a pretrained checkpoint that is not bundled in the repository. Download `BEATs_iter3_plus_AS2M.pt` from the [official Microsoft BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) and place it at:

```
models/beats/BEATs_iter3_plus_AS2M.pt
```

Place the MIMII dataset under:

```
data/raw/MIMII/
```

---

## Running Examples

All example scripts are run from the project root and accept `--help` for a full argument listing:

```bash
python examples/dataset_example.py --root data/raw/MIMII
python examples/preprocessing_example.py --root data/raw/MIMII
python examples/fusion_example.py --root data/raw/MIMII
python examples/pipeline_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

See `README.md` for the full list of example commands and their arguments.

---

## Running Tests

Run the full test suite from the project root:

```bash
python -m pytest tests/
```

All 68 tests must pass before a pull request will be reviewed. The same suite runs automatically on every push and pull request via GitHub Actions.

To run a single test file:

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## Coding Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use 4-space indentation. Do not use tabs.
- Keep lines to a maximum of 100 characters.
- Write descriptive variable and function names; avoid single-letter names outside of loop indices.
- Add type annotations to all public function signatures.
- Keep functions focused — one responsibility per function.
- Do not leave commented-out code or debug print statements in committed code.
- Match the style of the surrounding module when editing existing files.

---

## Commit Message Style

Use the following format for all commit messages:

```
<type>: <short summary in present tense, 72 characters or fewer>
```

Allowed types:

| Type | When to use |
|---|---|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation changes only |
| `refactor` | Code restructuring with no behaviour change |
| `perf` | Performance improvement |
| `chore` | Dependency updates, CI changes, tooling |

Examples:

```
feat: add streaming inference support to InferencePipeline
fix: correct z-score normalisation in LearnedDriftMetrics
docs: add GPU benchmark results to RELEASE_NOTES
test: add edge case for empty healthy profile in test_profile.py
```

---

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes, following the coding and commit style guidelines above.

3. Ensure all tests pass locally before pushing:

   ```bash
   python -m pytest tests/
   ```

4. Push your branch and open a pull request against `main`:

   ```bash
   git push origin feat/your-feature-name
   ```

5. In the pull request description, include:
   - What the change does and why it is needed
   - Which modules or files are affected
   - How the change was tested

6. GitHub Actions will run the full test suite automatically. The pull request cannot be merged until all checks pass.

7. Address any review feedback with additional commits on the same branch. Do not force-push after a review has started.
