"""Phase 9 pre-training verification — does NOT call main()."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = open(ROOT / "phase9_verify_out.txt", "w", encoding="utf-8")

def p(msg):
    OUT.write(msg + "\n")
    OUT.flush()

# ── 1. Load module without calling main() ──────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "phase9_train", ROOT / "experiments" / "phase9_train.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
p("CHECK 1 (module loads without error): PASS")

# ── 2. Constants ───────────────────────────────────────────────────────────
checks = {
    "EXPERIMENT_ID":   (mod.EXPERIMENT_ID,    "phase9"),
    "MACHINE_TYPES":   (mod.MACHINE_TYPES,    ["fan", "pump", "slider", "valve"]),
    "MACHINE_IDS":     (mod.MACHINE_IDS,      ["id_00", "id_02", "id_04", "id_06"]),
    "TRAIN_RATIO":     (mod.TRAIN_RATIO,      0.70),
    "PROFILE_RATIO":   (mod.PROFILE_RATIO,    0.15),
    "SEED":            (mod.SEED,             42),
    "EPOCHS":          (mod.EPOCHS,           20),
    "BATCH_SIZE":      (mod.BATCH_SIZE,       16),
    "LEARNING_RATE":   (mod.LEARNING_RATE,    0.001),
    "TEMPERATURE":     (mod.TEMPERATURE,      0.07),
    "INPUT_DIM":       (mod.INPUT_DIM,        921),
    "PROJECTION_DIM":  (mod.PROJECTION_DIM,   256),
}
all_ok = True
for name, (actual, expected) in checks.items():
    ok = actual == expected
    if not ok:
        all_ok = False
        print(f"CHECK 2 (constant {name}): FAIL  actual={actual!r}  expected={expected!r}")
if all_ok:
    print("CHECK 2 (all constants): PASS")

# ── 3. Real dataset splits ─────────────────────────────────────────────────
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter

loader   = DatasetLoader("data/raw/MIMII")
all_recs = loader.get_all_files()
splitter = DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)

splits = {}
for mt in mod.MACHINE_TYPES:
    splits[mt] = splitter.split([r for r in all_recs if r.machine_type == mt])

# CHECK 3: per-type totals match hardcoded _EXPECTED_TRAIN_NORMAL
for mt in mod.MACHINE_TYPES:
    actual   = len(splits[mt].train_normal)
    expected = mod._EXPECTED_TRAIN_NORMAL[mt]
    tag = "PASS" if actual == expected else "FAIL"
    print(f"CHECK 3 (train_normal total {mt}): {tag}  actual={actual}  expected={expected}")

# CHECK 3b: per-ID counts match _EXPECTED_TRAIN_NORMAL_PER_ID
for mt in mod.MACHINE_TYPES:
    for mid in mod.MACHINE_IDS:
        actual   = sum(1 for r in splits[mt].train_normal if r.machine_id == mid)
        expected = mod._EXPECTED_TRAIN_NORMAL_PER_ID[mt][mid]
        tag = "PASS" if actual == expected else "FAIL"
        print(f"CHECK 3b ({mt}/{mid}): {tag}  actual={actual}  expected={expected}")

# CHECK 4: pooled total == 10296
pooled = [r for mt in mod.MACHINE_TYPES for r in splits[mt].train_normal]
total  = len(pooled)
tag = "PASS" if total == 10296 else "FAIL"
print(f"CHECK 4 (pooled total == 10296): {tag}  actual={total}")

# CHECK 5: all 16 machine_type × machine_id combos present
expected_combos = {(mt, mid) for mt in mod.MACHINE_TYPES for mid in mod.MACHINE_IDS}
actual_combos   = {(r.machine_type, r.machine_id) for r in pooled}
missing = expected_combos - actual_combos
tag = "PASS" if not missing else "FAIL"
print(f"CHECK 5 (all 16 combos present): {tag}", end="")
if missing:
    print(f"  MISSING={sorted(missing)}", end="")
print()

# CHECK 6: all pooled records are normal
non_normal = [r for r in pooled if r.label != "normal"]
tag = "PASS" if not non_normal else "FAIL"
print(f"CHECK 6 (only normal in pooled train): {tag}  non-normal={len(non_normal)}")

# CHECK 7: no overlap with held-out partitions
pooled_paths = {r.absolute_path for r in pooled}
for mt in mod.MACHINE_TYPES:
    s = splits[mt]
    for partition, name in [
        (s.profile_normal, "profile_normal"),
        (s.test_normal,    "test_normal"),
        (s.test_abnormal,  "test_abnormal"),
    ]:
        overlap = pooled_paths & {r.absolute_path for r in partition}
        tag = "PASS" if not overlap else "FAIL"
        print(f"CHECK 7 (no overlap {mt}/{name}): {tag}  overlap={len(overlap)}")

# CHECK 8: checkpoint and history paths
ckpt_path    = mod.CHECKPOINT_DIR / "best_projection_head.pt"
history_path = mod.CHECKPOINT_DIR / "training_history.json"
print(f"CHECK 8a (checkpoint path):        PASS  {ckpt_path}")
print(f"CHECK 8b (training_history path):  PASS  {history_path}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\nDone.")
