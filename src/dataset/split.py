"""Reproducible recording-level dataset splitter for MIMII-style datasets.

Splits AudioMetadata objects per (machine_type, machine_id) group into
train, profile, and test partitions with no overlap between normal splits.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from .metadata import AudioMetadata


@dataclass
class DatasetSplit:
    """Holds the four partitions produced by DatasetSplitter.

    Attributes:
        train_normal:   Normal recordings reserved for contrastive training.
        profile_normal: Normal recordings used to build the healthy profile.
        test_normal:    Normal recordings held out for evaluation.
        test_abnormal:  All abnormal recordings, used only for evaluation.
    """

    train_normal: List[AudioMetadata] = field(default_factory=list)
    profile_normal: List[AudioMetadata] = field(default_factory=list)
    test_normal: List[AudioMetadata] = field(default_factory=list)
    test_abnormal: List[AudioMetadata] = field(default_factory=list)


class DatasetSplitter:
    """Splits a flat list of AudioMetadata into non-overlapping partitions.

    Normal recordings are shuffled reproducibly and divided per
    (machine_type, machine_id) group according to *train_ratio* and
    *profile_ratio*. The remainder becomes test_normal. All abnormal
    recordings go directly into test_abnormal.

    Args:
        train_ratio:   Fraction of normal recordings for training (default 0.6).
        profile_ratio: Fraction of normal recordings for the healthy profile
                       (default 0.2). The test fraction is the remainder.
        seed:          Random seed for reproducible shuffling (default 42).
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        profile_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1).")
        if not (0 < profile_ratio < 1):
            raise ValueError("profile_ratio must be in (0, 1).")
        if train_ratio + profile_ratio >= 1.0:
            raise ValueError("train_ratio + profile_ratio must be less than 1.")

        self.train_ratio = train_ratio
        self.profile_ratio = profile_ratio
        self.seed = seed

    def split(self, recordings: List[AudioMetadata]) -> DatasetSplit:
        """Partition *recordings* into the four dataset splits.

        Normal recordings are split per (machine_type, machine_id) group so
        that every group contributes proportionally to each partition. No
        normal recording appears in more than one split.

        Args:
            recordings: Flat list of AudioMetadata objects from any number of
                        machine types and IDs.

        Returns:
            A DatasetSplit containing the four non-overlapping partitions.
        """
        normal_by_group: dict[tuple[str, str], List[AudioMetadata]] = defaultdict(list)
        result = DatasetSplit()

        for rec in recordings:
            if rec.label == "normal":
                normal_by_group[(rec.machine_type, rec.machine_id)].append(rec)
            elif rec.label == "abnormal":
                result.test_abnormal.append(rec)
            else:
                raise ValueError(
                    f"Unexpected label '{rec.label}' for recording '{rec.absolute_path}'. "
                    "Expected 'normal' or 'abnormal'."
                )

        rng = random.Random(self.seed)

        for group_recs in normal_by_group.values():
            shuffled = group_recs.copy()
            rng.shuffle(shuffled)

            n = len(shuffled)
            train_end = int(n * self.train_ratio)
            profile_end = train_end + int(n * self.profile_ratio)

            result.train_normal.extend(shuffled[:train_end])
            result.profile_normal.extend(shuffled[train_end:profile_end])
            result.test_normal.extend(shuffled[profile_end:])

        return result
