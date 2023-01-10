# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utiliy functions for the obfuscated datasets."""

import enum
from typing import Optional, Sequence, Tuple

import numpy as np


CLEAN = 'Clean'

NUM_FEW_SHOT_EXAMPLES = 10

TRAIN_OBFUSCATIONS = [
    CLEAN,
    'AdversarialPatches',
    'BackgroundBlurComposition',
    'ColorNoiseBlocks',
    'Halftoning',
    'HighContrastBorder',
    'IconOverlay',
    'ImageOverlay',
    'Interleave',
    'InvertLines',
    'LineShift',
    'PerspectiveTransform',
    'PhotoComposition',
    'RotateBlocks',
    'RotateImage',
    'StyleTransfer',
    'SwirlWarp',
    'TextOverlay',
    'Texturize',
    'WavyColorWarp',
]

HOLD_OUT_OBFUSCATIONS = [
    'ColorPatternOverlay',
    'LowContrastTriangles',
    'PerspectiveComposition',
    ]

EVAL_OBFUSCATIONS = TRAIN_OBFUSCATIONS + HOLD_OUT_OBFUSCATIONS


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: str) -> 'Split':
    return {
        'TRAIN': Split.TRAIN,
        'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
        'VALID': Split.VALID,
        'VALIDATION': Split.VALID,
        'TEST': Split.TEST
    }[name.upper()]


class Dataset(enum.Enum):
  COCO = 'obfuscated_coco'
  IMAGENET = 'obfuscated_imagenet'

_NUM_EXAMPLES_DICT = {
    Dataset.IMAGENET: {
        Split.TRAIN_AND_VALID: 1281167,
        Split.TRAIN: 1271167,
        Split.VALID: 10000,
        Split.TEST: 50000
    },
    Dataset.COCO: {
        Split.TRAIN_AND_VALID: 116925,
        Split.TRAIN: 111925,
        Split.VALID: 5000,
        Split.TEST: 4940
    },
}


def get_num_examples(dataset: Dataset, split: Split,
                     _: Optional[str] = None) -> int:
  return _NUM_EXAMPLES_DICT[dataset][split]


def get_obfuscations(split: Split) -> Sequence[str]:
  return EVAL_OBFUSCATIONS if split == Split.TEST else TRAIN_OBFUSCATIONS


# We use the ImageNet validation split as our test split as there are no labels
# for the ImageNet test split.
def get_subsets(split: Split, obfuscations: Optional[Sequence[str]] = None
                ) -> Sequence[str]:
  """Returns list of subsets."""
  if obfuscations is None:
    obfuscations = get_obfuscations(split)

  if split == Split.TEST:
    return [f'validation_{obfuscation}' for obfuscation in obfuscations]
  else:
    return [f'train_{obfuscation}' for obfuscation in obfuscations]


def shard(dataset: Dataset, split: Split, obfuscation: Optional[str] = None,
          shard_index: int = 0, num_shards: int = 1) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(get_num_examples(dataset, split, obfuscation))
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000] for
    # ImageNet and the same with 5000 instead of 10000 for Coco.
    offset = get_num_examples(dataset, Split.VALID, obfuscation)
    start += offset
    end += offset
  return start, end
