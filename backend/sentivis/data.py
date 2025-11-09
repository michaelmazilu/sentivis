from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

FER_IMAGE_SIZE = 48
FER_SPLITS: Sequence[str] = ("Training", "PublicTest", "PrivateTest")


class Fer2013Dataset(Dataset):
    """
    Minimal FER-2013 dataset wrapper around the kaggle CSV file.
    Provides grayscale 48x48 tensors in the [0, 1] range.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str = "Training",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        if split not in FER_SPLITS:
            raise ValueError(f"Unsupported FER split '{split}'. Expected one of {FER_SPLITS}")

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"FER-2013 csv not found at {self.csv_path}")

        dataframe = pd.read_csv(self.csv_path)
        expected_columns = {"emotion", "pixels", "Usage"}
        if not expected_columns.issubset(dataframe.columns):
            missing = expected_columns - set(dataframe.columns)
            raise ValueError(f"FER-2013 csv missing required columns: {missing}")

        subset = dataframe[dataframe["Usage"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No samples found for split '{split}' in {self.csv_path}")

        pixel_arrays = []
        for pixels in subset["pixels"].tolist():
            vector = np.fromstring(pixels, dtype=np.uint8, sep=" ")
            if vector.size != FER_IMAGE_SIZE * FER_IMAGE_SIZE:
                raise ValueError(
                    f"Unexpected pixel length {vector.size}. "
                    f"Expected {FER_IMAGE_SIZE * FER_IMAGE_SIZE}"
                )
            pixel_arrays.append(vector)

        self.images = np.stack(pixel_arrays).reshape(-1, FER_IMAGE_SIZE, FER_IMAGE_SIZE)
        self.labels = subset["emotion"].astype(np.int64).to_numpy()
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        label = int(self.labels[index])

        pil_image = Image.fromarray(image)
        if self.transform is not None:
            tensor = self.transform(pil_image)
        else:
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        return tensor, label
