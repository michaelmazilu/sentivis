from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

EMOTION_LABELS: Sequence[str] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


class EmotionNet(nn.Module):
    """
    Compact convolutional network tailored for FER-2013's 48x48 grayscale input.
    """

    def __init__(self, num_classes: int = len(EMOTION_LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x24
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6x6
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    scores: List[float]


class EmotionClassifier:
    """
    Loads a trained EmotionNet checkpoint and exposes a convenience predict method.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)
        self.model = EmotionNet()
        self.model.to(self.device)

        self.labels: Sequence[str] = EMOTION_LABELS
        self._ready = False
        self._mean = 0.5
        self._std = 0.5

        if weights_path is not None:
            self.load_weights(weights_path)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load_weights(self, weights_path: str | Path) -> None:
        checkpoint_path = Path(weights_path)
        if not checkpoint_path.exists():
            LOGGER.warning("Emotion weights not found at %s", checkpoint_path)
            self._ready = False
            return

        LOGGER.info("Loading emotion model weights from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

        label_map = checkpoint.get("label_map")
        if label_map:
            self.labels = tuple(label_map)

        self.model.eval()
        self._ready = True

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        if not self._ready:
            raise RuntimeError("Emotion classifier is not ready. Train and load weights first.")
        if face_bgr.size == 0:
            raise ValueError("Cannot run emotion inference on an empty crop")

        tensor = self._preprocess(face_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits[0], dim=0)
            confidence, index = torch.max(probs, dim=0)

        return EmotionPrediction(
            label=self.labels[int(index)],
            confidence=float(confidence),
            scores=probs.tolist(),
        )

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        tensor = (tensor - self._mean) / self._std
        return tensor
