from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from .emotion_model import EmotionClassifier, EmotionPrediction

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    score: float


@dataclass
class FaceAnalysis:
    box: BoundingBox
    emotion: EmotionPrediction


class InferencePipeline:
    def __init__(self, min_detection_confidence: float = 0.6) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self._classifier = EmotionClassifier()

    def reset(self) -> None:
        """
        Re-create heavy components. Helpful if the client reconnects and we want to
        clean up the Mediapipe graphs.
        """
        try:
            self._detector.close()
        except AttributeError:
            pass

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )

    def process_frame(self, frame_payload: str) -> Dict[str, Any]:
        if not frame_payload:
            raise ValueError("Frame payload is empty")

        frame = self._decode_frame(frame_payload)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)

        analysis: Optional[FaceAnalysis] = None
        if results.detections:
            detection = results.detections[0]
            analysis = self._analyze_detection(detection, frame)

        response: Dict[str, Any] = {
            "timestamp": time.time(),
            "faces": [],
        }

        if analysis is not None:
            response["faces"].append(
                {
                    "box": {
                        "x": analysis.box.x,
                        "y": analysis.box.y,
                        "width": analysis.box.width,
                        "height": analysis.box.height,
                        "score": analysis.box.score,
                    },
                    "emotion": {
                        "label": analysis.emotion.label,
                        "confidence": analysis.emotion.confidence,
                        "scores": analysis.emotion.scores,
                    },
                }
            )

        return response

    @staticmethod
    def _decode_frame(frame_payload: str) -> np.ndarray:
        if "," in frame_payload and frame_payload.startswith("data:"):
            _, frame_payload = frame_payload.split(",", 1)

        try:
            frame_bytes = base64.b64decode(frame_payload)
        except (ValueError, TypeError) as exc:
            raise ValueError("Frame payload is not valid base64 data") from exc

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode frame into an image")

        return frame

    def _analyze_detection(self, detection: Any, frame: np.ndarray) -> Optional[FaceAnalysis]:
        image_height, image_width, _ = frame.shape

        try:
            bbox = detection.location_data.relative_bounding_box
        except AttributeError as exc:
            LOGGER.warning("Detection missing bounding box: %s", exc)
            return None

        xmin = max(bbox.xmin, 0.0)
        ymin = max(bbox.ymin, 0.0)
        width = min(bbox.width, 1.0 - xmin)
        height = min(bbox.height, 1.0 - ymin)

        xmax = min(xmin + width, 1.0)
        ymax = min(ymin + height, 1.0)

        start_x = int(xmin * image_width)
        start_y = int(ymin * image_height)
        end_x = int(xmax * image_width)
        end_y = int(ymax * image_height)

        if end_x <= start_x or end_y <= start_y:
            LOGGER.debug("Ignoring degenerate bounding box: %s", bbox)
            return None

        crop = frame[start_y:end_y, start_x:end_x]
        if crop.size == 0:
            LOGGER.debug("Empty crop for bounding box: %s", bbox)
            return None

        try:
            prediction = self._classifier.predict(crop)
        except ValueError as exc:
            LOGGER.warning("Failed to run emotion classifier: %s", exc)
            return None

        detection_score = float(detection.score[0]) if detection.score else 0.0

        return FaceAnalysis(
            box=BoundingBox(
                x=xmin,
                y=ymin,
                width=width,
                height=height,
                score=detection_score,
            ),
            emotion=prediction,
        )
