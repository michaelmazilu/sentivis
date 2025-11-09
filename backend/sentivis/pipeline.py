from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .emotion_model import EmotionClassifier

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    score: float


class InferencePipeline:
    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        emotion_weights_path: Optional[str] = None,
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._detector = self._create_detector()
        self._emotion_classifier = EmotionClassifier(weights_path=emotion_weights_path)

    def reset(self) -> None:
        """
        Re-create heavy components. Helpful if the client reconnects and we want to
        clean up the Mediapipe graphs.
        """
        try:
            self._detector.close()
        except AttributeError:
            pass

        self._detector = self._create_detector()

    def process_frame(self, frame_payload: str) -> Dict[str, Any]:
        if not frame_payload:
            raise ValueError("Frame payload is empty")

        frame = self._decode_frame(frame_payload)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)

        faces: List[Dict[str, Any]] = []
        if results.detections:
            for detection in results.detections:
                bbox_and_crop = self._extract_bounding_box(detection, frame)
                if bbox_and_crop is None:
                    continue

                detection_box, face_crop = bbox_and_crop
                face_payload: Dict[str, Any] = {
                    "box": {
                        "x": detection_box.x,
                        "y": detection_box.y,
                        "width": detection_box.width,
                        "height": detection_box.height,
                        "score": detection_box.score,
                    },
                }

                emotion_payload = self._run_emotion_classifier(face_crop)
                if emotion_payload is not None:
                    face_payload["emotion"] = emotion_payload

                faces.append(face_payload)

        response: Dict[str, Any] = {
            "timestamp": time.time(),
            "faces": faces,
            "models": {
                "emotion": (
                    "ready"
                    if self._emotion_classifier.is_ready
                    else "unavailable"
                ),
            },
        }

        return response

    def _create_detector(self) -> mp.solutions.face_detection.FaceDetection:
        return mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self._min_detection_confidence,
        )

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

    def _extract_bounding_box(
        self,
        detection: Any,
        frame: np.ndarray,
    ) -> Optional[Tuple[BoundingBox, np.ndarray]]:
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

        detection_score = float(detection.score[0]) if detection.score else 0.0

        return (
            BoundingBox(
                x=xmin,
                y=ymin,
                width=width,
                height=height,
                score=detection_score,
            ),
            crop,
        )

    def _run_emotion_classifier(self, face_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self._emotion_classifier.is_ready:
            return None

        try:
            prediction = self._emotion_classifier.predict(face_bgr)
        except (ValueError, RuntimeError) as exc:
            LOGGER.debug("Skipping emotion classification: %s", exc)
            return None

        scores_payload = [
            {"label": label, "confidence": score}
            for label, score in zip(self._emotion_classifier.labels, prediction.scores)
        ]

        return {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "scores": scores_payload,
        }
