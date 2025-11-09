# Sentivis

Sentivis is a full-stack prototype that streams webcam frames to a Flask backend, detects faces with MediaPipe, and classifies their emotion with a FER‑2013 convolutional model that you can train yourself. The Next.js frontend renders detections, latency, and model health in real time.

## Repository layout

```
app/                     # Next.js app router UI
backend/                 # Flask-SocketIO backend + training scripts
backend/sentivis/        # Reusable inference + dataset utilities
backend/artifacts/       # Emotion model weights (gitignored, contains .gitkeep)
```

## Prerequisites

- Node.js 18+ (for the frontend)
- Python 3.10+ with `venv` (for the backend + training)
- FER‑2013 dataset (`fer2013.csv`) downloaded from Kaggle. Place it anywhere on disk; the training script takes the path as an argument.

## Frontend (Next.js)

```bash
npm install
npm run dev
```

Set `NEXT_PUBLIC_SENTIVIS_SOCKET_URL` if the backend is not running on `http://localhost:5001`.

## Backend (Flask + Socket.IO)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Key environment variables:

- `SENTIVIS_EMOTION_WEIGHTS` (optional) – absolute or relative path to the `.pt` weights file produced by training. Defaults to `backend/artifacts/emotion-net.pt`.

## Training the FER-2013 model

1. Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
2. Run the training script (use `--help` to see all options):

```bash
cd backend
source .venv/bin/activate  # if not already active
python train_emotion_model.py /path/to/fer2013.csv \
  --output artifacts/emotion-net.pt \
  --epochs 25 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --num-workers 4
```

The script:

- Wraps the FER-2013 CSV via `sentivis.data.Fer2013Dataset`
- Uses a lightweight CNN (`EmotionNet`) optimized for 48×48 grayscale inputs
- Tracks validation accuracy, keeps the best checkpoint, and finally reports test accuracy on the `PrivateTest` split
- Saves a checkpoint with the `state_dict`, label map, and metrics to the path you provide

Once the weights exist at `backend/artifacts/emotion-net.pt` (or the location referenced by `SENTIVIS_EMOTION_WEIGHTS`), restart the backend. The API will begin sending emotion labels and confidences for each tracked face, and the frontend dashboard will show “Emotion Model Ready”.

## How inference works

1. The browser captures 320×240 JPEG frames at ~6–7 fps and sends them via Socket.IO.
2. The backend decodes the frame, detects faces with MediaPipe, crops each face, and runs the trained FER classifier if weights are available.
3. The backend emits an `inference` event per frame that contains:
   - Bounding-box coordinates and detection score for each face
   - Emotion prediction (`label`, confidence, and the full probability vector)
   - Model health metadata (`models.emotion`), latency metrics, and frame IDs
4. The frontend mirrors the webcam feed, draws bounding boxes + emotion labels, and surfaces latency/model status cards.

## Troubleshooting

- **No emotion label on the UI:** Verify that `backend/artifacts/emotion-net.pt` exists (or update `SENTIVIS_EMOTION_WEIGHTS`). The stats card will show “Not loaded” until the checkpoint is found.
- **Training is slow:** Lower `--batch-size` or `--epochs`. You can also set `--num-workers 0` if multiprocessing causes issues on macOS.
- **Socket connection failures:** Ensure the backend is reachable at the URL set in `NEXT_PUBLIC_SENTIVIS_SOCKET_URL` and that CORS/network settings allow websocket traffic.
