'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Script from 'next/script';

type SocketClient = {
  emit: (event: string, payload?: unknown) => void;
  on: (event: string, callback: (...args: any[]) => void) => void;
  off: (event: string, callback?: (...args: any[]) => void) => void;
  disconnect: () => void;
  connected: boolean;
};

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
};

type EmotionSummary = {
  label: string;
  confidence: number;
  scores: number[];
};

type FaceInference = {
  box: FaceBox;
  emotion: EmotionSummary;
};

type InferencePayload = {
  faces: FaceInference[];
  metrics?: {
    latencyMs?: number;
  };
};

declare global {
  interface Window {
    io?: (url: string, config?: Record<string, unknown>) => SocketClient;
  }
}

const CAPTURE_WIDTH = 320;
const CAPTURE_HEIGHT = 240;
const STREAM_INTERVAL_MS = 150;
const SOCKET_URL =
  process.env.NEXT_PUBLIC_SENTIVIS_SOCKET_URL ?? 'http://localhost:5001';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<SocketClient | null>(null);
  const frameIdRef = useRef(0);

  const [scriptLoaded, setScriptLoaded] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'connecting' | 'ready' | 'error'>('idle');
  const [latestFaces, setLatestFaces] = useState<FaceInference[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const displayEmotion = useMemo(() => {
    if (!latestFaces.length) {
      return { label: 'No face detected', confidence: 0 };
    }

    const { label, confidence } = latestFaces[0].emotion;
    return {
      label: label.toUpperCase(),
      confidence: Math.round(confidence * 100),
    };
  }, [latestFaces]);

  const drawOverlay = useCallback(
    (faces: FaceInference[]) => {
      const canvas = overlayCanvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const context = canvas.getContext('2d');
      if (!context) return;

      context.clearRect(0, 0, canvas.width, canvas.height);

      context.strokeStyle = '#22c55e';
      context.lineWidth = 3;
      context.font = '16px sans-serif';
      context.fillStyle = '#22c55e';

      faces.forEach((face) => {
        const x = face.box.x * canvas.width;
        const y = face.box.y * canvas.height;
        const width = face.box.width * canvas.width;
        const height = face.box.height * canvas.height;

        context.strokeRect(x, y, width, height);

        const label = `${face.emotion.label} ${(face.emotion.confidence * 100).toFixed(0)}%`;
        const labelWidth = context.measureText(label).width + 12;
        const labelX = Math.max(0, Math.min(canvas.width - labelWidth, x));
        const labelY = y - 28 < 0 ? y + height + 4 : y - 28;

        context.fillRect(labelX, labelY, labelWidth, 24);
        context.fillStyle = '#0f172a';
        context.fillText(label, labelX + 6, labelY + 16);
        context.fillStyle = '#22c55e';
      });
    },
    [],
  );

  useEffect(() => {
    drawOverlay(latestFaces);
  }, [latestFaces, drawOverlay]);

  useEffect(() => {
    captureCanvasRef.current = document.createElement('canvas');
    captureCanvasRef.current.width = CAPTURE_WIDTH;
    captureCanvasRef.current.height = CAPTURE_HEIGHT;

    return () => {
      captureCanvasRef.current = null;
    };
  }, []);

  useEffect(() => {
    const openCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 640, height: 480 },
        });

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play();

          const canvas = overlayCanvasRef.current;
          if (canvas) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
        }
      } catch (error) {
        setErrorMessage('Camera access is required for Sentivis.');
        console.error(error);
      }
    };

    openCamera();

    return () => {
      const video = videoRef.current;
      const stream = video?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!scriptLoaded) return;
    if (!window.io || socketRef.current) return;

    setConnectionStatus('connecting');
    setErrorMessage(null);

    const socket = window.io(SOCKET_URL, {
      transports: ['websocket'],
      reconnectionAttempts: 5,
    }) as SocketClient;

    socketRef.current = socket;

    const handleInference = (payload: InferencePayload) => {
      setLatestFaces(payload.faces ?? []);
      if (payload.metrics?.latencyMs !== undefined) {
        setLatency(Math.round(payload.metrics.latencyMs));
      }
    };

    const handleStatus = (status: { status: string }) => {
      if (status.status === 'ready') {
        setConnectionStatus('ready');
      }
    };

    const handleError = (payload: { message?: string }) => {
      setErrorMessage(payload.message ?? 'Unexpected backend error');
      setConnectionStatus('error');
    };

    socket.on('connect', () => {
      setConnectionStatus('ready');
    });
    socket.on('disconnect', () => {
      setConnectionStatus('error');
    });
    socket.on('inference', handleInference);
    socket.on('server_status', handleStatus);
    socket.on('error', handleError);

    return () => {
      socket.off('inference', handleInference);
      socket.off('server_status', handleStatus);
      socket.off('error', handleError);
      socket.disconnect();
      socketRef.current = null;
      setConnectionStatus('idle');
    };
  }, [scriptLoaded]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      const socket = socketRef.current;
      const video = videoRef.current;
      const captureCanvas = captureCanvasRef.current;

      if (!socket || !socket.connected || !video || !captureCanvas) {
        return;
      }

      const context = captureCanvas.getContext('2d');
      if (!context) return;

      context.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      const frame = captureCanvas.toDataURL('image/jpeg', 0.7);

      socket.emit('frame', {
        frame,
        frameId: frameIdRef.current++,
        timestamp: Date.now(),
      });
    }, STREAM_INTERVAL_MS);

    return () => window.clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <Script
        src="https://cdn.socket.io/4.7.5/socket.io.min.js"
        strategy="afterInteractive"
        onLoad={() => setScriptLoaded(true)}
      />

      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-8">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold">Sentivis</h1>
          <p className="text-sm text-slate-400">
            Real-time facial sentiment analysis. Position your face inside the frame to see live predictions.
          </p>
        </header>

        <section className="relative aspect-video w-full overflow-hidden rounded-xl border border-slate-800 bg-black">
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full object-cover"
            playsInline
            muted
            autoPlay
          />
          <canvas
            ref={overlayCanvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
          {connectionStatus !== 'ready' && (
            <div className="absolute bottom-4 left-4 rounded-lg bg-slate-900/80 px-4 py-2 text-sm text-slate-300">
              {connectionStatus === 'connecting'
                ? 'Connecting to sentiment engine...'
                : connectionStatus === 'error'
                  ? 'Disconnected from backend'
                  : 'Initializing...'}
            </div>
          )}
        </section>

        <section className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4 md:grid-cols-3">
          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-4">
            <h2 className="text-xs uppercase text-slate-500">Current Emotion</h2>
            <p className="mt-3 text-2xl font-medium text-slate-100">
              {displayEmotion.label}
            </p>
            {displayEmotion.confidence > 0 && (
              <p className="text-sm text-slate-400">
                Confidence: {displayEmotion.confidence}%
              </p>
            )}
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-4">
            <h2 className="text-xs uppercase text-slate-500">Latency</h2>
            <p className="mt-3 text-2xl font-medium text-slate-100">
              {latency !== null ? `${latency} ms` : '—'}
            </p>
            <p className="text-sm text-slate-400">Round-trip inference time</p>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-4">
            <h2 className="text-xs uppercase text-slate-500">Status</h2>
            <p className="mt-3 text-2xl font-medium text-slate-100 capitalize">
              {connectionStatus}
            </p>
            {errorMessage && (
              <p className="text-sm text-rose-400">{errorMessage}</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
