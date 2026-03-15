/**
 * useWebcam - Custom hook for webcam streaming and frame capture
 */

import { useEffect, useRef, useCallback, useState } from "react";

interface UseWebcamOptions {
  onFrame?: (base64Frame: string) => void;
  captureInterval?: number;
  autoCapture?: boolean;
}

export function useWebcam({
  onFrame,
  captureInterval = 3000,
  autoCapture = false,
}: UseWebcamOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout>();

  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState(false);

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Guard: video must be playing and have real dimensions
    if (!video || !canvas) return null;
    if (video.readyState < 2) return null;        // HAVE_CURRENT_DATA
    if (!video.videoWidth || !video.videoHeight) return null;

    // Capture at 640×480 — plenty for Gemini Vision, keeps payload small
    canvas.width = 640;
    canvas.height = 480;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, 640, 480);

    // Return full data URL (gemini_service strips the prefix itself)
    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError(null);

      // Don't request facingMode on desktop — causes hangs or wrong camera
      const isMobile = /Mobi|Android/i.test(navigator.userAgent);

      const constraints: MediaStreamConstraints = {
        video: isMobile
          ? { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      setHasPermission(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        // Wait for video to actually be ready before marking active
        await new Promise<void>((resolve) => {
          const v = videoRef.current!;
          if (v.readyState >= 2) { resolve(); return; }
          v.onloadeddata = () => resolve();
        });

        await videoRef.current.play();
        setIsActive(true);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Camera access denied";
      setError(message);
      console.error("Camera error:", err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsActive(false);
    clearInterval(captureIntervalRef.current);
  }, []);

  // Auto-capture frames at interval (only used if autoCapture=true)
  useEffect(() => {
    if (autoCapture && isActive && onFrame) {
      captureIntervalRef.current = setInterval(() => {
        const frame = captureFrame();
        if (frame) onFrame(frame);
      }, captureInterval);
      return () => clearInterval(captureIntervalRef.current);
    }
  }, [autoCapture, isActive, onFrame, captureFrame, captureInterval]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  return {
    videoRef,
    canvasRef,
    isActive,
    error,
    hasPermission,
    startCamera,
    stopCamera,
    captureFrame,
  };
}