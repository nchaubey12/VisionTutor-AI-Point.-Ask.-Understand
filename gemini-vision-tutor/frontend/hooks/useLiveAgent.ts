/**
 * useLiveAgent - Gemini Live API real-time audio+vision hook
 * Fixed: correct model (gemini-2.5-flash-native-audio-latest),
 *        stopAudio signal on mute, input/output transcription support
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface LiveMessage {
  type: "connected" | "audio" | "text" | "interrupted" | "turn_complete"
      | "error" | "reconnected" | "gate_reset" | "status"
      | "input_transcription" | "output_transcription";
  data?: string;
  text?: string;
  message?: string;
}

interface UseLiveAgentOptions {
  onMessage?:           (msg: LiveMessage) => void;
  onTranscript?:        (text: string) => void;
  onInputTranscript?:   (text: string) => void;   // ✅ what user said
  onOutputTranscript?:  (text: string) => void;   // ✅ what Gemini said
  onInterrupted?:       () => void;
  onTurnComplete?:      () => void;
  onStatus?:            (msg: string) => void;
  videoRef: React.RefObject<HTMLVideoElement>;
}

const WS_URL =
  process.env.NEXT_PUBLIC_LIVE_WS_URL ||
  `ws://${typeof window !== "undefined" ? window.location.hostname : "localhost"}:8081/ws/live`;

const MIC_SAMPLE_RATE   = 16000;
const OUT_SAMPLE_RATE   = 24000;
const FRAME_INTERVAL_MS = 4000;

export function useLiveAgent({
  onMessage,
  onTranscript,
  onInputTranscript,
  onOutputTranscript,
  onInterrupted,
  onTurnComplete,
  onStatus,
  videoRef,
}: UseLiveAgentOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [isMicOn,     setIsMicOn]     = useState(false);
  const [isCameraOn,  setIsCameraOn]  = useState(false);
  const [isSpeaking,  setIsSpeaking]  = useState(false);
  const [isStarting,  setIsStarting]  = useState(false);
  const [error,       setError]       = useState<string | null>(null);

  const wsRef          = useRef<WebSocket | null>(null);
  const audioCtxRef    = useRef<AudioContext | null>(null);
  const micStreamRef   = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef      = useRef<MediaStreamAudioSourceNode | null>(null);
  const frameTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef      = useRef<HTMLCanvasElement | null>(null);
  const outCtxRef      = useRef<AudioContext | null>(null);
  const isMutedRef     = useRef(false);   // ✅ track mute state for worklet

  const nextPlayTimeRef  = useRef<number>(0);
  const isSpeakingRef    = useRef<boolean>(false);
  const scheduledSources = useRef<AudioBufferSourceNode[]>([]);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const sendWs = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const toBase64 = (buf: ArrayBuffer): string => {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
  };

  const fromBase64 = (b64: string): ArrayBuffer => {
    const s = atob(b64);
    const buf = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) buf[i] = s.charCodeAt(i);
    return buf.buffer;
  };

  const ensureOutCtx = useCallback(async (): Promise<AudioContext | null> => {
    if (!outCtxRef.current || outCtxRef.current.state === "closed") {
      outCtxRef.current = new AudioContext({ sampleRate: OUT_SAMPLE_RATE });
    }
    if (outCtxRef.current.state === "suspended") {
      await outCtxRef.current.resume();
    }
    return outCtxRef.current;
  }, []);

  const enqueueAudio = useCallback(async (b64: string) => {
    const ctx = await ensureOutCtx();
    if (!ctx) return;

    const raw    = fromBase64(b64);
    const int16  = new Int16Array(raw);
    const f32    = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768.0;

    const audioBuf = ctx.createBuffer(1, f32.length, OUT_SAMPLE_RATE);
    audioBuf.copyToChannel(f32, 0);

    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, nextPlayTimeRef.current);
    src.start(startAt);
    nextPlayTimeRef.current = startAt + audioBuf.duration;

    if (!isSpeakingRef.current) {
      isSpeakingRef.current = true;
      setIsSpeaking(true);
    }

    scheduledSources.current.push(src);
    src.onended = () => {
      scheduledSources.current = scheduledSources.current.filter(s => s !== src);
      if (scheduledSources.current.length === 0) {
        isSpeakingRef.current = false;
        setIsSpeaking(false);
      }
    };
  }, [ensureOutCtx]);

  const stopAudio = useCallback(() => {
    const now = outCtxRef.current?.currentTime ?? 0;
    for (const src of scheduledSources.current) {
      try { src.stop(now); } catch (_) {}
    }
    scheduledSources.current = [];
    nextPlayTimeRef.current  = 0;
    isSpeakingRef.current    = false;
    setIsSpeaking(false);
  }, []);

  // ── Mic ───────────────────────────────────────────────────────────────────

  const startMic = useCallback(async () => {
    if (isMicOn) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: MIC_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
        video: false,
      });
      micStreamRef.current = stream;
      isMutedRef.current = false;

      const inCtx = new AudioContext({ sampleRate: MIC_SAMPLE_RATE });
      audioCtxRef.current = inCtx;
      if (inCtx.state === "suspended") await inCtx.resume();

      await inCtx.audioWorklet.addModule("/audio-processor.worklet.js");
      const workletNode = new AudioWorkletNode(inCtx, "mic-processor");

      workletNode.port.onmessage = (e: MessageEvent<Int16Array>) => {
        // ✅ Don't send audio when muted — prevents silent PCM flooding Gemini
        if (isMutedRef.current) return;
        const int16 = e.data;
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          sendWs({ type: "audio", data: toBase64(int16.buffer as ArrayBuffer) });
        }
      };

      const src = inCtx.createMediaStreamSource(stream);
      src.connect(workletNode);
      workletNode.connect(inCtx.destination);

      sourceRef.current      = src;
      workletNodeRef.current = workletNode;

      setIsMicOn(true);
      setError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setError(msg);
    }
  }, [isMicOn, sendWs]);

  const stopMic = useCallback(() => {
    workletNodeRef.current?.port.close();
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    micStreamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    setIsMicOn(false);
  }, []);

  // ✅ Mute — stops sending audio chunks AND signals Gemini to respond
  const muteMic = useCallback(() => {
    isMutedRef.current = true;
    if (mediaStream) micStreamRef.current?.getAudioTracks().forEach(t => t.enabled = false);
    // Tell Gemini the user finished speaking so it responds
    sendWs({ type: "stopAudio" });
  }, [sendWs]);

  // ✅ Unmute — re-enables audio stream
  const unmuteMic = useCallback(() => {
    isMutedRef.current = false;
    micStreamRef.current?.getAudioTracks().forEach(t => t.enabled = true);
  }, []);

  // ── Camera ────────────────────────────────────────────────────────────────

  const startCamera = useCallback(() => {
    if (isCameraOn || frameTimerRef.current) return;
    frameTimerRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return;
      if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
      const c = canvasRef.current;
      c.width = 640; c.height = 480;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, 640, 480);
      const b64 = c.toDataURL("image/jpeg", 0.6).split(",", 2)[1];
      sendWs({ type: "image", data: b64 });
    }, FRAME_INTERVAL_MS);
    setIsCameraOn(true);
  }, [isCameraOn, videoRef, sendWs]);

  const stopCamera = useCallback(() => {
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    setIsCameraOn(false);
  }, []);

  // ── Connect / disconnect ──────────────────────────────────────────────────

  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setIsStarting(true);
    setError(null);

    await ensureOutCtx();

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setIsStarting(false);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data) as LiveMessage;
        onMessage?.(msg);

        switch (msg.type) {
          case "audio":
            if (msg.data) enqueueAudio(msg.data);
            break;
          case "text":
            if (msg.text) onTranscript?.(msg.text);
            break;
          // ✅ Input transcription = what the user said
          case "input_transcription":
            if (msg.text) onInputTranscript?.(msg.text);
            break;
          // ✅ Output transcription = what Gemini said
          case "output_transcription":
            if (msg.text) {
              onOutputTranscript?.(msg.text);
              onTranscript?.(msg.text);  // also feed main transcript
            }
            break;
          case "interrupted":
            stopAudio();
            onInterrupted?.();
            break;
          case "turn_complete":
            nextPlayTimeRef.current = 0;
            onTurnComplete?.();
            break;
          case "gate_reset":
            workletNodeRef.current?.port.postMessage({ type: "reset" });
            break;
          case "status":
            if (msg.message) onStatus?.(msg.message as string);
            break;
          case "error":
            setError(msg.message || "Unknown error");
            break;
        }
      } catch (e) {
        console.error("[Live] parse error:", e);
      }
    };

    ws.onclose = (evt) => {
      setIsConnected(false);
      setIsMicOn(false);
      setIsCameraOn(false);
      setIsSpeaking(false);
    };

    ws.onerror = () => {
      setError("Connection failed — is the backend running?");
      setIsStarting(false);
    };
  }, [enqueueAudio, stopAudio, ensureOutCtx, onMessage, onTranscript,
      onInputTranscript, onOutputTranscript, onInterrupted, onTurnComplete, onStatus]);

  const disconnect = useCallback(() => {
    stopMic();
    stopCamera();
    stopAudio();
    sendWs({ type: "disconnect" });
    wsRef.current?.close();
    wsRef.current = null;
    outCtxRef.current?.close().catch(() => {});
    outCtxRef.current = null;
    setIsConnected(false);
    setIsSpeaking(false);
  }, [stopMic, stopCamera, stopAudio, sendWs]);

  const sendText = useCallback((text: string) => {
    sendWs({ type: "text", text });
  }, [sendWs]);

  useEffect(() => () => { disconnect(); }, [disconnect]);

  return {
    isConnected,
    isMicOn,
    isCameraOn,
    isSpeaking,
    isStarting,
    error,
    connect,
    disconnect,
    startMic,
    stopMic,
    muteMic,    
    unmuteMic, 
    startCamera,
    stopCamera,
    sendText,
  };
}