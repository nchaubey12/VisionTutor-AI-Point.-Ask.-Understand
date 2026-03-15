/**
 * useWebSocket - Fixed for React StrictMode double-mount
 *
 * Root cause: React StrictMode (enabled by default in Next.js dev) mounts
 * every component TWICE. This causes useEffect to run → cleanup → run again
 * in rapid succession, creating and immediately destroying WebSockets.
 *
 * Fix: use a module-level singleton so the socket survives the double-mount.
 */

import { useEffect, useRef, useCallback, useState } from "react";

export type MessageType =
  | "connected" | "frame_analyzed" | "text_chunk"
  | "explanation_start" | "explanation_complete"
  | "diagram" | "interrupted" | "response_start"
  | "generating_diagram" | "generating_practice"
  | "practice_question" | "session_reset" | "error" | "status";

export interface TutorMessage {
  type: MessageType;
  [key: string]: unknown;
}

interface UseWebSocketOptions {
  onMessage:     (msg: TutorMessage) => void;
  onConnect?:    () => void;
  onDisconnect?: () => void;
}

// ── Module-level singleton ────────────────────────────────────────────────────
// Lives outside React — survives StrictMode double-mount unmount/remount cycle
let _socket:      WebSocket | null = null;
let _listeners:   Set<(msg: TutorMessage) => void> = new Set();
let _connectCbs:  Set<() => void> = new Set();
let _disconnCbs:  Set<() => void> = new Set();
let _reconnTimer: ReturnType<typeof setTimeout> | null = null;
let _connecting   = false;

function getWsUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.hostname}:8080/ws/tutor`;
}

function connectSocket() {
  // Already open or connecting — do nothing
  if (_connecting) return;
  if (_socket && (_socket.readyState === WebSocket.OPEN ||
                  _socket.readyState === WebSocket.CONNECTING)) return;

  _connecting = true;
  const url = getWsUrl();
  console.log("[WS] Connecting:", url);

  let ws: WebSocket;
  try {
    ws = new WebSocket(url);
  } catch (e) {
    console.error("[WS] Failed to create socket:", e);
    _connecting = false;
    scheduleReconnect();
    return;
  }

  _socket = ws;

  ws.onopen = () => {
    _connecting = false;
    console.log("[WS] Connected ✓");
    _connectCbs.forEach(cb => cb());
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data) as TutorMessage;
      _listeners.forEach(cb => cb(msg));
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  ws.onclose = (evt) => {
    _connecting = false;
    console.log(`[WS] Closed (code=${evt.code})`);
    _socket = null;
    _disconnCbs.forEach(cb => cb());
    // Only reconnect on abnormal close (not intentional close code 1000)
    if (evt.code !== 1000) {
      scheduleReconnect();
    }
  };

  ws.onerror = () => {
    _connecting = false;
    // onclose fires after onerror — reconnect handled there
  };
}

function scheduleReconnect() {
  if (_reconnTimer) return;
  _reconnTimer = setTimeout(() => {
    _reconnTimer = null;
    connectSocket();
  }, 3000);
}

function sendToSocket(data: object) {
  if (_socket?.readyState === WebSocket.OPEN) {
    _socket.send(JSON.stringify(data));
  } else {
    console.warn("[WS] Not connected — cannot send:", data);
  }
}

// ── React hook ────────────────────────────────────────────────────────────────

export function useWebSocket({ onMessage, onConnect, onDisconnect }: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId,   setSessionId]   = useState<string | null>(null);

  // Stable refs so callbacks don't need to be in dependency arrays
  const onMessageRef    = useRef(onMessage);
  const onConnectRef    = useRef(onConnect);
  const onDisconnectRef = useRef(onDisconnect);
  onMessageRef.current    = onMessage;
  onConnectRef.current    = onConnect;
  onDisconnectRef.current = onDisconnect;

  useEffect(() => {
    // Register this component's callbacks
    const msgCb = (msg: TutorMessage) => {
      if (msg.type === "connected" && msg.session_id) {
        setSessionId(msg.session_id as string);
      }
      onMessageRef.current(msg);
    };

    const connCb = () => {
      setIsConnected(true);
      onConnectRef.current?.();
    };

    const discCb = () => {
      setIsConnected(false);
      setSessionId(null);
      onDisconnectRef.current?.();
    };

    _listeners.add(msgCb);
    _connectCbs.add(connCb);
    _disconnCbs.add(discCb);

    // Sync initial state if socket already open
    if (_socket?.readyState === WebSocket.OPEN) {
      setIsConnected(true);
    }

    // Start connection (idempotent — won't double-connect)
    connectSocket();

    return () => {
      // Cleanup: remove this component's callbacks
      // Do NOT close the socket — it's a singleton that outlives remounts
      _listeners.delete(msgCb);
      _connectCbs.delete(connCb);
      _disconnCbs.delete(discCb);
    };
  }, []); // Empty deps — run once per actual mount

  const sendFrame = useCallback((image: string, force = false) => {
  console.log("sendFrame: socket readyState =", _socket?.readyState, "| image length =", image?.length);
  sendToSocket({ type: "frame", image, force_reanalyze: force });
}, []);
  

  const sendVoiceInput  = useCallback((text: string) =>
    sendToSocket({ type: "voice_input", text }), []);

  const requestDiagram  = useCallback((concept: string) =>
    sendToSocket({ type: "request_diagram", concept }), []);

  const requestPractice = useCallback(() =>
    sendToSocket({ type: "request_practice" }), []);

  const requestNextStep = useCallback(() =>
    sendToSocket({ type: "next_step" }), []);

  const resetSession    = useCallback(() =>
    sendToSocket({ type: "new_session" }), []);

  const send            = useCallback((data: object) =>
    sendToSocket(data), []);

  return {
    isConnected, sessionId, send,
    sendFrame, sendVoiceInput,
    requestDiagram, requestPractice, requestNextStep, resetSession,
  };
}
