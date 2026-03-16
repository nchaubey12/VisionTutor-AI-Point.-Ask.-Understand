/**
 * useWebSocket - Fixed for React StrictMode double-mount
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
    if (evt.code !== 1000) {
      scheduleReconnect();
    }
  };

  ws.onerror = () => {
    _connecting = false;
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

export function useWebSocket({ onMessage, onConnect, onDisconnect }: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId,   setSessionId]   = useState<string | null>(null);

  const onMessageRef    = useRef(onMessage);
  const onConnectRef    = useRef(onConnect);
  const onDisconnectRef = useRef(onDisconnect);
  onMessageRef.current    = onMessage;
  onConnectRef.current    = onConnect;
  onDisconnectRef.current = onDisconnect;

  useEffect(() => {
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

    if (_socket?.readyState === WebSocket.OPEN) {
      setIsConnected(true);
    }

    connectSocket();

    return () => {
      _listeners.delete(msgCb);
      _connectCbs.delete(connCb);
      _disconnCbs.delete(discCb);
    };
  }, []);

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

  // ── NEW: interrupt — stops backend mid-stream immediately ──
  const sendInterrupt   = useCallback(() =>
    sendToSocket({ type: "interrupt" }), []);

  return {
    isConnected, sessionId, send,
    sendFrame, sendVoiceInput,
    requestDiagram, requestPractice, requestNextStep, resetSession,
    sendInterrupt,
  };
}