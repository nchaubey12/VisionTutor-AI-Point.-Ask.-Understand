"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  CameraOff, Camera, Mic, MicOff, RefreshCw, BookOpen, Zap,
  ChevronRight, Volume2, VolumeX, Sparkles, AlertCircle,
  CheckCircle2, Brain, Lightbulb, Loader2, Wifi, WifiOff, Radio,
} from "lucide-react";
import { useWebSocket, TutorMessage } from "../hooks/useWebSocket";
import { useWebcam } from "../hooks/useWebcam";
import { useSpeech } from "../hooks/useSpeech";
import { useLiveAgent } from "../hooks/useLiveAgent";

interface ProblemInfo { subject: string; problem: string; difficulty: string; has_errors: boolean; total_steps: number; }
interface DiagramData { svg: string; concept: string; }
interface PracticeQuestion { question: string; hint: string; answer: string; }

export default function TutorPage() {
  const [mode, setMode] = useState<"standard" | "live">("standard");
  const [problemInfo, setProblemInfo]           = useState<ProblemInfo | null>(null);
  const [explanation, setExplanation]           = useState("");
  const [currentStep, setCurrentStep]           = useState(0);
  const [totalSteps, setTotalSteps]             = useState(1);
  const [stepTitle, setStepTitle]               = useState("");
  const [diagram, setDiagram]                   = useState<DiagramData | null>(null);
  const [practice, setPractice]                 = useState<PracticeQuestion | null>(null);
  const [followUpQuestion, setFollowUpQuestion] = useState("");
  const [isMuted, setIsMuted]                   = useState(false);
  const [showPractice, setShowPractice]         = useState(false);
  const [showAnswer, setShowAnswer]             = useState(false);
  const [isAnalyzing, setIsAnalyzing]           = useState(false);
  const [keepCameraOn, setKeepCameraOn]         = useState(false);
  const [isExplaining, setIsExplaining]         = useState(false);
  const [isGenerating, setIsGenerating]         = useState(false);
  const [liveTranscript, setLiveTranscript]     = useState("");
  const [statusMsg, setStatusMsg]               = useState("Point camera at homework, then click Analyze");
  const [cameraFrameColor, setCameraFrameColor] = useState<"default"|"green"|"red">("default");
  const explanationRef = useRef<HTMLDivElement>(null);

  const { videoRef, canvasRef, isActive, error: cameraError, startCamera, stopCamera, captureFrame } = useWebcam();

  const { isListening, isSpeaking, isSupported: speechSupported, startListening, stopListening, speak, stopSpeaking } = useSpeech({
    onTranscript: (text) => { setStatusMsg(`You said: "${text}"`); sendVoiceInput(text); },
  });

  const handleMessage = useCallback((message: TutorMessage) => {
    switch (message.type) {
      case "connected": setStatusMsg("Connected — ready to tutor"); break;
      case "frame_analyzed":
        setIsAnalyzing(false); setCameraFrameColor("green");
        setTimeout(() => setCameraFrameColor("default"), 2000);
        setProblemInfo({ subject: message.subject as string, problem: message.problem as string, difficulty: message.difficulty as string, has_errors: message.has_errors as boolean, total_steps: message.total_steps as number });
        setTotalSteps(message.total_steps as number);
        setStatusMsg(`Detected: ${message.subject}`);
        break;
      case "explanation_start":
        setIsExplaining(true); setCurrentStep(message.step as number);
        setStepTitle(message.step_title as string || `Step ${(message.step as number) + 1}`);
        setExplanation(""); break;
      case "text_chunk": setExplanation(prev => prev + (message.text as string)); break;
      case "explanation_complete": {
        setIsExplaining(false);
        const ft = message.full_text as string;
        setExplanation(ft);
        if (message.follow_up) setFollowUpQuestion(message.follow_up as string);
        setStatusMsg("Explanation complete");
        if (!isMuted && ft) speak(ft);
        break;
      }
      case "diagram": setDiagram({ svg: message.svg as string, concept: message.concept as string }); setIsGenerating(false); break;
      case "interrupted": stopSpeaking(); setIsExplaining(false); break;
      case "response_start": setExplanation(""); setIsExplaining(true); setStepTitle("Answering..."); break;
      case "generating_diagram": setIsGenerating(true); break;
      case "generating_practice": setIsGenerating(true); break;
      case "practice_question":
        setPractice({ question: message.question as string, hint: message.hint as string, answer: message.answer as string });
        setShowPractice(true); setShowAnswer(false); setIsGenerating(false); break;
      case "session_reset":
        setProblemInfo(null); setExplanation(""); setDiagram(null); setPractice(null);
        setCurrentStep(0); setShowPractice(false); setCameraFrameColor("default");
        setStatusMsg("Session reset"); break;
      case "error":
        setIsAnalyzing(false); setIsExplaining(false); setIsGenerating(false);
        setCameraFrameColor("red"); setTimeout(() => setCameraFrameColor("default"), 3000);
        setStatusMsg(`${message.message as string}`); break;
    }
  }, [isMuted, speak, stopSpeaking]);

  const { isConnected, sendFrame, sendVoiceInput, requestDiagram, requestPractice, requestNextStep, resetSession } = useWebSocket({
    onMessage: handleMessage,
    onConnect: () => setStatusMsg("Connected — ready to tutor"),
    onDisconnect: () => setStatusMsg("Reconnecting..."),
  });

  // ── Live agent callbacks ──────────────────────────────────────────────────
  // These are defined before useLiveAgent so they can be passed in cleanly.

  // FIX: When Gemini is interrupted, stop sending mic audio immediately so
  // Gemini doesn't receive the tail of the interruption as a new question.
  // We restart the mic 600ms later so it's ready for the follow-up.
  const handleInterrupted = useCallback(() => {
    setStatusMsg("Listening...");
  }, []);

  // FIX: After Gemini finishes a turn, do a brief mic stop/start cycle.
  // This flushes any stale audio buffered in the worklet and ensures the
  // AudioWorklet onmessage closure is bound to the current WebSocket.
  // Without this, subsequent questions are sent but Gemini ignores them
  // because the stream context is stale from the previous turn.
  const handleTurnComplete = useCallback(() => {
    setStatusMsg("Listening...");
  }, []);

  const {
    isConnected: liveConnected,
    isMicOn,
    isCameraOn,
    isSpeaking: liveGeminiSpeaking,
    isStarting: liveStarting,
    error: liveError,
    connect: liveConnect,
    disconnect: liveDisconnect,
    startMic,
    stopMic,
    startCamera: liveStartCamera,
    stopCamera: liveStopCamera,
  } = useLiveAgent({
    videoRef,
    onTranscript: (text) => setLiveTranscript(prev => prev + text + " "),
    onInterrupted: handleInterrupted,
    onTurnComplete: handleTurnComplete,
    onStatus: (msg) => setStatusMsg(msg),
  });

  // Gemini Live is full-duplex — it handles its own echo cancellation.
  // Never pause the mic: pausing causes a VAD silence gap that makes
  // Gemini clear its audio buffer and ask "what was your question?".
  // Just update the status label based on who is speaking.
  useEffect(() => {
    if (mode !== "live" || !liveConnected) return;
    if (liveGeminiSpeaking) {
      setStatusMsg("Gemini speaking — just talk to interrupt");
    } else if (isMicOn) {
      setStatusMsg("Listening — ask your question");
    }
  }, [liveGeminiSpeaking, mode, liveConnected, isMicOn]);

  const switchToLive = useCallback(() => { stopSpeaking(); setMode("live"); liveConnect(); }, [stopSpeaking, liveConnect]);
  const switchToStandard = useCallback(() => { liveDisconnect(); setMode("standard"); setLiveTranscript(""); setStatusMsg("Standard mode"); }, [liveDisconnect]);

  const handleAnalyze = useCallback(() => {
    if (!isActive || !isConnected) return;
    const frame = captureFrame();
    if (!frame) return;
    setIsAnalyzing(true);
    sendFrame(frame, true);
    if (!keepCameraOn) stopCamera();
  }, [isActive, isConnected, captureFrame, sendFrame, keepCameraOn, stopCamera]);

  useEffect(() => {
    if (explanationRef.current) explanationRef.current.scrollTop = explanationRef.current.scrollHeight;
  }, [explanation, liveTranscript]);

  const camBorderColor =
    cameraFrameColor === "green" ? "#10b981" :
    cameraFrameColor === "red"   ? "#ef4444" :
    mode === "live" && liveGeminiSpeaking ? "#3b82f6" :
    mode === "live" && isMicOn            ? "#10b981" :
    "rgba(255,255,255,0.08)";

  return (
    <div style={{ minHeight: "100vh", background: "#080810", color: "#fff", fontFamily: "'DM Sans', system-ui, sans-serif", display: "flex", flexDirection: "column" }}>

      {/* Ambient bg */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: "-30%", left: "-15%", width: 700, height: 700, background: "radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)", borderRadius: "50%" }} />
        <div style={{ position: "absolute", bottom: "-20%", right: "-10%", width: 600, height: 600, background: "radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%)", borderRadius: "50%" }} />
        <div style={{ position: "absolute", top: "40%", left: "40%", width: 400, height: 400, background: "radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)", borderRadius: "50%" }} />
      </div>

      {/* Header */}
      <header style={{ position: "relative", zIndex: 10, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 28px", height: 56, borderBottom: "1px solid rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", background: "rgba(8,8,16,0.8)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, borderRadius: 10, background: "linear-gradient(135deg, #6366f1, #3b82f6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Brain size={16} color="#fff" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.02em" }}>VisionTutor AI</span>
          {mode === "live" ? (
            <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 10px", borderRadius: 20, background: "rgba(16,185,129,0.15)", color: "#10b981", border: "1px solid rgba(16,185,129,0.3)", display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#10b981", animation: "pulse 1.5s infinite" }} />
              GEMINI LIVE
            </span>
          ) : (
            <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 10px", borderRadius: 20, background: "rgba(99,102,241,0.15)", color: "#a5b4fc", border: "1px solid rgba(99,102,241,0.3)" }}>
              GEMINI 2.5 FLASH
            </span>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={mode === "standard" ? switchToLive : switchToStandard} disabled={liveStarting}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: "pointer", transition: "all 0.2s", border: mode === "live" ? "1px solid rgba(239,68,68,0.4)" : "1px solid rgba(16,185,129,0.3)", background: mode === "live" ? "rgba(239,68,68,0.1)" : "rgba(16,185,129,0.1)", color: mode === "live" ? "#f87171" : "#10b981" }}>
            {liveStarting ? <><Loader2 size={11} style={{ animation: "spin 1s linear infinite" }} /> Starting...</> :
             mode === "live" ? <><Radio size={11} /> Stop Live</> : <><Radio size={11} /> Go Live</>}
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 500, background: (mode === "live" ? liveConnected : isConnected) ? "rgba(16,185,129,0.1)" : "rgba(239,68,68,0.1)", border: (mode === "live" ? liveConnected : isConnected) ? "1px solid rgba(16,185,129,0.3)" : "1px solid rgba(239,68,68,0.3)", color: (mode === "live" ? liveConnected : isConnected) ? "#10b981" : "#f87171" }}>
            {(mode === "live" ? liveConnected : isConnected) ? <Wifi size={11} /> : <WifiOff size={11} />}
            {(mode === "live" ? liveConnected : isConnected) ? "Connected" : "Connecting"}
          </div>

          <button onClick={() => { stopSpeaking(); resetSession(); }} style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.4)" }}>
            <RefreshCw size={13} />
          </button>
        </div>
      </header>

      {/* Main */}
      <main style={{ flex: 1, display: "flex", position: "relative", zIndex: 10, overflow: "hidden" }}>

        {/* LEFT — Camera + controls */}
        <div style={{ width: 320, flexShrink: 0, display: "flex", flexDirection: "column", padding: "16px 12px 16px 16px", gap: 12, borderRight: "1px solid rgba(255,255,255,0.05)" }}>

          {/* Camera */}
          <div style={{ position: "relative", borderRadius: 16, overflow: "hidden", background: "#0d0d1a", aspectRatio: "4/3", border: `2px solid ${camBorderColor}`, transition: "border-color 0.4s" }}>
            <video ref={videoRef} autoPlay muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {!isActive && (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.85)", gap: 8 }}>
                <CameraOff size={24} color="rgba(255,255,255,0.2)" />
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.3)" }}>Camera off</span>
              </div>
            )}

            {isAnalyzing && (
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                <div style={{ position: "absolute", inset: "0 0 auto 0", height: 2, background: "linear-gradient(90deg, transparent, #6366f1, transparent)", animation: "scan 1.8s linear infinite" }} />
                <div style={{ position: "absolute", inset: 0, background: "rgba(99,102,241,0.04)", animation: "pulse 1s ease-in-out infinite" }} />
              </div>
            )}

            {mode === "live" && liveGeminiSpeaking && (
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "rgba(59,130,246,0.05)", animation: "pulse 1.5s ease-in-out infinite" }} />
            )}
            {mode === "live" && isMicOn && !liveGeminiSpeaking && (
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "rgba(16,185,129,0.04)", animation: "pulse 2s ease-in-out infinite" }} />
            )}

            {["tl","tr","bl","br"].map(c => (
              <div key={c} style={{ position: "absolute", width: 16, height: 16, top: c.startsWith("t") ? 8 : "auto", bottom: c.startsWith("b") ? 8 : "auto", left: c.endsWith("l") ? 8 : "auto", right: c.endsWith("r") ? 8 : "auto", borderTop: c.startsWith("t") ? `2px solid ${camBorderColor}` : "none", borderBottom: c.startsWith("b") ? `2px solid ${camBorderColor}` : "none", borderLeft: c.endsWith("l") ? `2px solid ${camBorderColor}` : "none", borderRight: c.endsWith("r") ? `2px solid ${camBorderColor}` : "none", transition: "border-color 0.4s", borderRadius: c === "tl" ? "4px 0 0 0" : c === "tr" ? "0 4px 0 0" : c === "bl" ? "0 0 0 4px" : "0 0 4px 0" }} />
            ))}

            {(isAnalyzing || cameraFrameColor === "green" || (mode === "live" && liveConnected)) && (
              <div style={{ position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)", display: "flex", alignItems: "center", gap: 5, padding: "4px 10px", background: "rgba(0,0,0,0.75)", borderRadius: 20, backdropFilter: "blur(8px)" }}>
                {isAnalyzing && <><Loader2 size={10} style={{ animation: "spin 1s linear infinite", color: "#6366f1" }} /><span style={{ fontSize: 10, color: "#a5b4fc" }}>Analyzing...</span></>}
                {cameraFrameColor === "green" && <><CheckCircle2 size={10} color="#10b981" /><span style={{ fontSize: 10, color: "#10b981" }}>Detected!</span></>}
                {mode === "live" && liveConnected && !isAnalyzing && cameraFrameColor !== "green" && (
                  <><span style={{ width: 6, height: 6, borderRadius: "50%", background: liveGeminiSpeaking ? "#3b82f6" : isMicOn ? "#10b981" : "rgba(255,255,255,0.2)", animation: (liveGeminiSpeaking || isMicOn) ? "pulse 1s infinite" : "none" }} />
                  <span style={{ fontSize: 10, color: "rgba(255,255,255,0.5)" }}>{liveGeminiSpeaking ? "Gemini speaking" : isMicOn ? "Listening" : "Mic off"}</span></>
                )}
              </div>
            )}
          </div>

          {/* Action buttons */}
          {mode === "standard" ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={isActive ? stopCamera : startCamera}
                  style={{ flex: "0 0 auto", padding: "11px 14px", borderRadius: 12, border: isActive ? "1px solid rgba(59,130,246,0.4)" : "1px solid rgba(255,255,255,0.08)", background: isActive ? "rgba(59,130,246,0.12)" : "rgba(255,255,255,0.04)", color: isActive ? "#60a5fa" : "rgba(255,255,255,0.4)", fontSize: 12, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                  {isActive ? <><Camera size={13} /> On</> : <><CameraOff size={13} /> Off</>}
                </button>
                <button onClick={handleAnalyze} disabled={isAnalyzing || !isActive || !isConnected}
                  style={{ flex: 1, padding: "11px 0", borderRadius: 12, border: "none", fontWeight: 700, fontSize: 13, cursor: isAnalyzing || !isActive || !isConnected ? "not-allowed" : "pointer", opacity: isAnalyzing || !isActive || !isConnected ? 0.5 : 1, background: isAnalyzing ? "rgba(99,102,241,0.3)" : "linear-gradient(135deg, #6366f1, #3b82f6)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", gap: 7, transition: "all 0.2s" }}>
                  {isAnalyzing ? <><Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> Analyzing...</> : <><Zap size={14} /> Analyze Homework</>}
                </button>
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", padding: "0 2px" }}>
                <input type="checkbox" checked={keepCameraOn} onChange={e => setKeepCameraOn(e.target.checked)}
                  style={{ width: 14, height: 14, accentColor: "#6366f1", cursor: "pointer" }} />
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>Keep camera on after capture</span>
              </label>
            </div>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => isMicOn ? stopMic() : startMic()}
                disabled={!liveConnected}
                style={{ flex: 1, padding: "10px 0", borderRadius: 12, fontSize: 12, fontWeight: 600, cursor: "pointer", border: isMicOn ? "1px solid rgba(16,185,129,0.5)" : "1px solid rgba(255,255,255,0.1)", background: isMicOn ? "rgba(16,185,129,0.15)" : "rgba(255,255,255,0.05)", color: isMicOn ? "#10b981" : "rgba(255,255,255,0.4)", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, opacity: !liveConnected ? 0.4 : 1 }}>
                {isMicOn
                  ? <><Mic size={13} style={{ animation: "pulse 1.5s infinite" }} /> Mute</>
                  : <><MicOff size={13} /> Unmute</>}
              </button>
              <button onClick={() => {
                  if (isCameraOn) { liveStopCamera(); stopCamera(); }
                  else { startCamera(); liveStartCamera(); }
                }} disabled={!liveConnected}
                style={{ flex: 1, padding: "10px 0", borderRadius: 12, fontSize: 12, fontWeight: 600, cursor: "pointer", border: isCameraOn ? "1px solid rgba(59,130,246,0.5)" : "1px solid rgba(255,255,255,0.1)", background: isCameraOn ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.05)", color: isCameraOn ? "#60a5fa" : "rgba(255,255,255,0.4)", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, opacity: !liveConnected ? 0.4 : 1 }}>
                {isCameraOn ? <><Camera size={13} /> Camera On</> : <><CameraOff size={13} /> Camera Off</>}
              </button>
            </div>
          )}

          {/* Problem card */}
          {mode === "standard" && problemInfo && (
            <div style={{ borderRadius: 12, padding: "12px 14px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{problemInfo.subject}</span>
                <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.4)", border: "1px solid rgba(255,255,255,0.08)" }}>{problemInfo.difficulty}</span>
              </div>
              <p style={{ fontSize: 13, color: "rgba(255,255,255,0.8)", lineHeight: 1.5, margin: 0, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{problemInfo.problem}</p>
              {problemInfo.has_errors && (
                <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#fbbf24" }}>
                  <AlertCircle size={10} /><span>Errors detected in your work</span>
                </div>
              )}
              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 4 }}>
                {Array.from({ length: totalSteps }).map((_, i) => (
                  <div key={i} style={{ flex: 1, height: 3, borderRadius: 2, background: i <= currentStep ? "#6366f1" : "rgba(255,255,255,0.1)", transition: "background 0.3s" }} />
                ))}
                <span style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", marginLeft: 4, whiteSpace: "nowrap" }}>{currentStep + 1}/{totalSteps}</span>
              </div>
            </div>
          )}

          {/* Live info */}
          {mode === "live" && liveConnected && (
            <div style={{ borderRadius: 12, padding: "12px 14px", background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.15)" }}>
              <p style={{ fontSize: 12, color: "rgba(255,255,255,0.45)", lineHeight: 1.6, margin: 0 }}>
                Enable mic to speak. Enable camera so Gemini sees your homework. Interrupt at any time.
              </p>
              {liveError && <p style={{ marginTop: 6, fontSize: 11, color: "#f87171" }}>{liveError}</p>}
            </div>
          )}

          <p style={{ fontSize: 11, color: "rgba(255,255,255,0.22)", textAlign: "center", margin: 0, lineHeight: 1.5 }}>{statusMsg}</p>
        </div>

        {/* CENTER — Main content */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "16px 12px", gap: 10, minWidth: 0 }}>

          {(stepTitle || mode === "live") && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0, padding: "0 4px" }}>
              <div style={{ width: 3, height: 20, borderRadius: 2, background: mode === "live" ? (liveGeminiSpeaking ? "#3b82f6" : isMicOn ? "#10b981" : "rgba(255,255,255,0.15)") : "linear-gradient(180deg, #6366f1, #3b82f6)", transition: "background 0.3s" }} />
              <h2 style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.7)", margin: 0 }}>
                {mode === "live" ? "Live Session" : stepTitle}
              </h2>
              {isExplaining && <Loader2 size={12} style={{ animation: "spin 1s linear infinite", color: "#6366f1" }} />}
              {mode === "live" && liveGeminiSpeaking && <span style={{ fontSize: 11, color: "#60a5fa" }}>Gemini speaking...</span>}
              {mode === "live" && isMicOn && !liveGeminiSpeaking && <span style={{ fontSize: 11, color: "#10b981" }}>Listening...</span>}
            </div>
          )}

          <div ref={explanationRef} style={{ flex: 1, overflowY: "auto", borderRadius: 16, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", padding: "20px 24px" }}>

            {mode === "standard" && !explanation && !isAnalyzing && !isExplaining && (
              <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: 28 }}>
                <div style={{ width: 64, height: 64, borderRadius: 20, background: "linear-gradient(135deg, rgba(99,102,241,0.2), rgba(59,130,246,0.2))", border: "1px solid rgba(99,102,241,0.25)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Sparkles size={28} color="#818cf8" />
                </div>
                <div>
                  <h3 style={{ fontSize: 20, fontWeight: 700, color: "rgba(255,255,255,0.7)", margin: "0 0 8px", letterSpacing: "-0.02em" }}>Ready to Tutor</h3>
                  <p style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", maxWidth: 320, lineHeight: 1.6, margin: "0 auto" }}>
                    Point your camera at any homework — math, science, English — and click Analyze. Or switch to Live for real-time voice.
                  </p>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%", maxWidth: 400 }}>
                  {[
                    { icon: "📷", label: "Show homework", desc: "Point camera at your work" },
                    { icon: "⚡", label: "Instant analysis", desc: "Gemini reads it in seconds" },
                    { icon: "🎓", label: "Step-by-step", desc: "Clear explanations with visuals" },
                    { icon: "🎙️", label: "Ask questions", desc: "Hold mic to ask anything" },
                  ].map((f, i) => (
                    <div key={i} style={{ padding: "12px 14px", borderRadius: 12, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", textAlign: "left" }}>
                      <div style={{ fontSize: 18, marginBottom: 4 }}>{f.icon}</div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginBottom: 2 }}>{f.label}</div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>{f.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {mode === "standard" && isAnalyzing && !explanation && (
              <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
                <div style={{ display: "flex", gap: 6 }}>
                  {[0,1,2,3].map(i => (
                    <div key={i} style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366f1", animation: `bounce 1s ease-in-out infinite`, animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
                <p style={{ fontSize: 14, color: "rgba(255,255,255,0.35)", margin: 0 }}>Gemini is reading your homework...</p>
              </div>
            )}

            {mode === "standard" && explanation && (
              <div>
                <p style={{ fontSize: 15, color: "rgba(255,255,255,0.82)", lineHeight: 1.8, margin: 0, fontWeight: 400 }}>
                  {explanation}
                  {isExplaining && <span style={{ display: "inline-block", width: 2, height: 16, background: "#6366f1", marginLeft: 2, verticalAlign: "middle", animation: "pulse 1s infinite" }} />}
                </p>
                {followUpQuestion && !isExplaining && (
                  <div style={{ marginTop: 20, padding: "14px 16px", borderRadius: 12, background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.2)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                      <Lightbulb size={12} color="#a78bfa" />
                      <span style={{ fontSize: 10, fontWeight: 700, color: "#a78bfa", textTransform: "uppercase", letterSpacing: "0.08em" }}>Check Your Understanding</span>
                    </div>
                    <p style={{ fontSize: 13, color: "rgba(255,255,255,0.6)", margin: 0, lineHeight: 1.6 }}>{followUpQuestion}</p>
                  </div>
                )}
              </div>
            )}

            {mode === "live" && !liveConnected && !liveStarting && (
              <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: 20 }}>
                <div style={{ width: 72, height: 72, borderRadius: 24, background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Radio size={32} color="#10b981" />
                </div>
                <div>
                  <h3 style={{ fontSize: 20, fontWeight: 700, color: "rgba(255,255,255,0.7)", margin: "0 0 8px", letterSpacing: "-0.02em" }}>Live Mode</h3>
                  <p style={{ fontSize: 13, color: "rgba(255,255,255,0.3)", maxWidth: 300, lineHeight: 1.6, margin: "0 auto" }}>Real-time voice + vision. Gemini hears and sees you simultaneously. Interrupt naturally.</p>
                </div>
              </div>
            )}

            {mode === "live" && liveStarting && (
              <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12 }}>
                <Loader2 size={28} style={{ animation: "spin 1s linear infinite", color: "#10b981" }} />
                <p style={{ fontSize: 14, color: "rgba(255,255,255,0.35)", margin: 0 }}>Connecting to Gemini Live...</p>
              </div>
            )}

            {mode === "live" && liveConnected && (
              <div style={{ height: "100%" }}>
                {liveTranscript ? (
                  <p style={{ fontSize: 15, color: "rgba(255,255,255,0.8)", lineHeight: 1.8, margin: 0 }}>{liveTranscript}</p>
                ) : (
                  <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
                    <div style={{ width: 80, height: 80, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", border: `2px solid ${liveGeminiSpeaking ? "#3b82f6" : isMicOn ? "#10b981" : "rgba(255,255,255,0.08)"}`, background: liveGeminiSpeaking ? "rgba(59,130,246,0.1)" : isMicOn ? "rgba(16,185,129,0.1)" : "rgba(255,255,255,0.03)", transition: "all 0.3s", animation: (liveGeminiSpeaking || isMicOn) ? "pulse 2s ease-in-out infinite" : "none" }}>
                      {liveGeminiSpeaking ? <Volume2 size={32} color="#60a5fa" style={{ animation: "pulse 1.5s infinite" }} /> :
                       isMicOn ? <Mic size={32} color="#10b981" style={{ animation: "pulse 1.5s infinite" }} /> :
                       <Radio size={32} color="rgba(255,255,255,0.15)" />}
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <p style={{ fontSize: 15, fontWeight: 600, color: "rgba(255,255,255,0.4)", margin: "0 0 4px" }}>
                        {liveGeminiSpeaking ? "Gemini is speaking..." : isMicOn ? "Listening to you..." : "Turn on Mic to start"}
                      </p>
                      <p style={{ fontSize: 12, color: "rgba(255,255,255,0.2)", margin: 0 }}>
                        {!isMicOn && !isCameraOn ? "Use buttons on the left" : isMicOn && !isCameraOn ? "Enable camera to show homework" : "Speak naturally — interrupt anytime"}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action bar */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0, padding: "0 4px" }}>
            {mode === "standard" && (
              <>
                <button onMouseDown={startListening} onMouseUp={stopListening} onTouchStart={startListening} onTouchEnd={stopListening} disabled={!speechSupported}
                  style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 10, fontSize: 12, fontWeight: 500, cursor: "pointer", border: isListening ? "1px solid rgba(239,68,68,0.4)" : "1px solid rgba(255,255,255,0.08)", background: isListening ? "rgba(239,68,68,0.1)" : "rgba(255,255,255,0.04)", color: isListening ? "#f87171" : "rgba(255,255,255,0.5)", userSelect: "none", opacity: !speechSupported ? 0.3 : 1 }}>
                  {isListening ? <><div style={{ width: 7, height: 7, borderRadius: "50%", background: "#ef4444", animation: "pulse 1s infinite" }} />Listening...</> : <><Mic size={12} />Hold to Speak</>}
                </button>

                <button onClick={() => { setIsMuted(!isMuted); if (isSpeaking) stopSpeaking(); }}
                  style={{ width: 34, height: 34, borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.35)" }}>
                  {isMuted ? <VolumeX size={13} /> : <Volume2 size={13} />}
                </button>

                <div style={{ flex: 1 }} />

                {problemInfo && !isExplaining && currentStep < totalSteps - 1 && (
                  <button onClick={requestNextStep}
                    style={{ display: "flex", alignItems: "center", gap: 5, padding: "8px 14px", borderRadius: 10, fontSize: 12, fontWeight: 600, cursor: "pointer", border: "1px solid rgba(99,102,241,0.3)", background: "rgba(99,102,241,0.1)", color: "#a5b4fc" }}>
                    Next Step <ChevronRight size={12} />
                  </button>
                )}

                {problemInfo && (
                  <button onClick={() => { requestDiagram(problemInfo.problem); setIsGenerating(true); }} disabled={isGenerating}
                    style={{ display: "flex", alignItems: "center", gap: 5, padding: "8px 14px", borderRadius: 10, fontSize: 12, fontWeight: 500, cursor: "pointer", border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.04)", color: "rgba(255,255,255,0.5)", opacity: isGenerating ? 0.4 : 1 }}>
                    {isGenerating ? <Loader2 size={11} style={{ animation: "spin 1s linear infinite" }} /> : <Sparkles size={11} />} Diagram
                  </button>
                )}

                {problemInfo && (
                  <button onClick={() => { requestPractice(); setIsGenerating(true); }} disabled={isGenerating}
                    style={{ display: "flex", alignItems: "center", gap: 5, padding: "8px 14px", borderRadius: 10, fontSize: 12, fontWeight: 600, cursor: "pointer", border: "1px solid rgba(16,185,129,0.25)", background: "rgba(16,185,129,0.08)", color: "#10b981", opacity: isGenerating ? 0.4 : 1 }}>
                    <BookOpen size={11} /> Practice
                  </button>
                )}
              </>
            )}

            {mode === "live" && liveConnected && (
              <>
                <div style={{ flex: 1 }} />
                <button onClick={() => setLiveTranscript("")}
                  style={{ padding: "7px 14px", borderRadius: 10, fontSize: 11, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", cursor: "pointer", color: "rgba(255,255,255,0.35)" }}>
                  Clear transcript
                </button>
              </>
            )}
          </div>
        </div>

        {/* RIGHT — Diagrams & Practice / Live tips */}
        <div style={{ width: 300, flexShrink: 0, display: "flex", flexDirection: "column", padding: "16px 16px 16px 12px", gap: 12, borderLeft: "1px solid rgba(255,255,255,0.05)", overflowY: "auto" }}>

          {mode === "live" ? (
            <div style={{ borderRadius: 14, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.12)", overflow: "hidden" }}>
              <div style={{ padding: "12px 14px", borderBottom: "1px solid rgba(16,185,129,0.1)", display: "flex", alignItems: "center", gap: 7 }}>
                <Radio size={13} color="#10b981" />
                <span style={{ fontSize: 11, fontWeight: 700, color: "#10b981", textTransform: "uppercase", letterSpacing: "0.06em" }}>How to use Live</span>
              </div>
              <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 14 }}>
                {[
                  ["Connect", "Click Go Live — session opens automatically"],
                  ["Turn on Mic", "Press Mic Off to start speaking to Gemini"],
                  ["Show homework", "Enable camera so Gemini sees your work"],
                  ["Just talk", "Ask questions, interrupt freely at any time"],
                ].map(([title, desc], i) => (
                  <div key={i} style={{ display: "flex", gap: 10 }}>
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.25)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <span style={{ fontSize: 10, fontWeight: 700, color: "#10b981" }}>{i+1}</span>
                    </div>
                    <div>
                      <p style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.55)", margin: "0 0 2px" }}>{title}</p>
                      <p style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", margin: 0, lineHeight: 1.5 }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <>
              {isGenerating && (
                <div style={{ borderRadius: 14, border: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.02)", padding: "28px 16px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
                  <Loader2 size={22} style={{ animation: "spin 1s linear infinite", color: "#6366f1" }} />
                  <p style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", margin: 0 }}>Generating with Gemini...</p>
                </div>
              )}

              {diagram && !isGenerating && (
                <div style={{ borderRadius: 14, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }} className="diagram-container">
                  <div style={{ padding: "8px 12px", background: "rgba(0,0,0,0.4)", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: 6 }}>
                    <Sparkles size={10} color="#818cf8" />
                    <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Visual Aid</span>
                    <span style={{ fontSize: 10, color: "rgba(255,255,255,0.2)", marginLeft: "auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 120 }}>{diagram.concept}</span>
                  </div>
                  <div style={{ background: "#fff", padding: 4 }} dangerouslySetInnerHTML={{ __html: diagram.svg }} />
                </div>
              )}

              {showPractice && practice && !isGenerating && (
                <div style={{ borderRadius: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", overflow: "hidden" }}>
                  <div style={{ padding: "8px 12px", background: "rgba(16,185,129,0.06)", borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", alignItems: "center", gap: 6 }}>
                    <CheckCircle2 size={10} color="#10b981" />
                    <span style={{ fontSize: 10, fontWeight: 700, color: "#10b981", textTransform: "uppercase", letterSpacing: "0.06em" }}>Practice</span>
                  </div>
                  <div style={{ padding: "14px" }}>
                    <p style={{ fontSize: 13, color: "rgba(255,255,255,0.75)", lineHeight: 1.6, margin: "0 0 12px" }}>{practice.question}</p>
                    {!showAnswer ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <button onClick={() => setShowAnswer(true)} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "rgba(255,255,255,0.25)", background: "none", border: "none", cursor: "pointer", padding: 0 }}>
                          <Lightbulb size={10} /> Show hint
                        </button>
                        <button onClick={() => setShowAnswer(true)} style={{ width: "100%", padding: "8px 0", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", color: "#10b981" }}>
                          Reveal Answer
                        </button>
                      </div>
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <div style={{ padding: "10px 12px", borderRadius: 8, background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.15)" }}>
                          <p style={{ fontSize: 9, fontWeight: 700, color: "#fbbf24", margin: "0 0 4px", textTransform: "uppercase" }}>Hint</p>
                          <p style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", margin: 0 }}>{practice.hint}</p>
                        </div>
                        <div style={{ padding: "10px 12px", borderRadius: 8, background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.15)" }}>
                          <p style={{ fontSize: 9, fontWeight: 700, color: "#10b981", margin: "0 0 4px", textTransform: "uppercase" }}>Answer</p>
                          <p style={{ fontSize: 12, color: "rgba(255,255,255,0.65)", margin: 0 }}>{practice.answer}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!diagram && !showPractice && !isGenerating && (
                <div style={{ borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.01)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, padding: "32px 20px", textAlign: "center", minHeight: 180 }}>
                  <Sparkles size={20} color="rgba(255,255,255,0.1)" />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 500, color: "rgba(255,255,255,0.2)", margin: "0 0 4px" }}>Diagrams & Practice</p>
                    <p style={{ fontSize: 11, color: "rgba(255,255,255,0.12)", margin: 0 }}>Generated visuals appear here</p>
                  </div>
                  {problemInfo && (
                    <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                      <button onClick={() => { requestDiagram(problemInfo.problem); setIsGenerating(true); }} style={{ padding: "6px 12px", borderRadius: 8, fontSize: 11, background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)", color: "#a5b4fc", cursor: "pointer" }}>
                        Generate diagram
                      </button>
                      <button onClick={() => { requestPractice(); setIsGenerating(true); }} style={{ padding: "6px 12px", borderRadius: 8, fontSize: 11, background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)", color: "#10b981", cursor: "pointer" }}>
                        Practice
                      </button>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </main>

      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        @keyframes scan { 0% { top: 0; opacity: 1; } 95% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
        .diagram-container svg { width: 100% !important; height: auto !important; display: block; }
      `}</style>
    </div>
  );
}