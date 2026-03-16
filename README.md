# 🎓 VisionTutor AI — Point. Ask. Understand.

> **Real-time AI tutoring powered by webcam vision and natural voice. No typing. No uploading. Just point your camera and learn.**

Built for the **Gemini Live Agent Challenge** — a multimodal live agent that turns any webcam into an intelligent homework tutor. Students point their camera at any problem, speak their question naturally, and receive spoken step-by-step explanations with generated visual aids — with full interruption support, just like a real tutor.

## 🎥 Demo: [VisionTutor AI — Live Demo](https://youtu.be/jc_m82hKUJI)

[![Live Agents Category](https://img.shields.io/badge/Category-Live%20Agents%20🗣️-4285F4?style=flat&logo=google)](https://github.com/nchaubey12/VisionTutor-AI-Point.-Ask.-Understand)
[![Gemini 2.5 Flash](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat&logo=google)](https://ai.google.dev/)
[![Gemini Live](https://img.shields.io/badge/Gemini-Live%20API-34A853?style=flat&logo=google)](https://ai.google.dev/)
[![Google Cloud](https://img.shields.io/badge/Google-Cloud%20Run-FF6F00?style=flat&logo=googlecloud)](https://cloud.google.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?style=flat&logo=firebase)](https://firebase.google.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-TypeScript-000000?style=flat&logo=nextdotjs)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📸 Screenshots

### Main Tutoring Interface — Ready to Tutor
<img src="gemini-vision-tutor/docs/screenshots/screenshot_02_tutor_frame_detection.png" width="900"/>

### Gemini Analyzing Homework — Frame Detection
<img src="gemini-vision-tutor/docs/screenshots/screenshot_02_tutor_interface.png" width="900"/>

### Step-by-step Explanation with Visual Aid
<img src="gemini-vision-tutor/docs/screenshots/screenshot_03_explanation.png" width="900"/>

### Live Agent Mode — Real-time Voice Interaction
<img src="gemini-vision-tutor/docs/screenshots/Screenshot_04_Live_Agent_Interaction.png" width="900"/>

### Live Agent Dashboard — Ready to Speak
<img src="gemini-vision-tutor/docs/screenshots/Screenshot_04_live_agent_dashbord_to_speak.png" width="900"/>

---

## 🗺 System Diagrams

### 1. Workflow Overview
> End-to-end tutoring pipeline — from student pointing their webcam to receiving a step-by-step explanation with visual aid, using Gemini's multimodal capabilities.

<img src="gemini-vision-tutor/docs/documents/01_Workflow_Overview.png" width="700"/>

---

### 2. System Architecture
> Full system architecture — Next.js frontend streams webcam frames to a FastAPI backend on Cloud Run, which orchestrates multiple AI agents backed by Gemini and returns explanations in real-time.

<img src="gemini-vision-tutor/docs/documents/02_System_Architecture.png" width="800"/>

---

### 3. LLM Orchestration & Multimodal Fusion
> How audio and video streams are fused — vision_agent reads the homework frame, teaching_agent generates the explanation, dialogue_agent manages the conversation turn, and reasoning_agent produces the step-by-step breakdown.

<img src="gemini-vision-tutor/docs/documents/03_LLM_Orchestration.png" width="800"/>

---

### 4. Deployment Pipeline
> From source code to Google Cloud — Terraform provisions Cloud Run, the deploy script builds and pushes the Docker image, and docker-compose enables local multi-service development.

<img src="gemini-vision-tutor/docs/documents/04_Deployment_Pipeline.png" width="700"/>

---

### 5. Firebase Data Flow
> How Firestore stores session data and how the backend services interact with Firebase during a tutoring session.

<img src="gemini-vision-tutor/docs/documents/05_Firebase_Flow.png" width="700"/>

---

## 🧠 The Problem

Every night, millions of students sit alone with homework they don't understand. Existing AI tools require typing — slow, frustrating, and completely disconnected from how students actually work.

- You **can't type** a complex geometry diagram
- You **can't describe** a messy handwritten equation fast enough
- You **can't interrupt** a chatbot mid-sentence to ask a follow-up
- Traditional tutors are **expensive** and unavailable at 11 PM

There is no tool that lets a student simply *point at their problem and talk.*

---

## 💡 The Solution

**No typing. No uploading. No waiting.**

VisionTutor AI turns any webcam into a real-time homework tutor. Point your camera at any problem — handwritten equations, diagrams, printed text — click **Analyze Homework** and the AI instantly reads it, identifies the subject, and walks you through a full step-by-step explanation with generated visual aids.

Switch to **Live Mode** for true conversational tutoring powered by Gemini Live API — speak naturally, interrupt freely, and get spoken responses in real-time while Gemini watches your homework through your camera.

- 📸 **Point** — camera detects your homework instantly (math, science, English and more)
- 🧠 **Analyze** — Gemini 2.5 Flash reads the problem, detects subject and difficulty level
- 📋 **Understand** — step-by-step explanation with generated visual diagrams on the right panel
- ❓ **Practice** — "Check Your Understanding" follow-up question after every explanation
- 🎙️ **Go Live** — switch to Gemini Live for real-time voice tutoring with full barge-in support

---

## ✅ Hackathon Checklist

- [x] Multimodal input — live webcam vision + microphone audio
- [x] Gemini 2.5 Flash — homework frame analysis and step-by-step explanation
- [x] Gemini Live API — real-time bidirectional voice mode with barge-in support
- [x] Multi-agent pipeline — vision, dialogue, reasoning, and teaching agents
- [x] Google GenAI SDK — Python backend orchestration
- [x] Visual Aid generation — diagrams generated alongside explanations
- [x] "Check Your Understanding" — follow-up questions after each explanation
- [x] Subject detection — auto-detects math, science, grammar and more
- [x] Next Step / Diagram / Practice controls in the UI
- [x] Google Cloud hosting — FastAPI backend deployed on Cloud Run
- [x] Firebase Firestore — session storage
- [x] Next.js + TypeScript frontend — webcam and audio capture in browser
- [x] Docker containerised backend + docker-compose for local dev
- [x] Automated deployment — Terraform IaC + cloud_run_deploy.sh (bonus)
- [x] Multi-subject support — math, science, English, and more

---

## 📁 Project Structure

```
VisionTutor-AI-Point.-Ask.-Understand/
├── gemini-vision-tutor/
│   │
│   ├── backend/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── dialogue_agent.py        ← Manages conversation turns
│   │   │   ├── reasoning_agent.py       ← Step-by-step reasoning logic
│   │   │   ├── teaching_agent.py        ← Tutoring response generation
│   │   │   └── vision_agent.py          ← Webcam frame analysis
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── live_agent.py            ← Gemini Live API session handler
│   │   │   ├── tutor_routes.py          ← FastAPI REST route definitions
│   │   │   └── websocket.py             ← WebSocket endpoint + stream manager
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── firestore_service.py     ← Firestore read/write helpers
│   │   │   ├── gemini_service.py        ← Gemini SDK wrapper
│   │   │   └── storage_service.py       ← File/media storage helpers
│   │   ├── tests/
│   │   │   └── test_gemini_service.py   ← Unit tests
│   │   ├── .env                         ← Local environment variables
│   │   ├── .env.example                 ← Environment variable template
│   │   ├── Dockerfile                   ← Container image for Cloud Run
│   │   └── main.py                      ← FastAPI app entry point
│   │
│   ├── frontend/
│   │   ├── hooks/
│   │   │   ├── useLiveAgent.ts          ← Gemini Live session hook
│   │   │   ├── useSpeech.ts             ← Speech input/output hook
│   │   │   ├── useWebcam.ts             ← Webcam capture hook
│   │   │   └── useWebSocket.ts          ← WebSocket connection hook
│   │   ├── pages/
│   │   │   ├── _app.tsx                 ← Next.js app wrapper
│   │   │   └── index.tsx                ← Main tutoring interface (single page)
│   │   ├── public/
│   │   │   └── audio-processor.js       ← AudioWorklet PCM processor
│   │   ├── styles/
│   │   │   └── globals.css              ← Global styles
│   │   ├── .env.local                   ← Frontend environment variables
│   │   ├── Dockerfile                   ← Frontend container
│   │   ├── next-env.d.ts
│   │   ├── next.config.js
│   │   ├── package.json
│   │   ├── postcss.config.js
│   │   ├── tailwind.config.js
│   │   └── tsconfig.json
│   │
│   └── infrastructure/
│       ├── terraform/
│       │   └── main.tf                  ← Cloud Run, IAM, Artifact Registry
│       └── cloud_run_deploy.sh          ← One-command Cloud Run deploy script
│
├── .env
├── .env.example
├── .gitignore
├── architecture-diagram.png             ← System architecture diagram
├── docker-compose.yml                   ← Local multi-service dev setup
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Docker
- Google Cloud CLI (`gcloud`)
- A Google Cloud project with Gemini API and Cloud Run enabled

### 1. Clone the Repository

```bash
git clone https://github.com/nchaubey12/VisionTutor-AI-Point.-Ask.-Understand.git
cd VisionTutor-AI-Point.-Ask.-Understand
```

### 2. Configure Keys

Edit `gemini-vision-tutor/backend/.env` (copy from `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk@your-project.iam.gserviceaccount.com
PORT=8000
ALLOWED_ORIGINS=http://localhost:3000
```

Edit `gemini-vision-tutor/frontend/.env.local` (copy from `.env.example`):

```env
NEXT_PUBLIC_BACKEND_WS_URL=ws://localhost:8000
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_firebase_project_id
```

### 3. Run with Docker Compose (Recommended)

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

### 4. Run Manually

**Backend:**
```bash
cd gemini-vision-tutor/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd gemini-vision-tutor/frontend
npm install
npm run dev
# App: http://localhost:3000
```

### 5. Deploy to Google Cloud

```bash
cd gemini-vision-tutor/infrastructure
chmod +x cloud_run_deploy.sh
./cloud_run_deploy.sh
```

Or manually:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/visiontutor-backend ./gemini-vision-tutor/backend

gcloud run deploy visiontutor-backend \
  --image gcr.io/YOUR_PROJECT_ID/visiontutor-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key,FIREBASE_PROJECT_ID=your_project
```

---
## 🧪 Reproducible Testing

Once the app is running at `http://localhost:3000`, follow these steps to verify every feature works correctly.

---

### ✅ 1. Health Check (Backend)
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{ "status": "ok" }
```

---

### ✅ 2. Test Standard Mode (Vision + Explanation)

1. Open `http://localhost:3000` in Chrome or Firefox
2. Allow **webcam** and **microphone** permissions when prompted
3. Point your camera at any homework problem (handwritten equation, printed text, or diagram)
4. Click **"Analyze Homework"**

Expected results:
- Subject is detected (e.g. `math`, `science`, `grammar`)
- Step-by-step explanation appears in the left panel
- SVG visual aid diagram renders in the right panel
- "Check Your Understanding" follow-up question appears below the explanation

---

### ✅ 3. Test Follow-up Voice Question (Standard Mode)

1. After an explanation loads, unmute your microphone
2. Ask a follow-up question verbally (e.g. *"Can you explain step 2 again?"*)

Expected: Gemini responds with a spoken clarification without re-analyzing the frame.

---

### ✅ 4. Test Live Mode (Gemini Live API)

1. Click the **"Go Live"** button
2. Enable your camera and unmute your microphone
3. Speak naturally — e.g. *"Can you explain this equation to me?"*

Expected results:
- Gemini responds in voice within **~1 second**
- Status indicator shows `listening` → `speaking` transitions
- **Barge-in:** interrupt Gemini mid-sentence with a new question — it stops and responds to the new question immediately

---

### ✅ 5. Run Backend Unit Tests
```bash
cd gemini-vision-tutor/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

Expected: all tests in `test_gemini_service.py` pass.

---

### ✅ 6. Test Without Firebase (Offline Fallback)

Run the backend without any Firebase credentials in `.env`. The app should still start and all tutoring features should work using the **in-memory fallback** — no crash, no hang.

---

### 🔍 Quick Feature Verification Table

| Feature | How to test | Expected result |
|---|---|---|
| Webcam frame analysis | Point camera → Analyze | Subject + problem detected |
| Step-by-step explanation | Click Analyze | Steps appear with titles |
| SVG visual aid | Click Analyze | Diagram renders in right panel |
| Practice question | Complete an explanation | "Check Your Understanding" appears |
| Live voice mode | Go Live → speak | Voice response in <1s |
| Barge-in / interrupt | Speak while Gemini talks | Gemini stops, responds to new question |
| Health endpoint | `curl /health` | `{"status": "ok"}` |
| Firebase fallback | Remove Firebase env vars | App still runs fully |

## 📡 API Reference

### WebSocket — `/ws/tutor` — Tutoring Session

**Video frame message** (JSON):
```json
{
  "type": "video_frame",
  "data": "<base64 JPEG>",
  "mime_type": "image/jpeg"
}
```

**Audio message** (binary frame):
```
PCM 16-bit, 16kHz mono — raw audio chunk bytes
```

**Interrupt message** (JSON):
```json
{ "type": "interrupt" }
```
---

### REST Endpoints

```
GET    /health                         ← health check
POST   /api/analyze                    ← analyze a captured homework frame
GET    /api/sessions                   ← list past sessions
GET    /api/sessions/{session_id}      ← full session transcript
```

---

## 📤 Example Tutor Response (Firestore)

```json
{
  "session_id": "sess_A3F19C",
  "subject_detected": "math",
  "difficulty": "middle",
  "problem": "2x + 6 = 14",
  "steps": [
    {
      "step": 1,
      "title": "Isolate the variable term",
      "explanation": "Subtract 6 from both sides: 2x + 6 - 6 = 14 - 6 → 2x = 8",
      "visual_aid": true
    },
    {
      "step": 2,
      "title": "Solve for x",
      "explanation": "Divide both sides by 2: 2x ÷ 2 = 8 ÷ 2 → x = 4",
      "visual_aid": true
    }
  ],
  "check_your_understanding": "How would you solve 2x - 6 = 14?",
  "answer": "x = 4"
}
```

---

## ☁️ Google Cloud Services Used

| Service | Usage |
|---|---|
| **Gemini 2.5 Flash** | Homework frame analysis, step-by-step explanation, visual aid generation |
| **Gemini Live API** | Real-time bidirectional voice mode — barge-in, interrupt, natural conversation |
| **Google GenAI SDK** | Python backend orchestration across all agents |
| **Cloud Run** | Serverless container hosting for the FastAPI backend |
| **Firebase Firestore** | Session storage, transcripts, subject detection results |
| **Google Artifact Registry** | Docker image storage for Cloud Run deployments |
| **Terraform IaC** | Cloud Run, IAM, and registry provisioning |

---

## 🏗️ How It Works — Agent Pipeline

```
Student points webcam at homework
              │
              ▼
useWebcam.ts captures JPEG frame
              │
              ▼
vision_agent.py  ──── Reads the frame, detects subject + problem
              │
              ▼
teaching_agent.py ─── Generates step-by-step explanation
reasoning_agent.py ── Structures the logical steps
dialogue_agent.py ─── Manages the conversation turn
              │
              ▼
Response returned to frontend:
  ├── Step title + explanation text displayed
  ├── Visual Aid diagram generated (right panel)
  ├── "Check Your Understanding" follow-up question
  └── Next Step / Diagram / Practice controls enabled

─────── OR switch to LIVE MODE ───────

useLiveAgent.ts opens Gemini Live session
  ├── Unmute mic → speak naturally
  ├── Enable camera → Gemini sees your homework
  ├── Interrupt at any time (barge-in)
  └── Real-time voice response streamed back
```

---

## 🔑 Keywords

`Gemini 2.5 Flash` · `Gemini Live API` · `Real-time AI tutoring` · `Multimodal AI` · `Multi-agent pipeline` · `WebSockets` · `FastAPI` · `Next.js` · `Firebase Firestore` · `Google Cloud Run` · `Vision AI` · `EdTech` · `Live Agents` · `Barge-in support` · `Terraform IaC` · `Responsible AI` · `Smart education`

---

## 👥 Team

Built with curiosity, caffeine, and a belief that every student deserves a patient tutor — for the **Gemini Live Agent Challenge**.

> *The most powerful thing about VisionTutor is not just the speed — it's that Gemini sees exactly what the student sees. When a student points at "this part" of a handwritten equation, Gemini knows exactly what they mean, generates a step-by-step explanation referencing that specific problem, and draws a visual aid — all without the student typing a single word.*
