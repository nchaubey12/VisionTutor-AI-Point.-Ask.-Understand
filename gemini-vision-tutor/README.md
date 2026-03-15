# 🎓 Gemini Vision Tutor

> **Real-Time Multimodal AI Tutor powered by Google Gemini**
> Built for the Gemini Live Agent Challenge

A real-time AI tutoring system that **sees your homework** through the camera, **understands the problem** using Gemini Vision, **explains step-by-step** with voice, and lets you **interrupt naturally** to ask questions — all in a live, bidirectional conversation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 **Live Vision Analysis** | Points camera at homework → Gemini identifies subject, problem, errors |
| 🗣️ **Interruptible Voice Tutor** | Hold-to-speak mic button; interrupt mid-explanation naturally |
| 🧠 **Multi-Agent Pipeline** | Vision → Reasoning → Teaching → Dialogue agents work in sequence |
| 🎨 **AI-Generated Diagrams** | SVG diagrams generated on-demand for visual learners |
| 📝 **Practice Questions** | Auto-generates similar practice problems with hints & answers |
| 💾 **Conversation Memory** | Firestore persists session history for coherent multi-turn teaching |
| ⚡ **Streaming Responses** | Text streams token-by-token for responsive feel |
| 🔊 **Text-to-Speech** | Browser TTS reads explanations aloud; mutable |

---

## 🏗️ Architecture

```
┌─────────────┐   WebSocket   ┌──────────────────────────┐   REST/Stream   ┌─────────────────┐
│  User Device│ ◄────────────► │  FastAPI Backend          │ ◄─────────────► │  Google Gemini  │
│  (Browser)  │               │                           │                 │                 │
│  - Camera   │               │  Agent Pipeline:           │                 │  - 1.5 Pro      │
│  - Mic      │               │  1. VisionAgent            │                 │    (vision)     │
│  - Speaker  │               │  2. ReasoningAgent         │                 │  - 1.5 Flash    │
│             │               │  3. TeachingAgent          │                 │    (dialogue)   │
│  Next.js    │               │  4. DialogueAgent          │                 └─────────────────┘
│  + WebRTC   │               │                           │
└─────────────┘               │  Services:                 │   ┌──────────────────────────────┐
                               │  - GeminiService          │   │  Google Cloud                │
                               │  - FirestoreService       ├──►│  - Cloud Run (hosting)       │
                               │  - StorageService         │   │  - Firestore (sessions)      │
                               └──────────────────────────┘   │  - Cloud Storage (files)     │
                                                               │  - Secret Manager (keys)     │
                                                               └──────────────────────────────┘
```

See [`architecture-diagram.svg`](./architecture-diagram.svg) for the full visual diagram.

### Agent Pipeline

```
Camera Frame
     │
     ▼
[VisionAgent]          ← Gemini 1.5 Pro Vision
  Extracts: subject, problem, errors, difficulty
     │
     ▼
[ReasoningAgent]       ← Gemini 1.5 Pro
  Creates: step-by-step teaching plan
     │
     ▼
[TeachingAgent]        ← Gemini 1.5 Pro
  Generates: explanation text + SVG diagrams
     │
     ▼
[DialogueAgent]        ← Gemini 1.5 Flash
  Handles: interruptions, questions, conversation
     │
     ▼
WebSocket → Browser (text chunks + diagrams + TTS)
```

---

## 📁 Project Structure

```
gemini-vision-tutor/
├── frontend/
│   ├── pages/
│   │   ├── index.tsx          # Main tutor UI
│   │   └── _app.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts    # WS connection + message routing
│   │   ├── useWebcam.ts       # Camera access + frame capture
│   │   └── useSpeech.ts       # Speech recognition + TTS
│   ├── styles/globals.css
│   ├── Dockerfile
│   ├── next.config.js
│   └── package.json
│
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── agents/
│   │   ├── vision_agent.py    # Frame analysis
│   │   ├── reasoning_agent.py # Problem solving + teaching plans
│   │   ├── teaching_agent.py  # Explanation + diagram generation
│   │   └── dialogue_agent.py  # Conversation + interruptions
│   ├── services/
│   │   ├── gemini_service.py  # Gemini API integration
│   │   ├── firestore_service.py
│   │   └── storage_service.py
│   ├── api/
│   │   ├── websocket.py       # /ws/tutor endpoint
│   │   └── tutor_routes.py    # REST endpoints
│   ├── Dockerfile
│   └── requirements.txt
│
├── infrastructure/
│   ├── cloud_run_deploy.sh    # One-command GCP deployment
│   └── terraform/
│       ├── main.tf            # All GCP resources
│       └── terraform.tfvars.example
│
├── docker-compose.yml         # Local dev stack
├── .env.example
├── architecture-diagram.svg
└── README.md
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- **Node.js** 20+
- **Python** 3.11+
- **Docker** + **Docker Compose** (optional, recommended)
- **Gemini API Key** — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Option A — Docker Compose (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/gemini-vision-tutor
cd gemini-vision-tutor

# 2. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 3. Start everything
docker-compose up --build

# Open http://localhost:3000
```

### Option B — Manual Setup

**Backend:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GEMINI_API_KEY=your_api_key_here

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Set WebSocket URL
echo "NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws/tutor" > .env.local

# Start dev server
npm run dev

# Open http://localhost:3000
```

---

## ☁️ Google Cloud Deployment

### Option A — Deployment Script (Recommended)

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud auth configure-docker

# 2. Set your project
export GOOGLE_CLOUD_PROJECT=your-project-id

# 3. Store your API key in Secret Manager
echo -n "your-gemini-api-key" | \
  gcloud secrets create gemini-api-key --data-file=- \
  --project=$GOOGLE_CLOUD_PROJECT

# 4. Run the deployment script
chmod +x infrastructure/cloud_run_deploy.sh
bash infrastructure/cloud_run_deploy.sh
```

### Option B — Terraform

```bash
cd infrastructure/terraform

# Copy and configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id, region, gemini_api_key

# Initialize and apply
terraform init
terraform plan
terraform apply

# Deploy containers (after Terraform creates infrastructure)
bash ../cloud_run_deploy.sh
```

### Manual Docker Build + Deploy

```bash
# Backend
docker build --platform linux/amd64 -t gcr.io/YOUR_PROJECT/tutor-backend ./backend
docker push gcr.io/YOUR_PROJECT/tutor-backend
gcloud run deploy tutor-backend \
  --image gcr.io/YOUR_PROJECT/tutor-backend \
  --platform managed --region us-central1 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT

# Frontend
BACKEND_WS_URL=wss://your-backend-url/ws/tutor
docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_WS_URL=$BACKEND_WS_URL \
  -t gcr.io/YOUR_PROJECT/tutor-frontend ./frontend
docker push gcr.io/YOUR_PROJECT/tutor-frontend
gcloud run deploy tutor-frontend \
  --image gcr.io/YOUR_PROJECT/tutor-frontend \
  --platform managed --region us-central1 \
  --allow-unauthenticated
```

---

## 🎬 Demo Instructions (4-minute walkthrough)

1. **Open the app** — browser requests camera + mic permissions, click Allow
2. **Show homework** — hold a math problem, science question, or any homework in front of the camera
3. **Click "Analyze Homework"** — Gemini Vision identifies the subject and problem (shown in left panel)
4. **Watch the explanation stream** — step-by-step explanation appears in real-time
5. **Interrupt with voice** — hold the microphone button and ask "Wait, why did you do that?"
6. **Request a diagram** — click "Diagram" to generate a visual aid for the concept
7. **Get practice** — click "Practice" to receive a similar question to test understanding
8. **Continue** — click "Next Step" to proceed through the teaching plan

---

## 📡 WebSocket API Reference

### Client → Server

```jsonc
// Send a camera frame for analysis
{ "type": "frame", "image": "<base64-jpeg>", "force_reanalyze": false }

// Send voice/text input (supports interruptions)
{ "type": "voice_input", "text": "Why did you divide by 2?" }

// Request a visual diagram
{ "type": "request_diagram", "concept": "quadratic formula" }

// Request a practice question
{ "type": "request_practice" }

// Advance to next step
{ "type": "next_step" }

// Reset session
{ "type": "new_session" }
```

### Server → Client

```jsonc
// Connection confirmed
{ "type": "connected", "session_id": "uuid", "message": "..." }

// Frame analyzed — problem detected
{ "type": "frame_analyzed", "subject": "math", "problem": "...", "difficulty": "high", "total_steps": 3 }

// Explanation step starting
{ "type": "explanation_start", "step": 0, "step_title": "Understanding the Problem", "total_steps": 3 }

// Streaming text chunk
{ "type": "text_chunk", "text": "First, let's identify..." }

// Explanation finished
{ "type": "explanation_complete", "full_text": "...", "step": 0, "follow_up": "Does that make sense?" }

// SVG diagram generated
{ "type": "diagram", "svg": "<svg>...</svg>", "concept": "..." }

// Practice question
{ "type": "practice_question", "question": "...", "hint": "...", "answer": "..." }

// Interruption acknowledged
{ "type": "interrupted" }
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `GOOGLE_CLOUD_PROJECT` | Optional | — | GCP project ID (enables Firestore/GCS) |
| `FIRESTORE_COLLECTION` | Optional | `tutor_sessions` | Firestore collection name |
| `GCS_BUCKET_NAME` | Optional | — | Cloud Storage bucket |
| `PORT` | Optional | `8080` | Backend server port |
| `ENV` | Optional | `production` | `development` enables hot reload |
| `NEXT_PUBLIC_WS_URL` | ✅ Frontend | — | WebSocket URL for frontend |

---

## 🛠️ Technology Stack

**Frontend**
- Next.js 14 (Pages Router)
- TypeScript
- TailwindCSS
- WebRTC (camera/mic access)
- Web Speech API (STT + TTS)
- Native WebSocket

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- google-generativeai SDK
- WebSockets (via FastAPI)
- Pydantic v2

**AI / ML**
- Gemini 1.5 Pro (vision, reasoning, diagram generation)
- Gemini 1.5 Flash (fast dialogue, interruptions)
- Streaming via AsyncIterator

**Google Cloud**
- Cloud Run (serverless containers)
- Firestore (conversation storage)
- Cloud Storage (media files)
- Secret Manager (API keys)
- Vertex AI / Google AI Studio

**Infrastructure**
- Docker + Docker Compose
- Terraform
- gcloud CLI scripts

---

## 🏆 Hackathon Compliance

| Requirement | Implementation |
|---|---|
| ✅ Gemini model usage | Gemini 1.5 Pro + Flash via google-generativeai SDK |
| ✅ Gemini Live API | Real-time streaming via AsyncIterator WebSocket pipeline |
| ✅ Google Cloud hosting | Cloud Run (both frontend + backend) |
| ✅ Multimodal input | Camera frames (vision) + voice input (speech) + text |
| ✅ Multimodal output | Text + TTS voice + SVG diagrams |
| ✅ Agent architecture | 4 specialized agents (Vision, Reasoning, Teaching, Dialogue) |

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

*Built with creativity, intelligent agents, and real-time AI for the Gemini Live Agent Challenge.*
