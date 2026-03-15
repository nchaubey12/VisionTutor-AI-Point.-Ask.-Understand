#!/bin/bash
# =============================================================================
# Gemini Vision Tutor - Google Cloud Run Deployment Script
# =============================================================================
# Usage: bash cloud_run_deploy.sh
# Prerequisites: gcloud CLI installed and authenticated
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${REGION:-us-central1}"
BACKEND_SERVICE="gemini-vision-tutor-backend"
FRONTEND_SERVICE="gemini-vision-tutor-frontend"
ARTIFACT_REGISTRY="gcr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────────
log_info "Starting deployment to Google Cloud Run..."
log_info "Project: ${PROJECT_ID} | Region: ${REGION}"

command -v gcloud >/dev/null 2>&1 || log_error "gcloud CLI not found. Install: https://cloud.google.com/sdk"
command -v docker >/dev/null 2>&1 || log_error "Docker not found. Install: https://docs.docker.com/get-docker/"

# Verify project is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
  log_error "Set GOOGLE_CLOUD_PROJECT env var: export GOOGLE_CLOUD_PROJECT=your-actual-project-id"
fi

# ── Enable required APIs ──────────────────────────────────────────────────────
log_info "Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

log_success "APIs enabled"

# ── Configure Docker for GCR ──────────────────────────────────────────────────
log_info "Configuring Docker authentication..."
gcloud auth configure-docker --quiet
log_success "Docker configured"

# ── Build & Deploy Backend ────────────────────────────────────────────────────
BACKEND_IMAGE="${ARTIFACT_REGISTRY}/${PROJECT_ID}/${BACKEND_SERVICE}:latest"

log_info "Building backend Docker image..."
cd "$(dirname "$0")/../backend"

docker build \
  --platform linux/amd64 \
  -t "${BACKEND_IMAGE}" \
  --label "deploy-time=$(date -u +%Y%m%dT%H%M%SZ)" \
  .

log_success "Backend image built: ${BACKEND_IMAGE}"

log_info "Pushing backend image to Container Registry..."
docker push "${BACKEND_IMAGE}"
log_success "Backend image pushed"

log_info "Deploying backend to Cloud Run..."
gcloud run deploy "${BACKEND_SERVICE}" \
  --image="${BACKEND_IMAGE}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=80 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=300 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},FIRESTORE_COLLECTION=tutor_sessions,ENV=production" \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
  --quiet

# Get the backend URL
BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

log_success "Backend deployed at: ${BACKEND_URL}"

# Convert https:// to wss:// for WebSocket URL
BACKEND_WS_URL="${BACKEND_URL/https:\/\//wss://}/ws/tutor"
log_info "WebSocket URL: ${BACKEND_WS_URL}"

# ── Build & Deploy Frontend ───────────────────────────────────────────────────
FRONTEND_IMAGE="${ARTIFACT_REGISTRY}/${PROJECT_ID}/${FRONTEND_SERVICE}:latest"

log_info "Building frontend Docker image..."
cd "$(dirname "$0")/../frontend"

docker build \
  --platform linux/amd64 \
  -t "${FRONTEND_IMAGE}" \
  --build-arg "NEXT_PUBLIC_WS_URL=${BACKEND_WS_URL}" \
  --label "deploy-time=$(date -u +%Y%m%dT%H%M%SZ)" \
  .

log_success "Frontend image built"

log_info "Pushing frontend image..."
docker push "${FRONTEND_IMAGE}"
log_success "Frontend image pushed"

log_info "Deploying frontend to Cloud Run..."
gcloud run deploy "${FRONTEND_SERVICE}" \
  --image="${FRONTEND_IMAGE}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --concurrency=100 \
  --min-instances=0 \
  --max-instances=5 \
  --timeout=60 \
  --set-env-vars="NEXT_PUBLIC_WS_URL=${BACKEND_WS_URL}" \
  --quiet

FRONTEND_URL=$(gcloud run services describe "${FRONTEND_SERVICE}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

log_success "Frontend deployed at: ${FRONTEND_URL}"

# ── Setup Firestore ───────────────────────────────────────────────────────────
log_info "Ensuring Firestore database exists..."
gcloud firestore databases create \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet 2>/dev/null || log_warn "Firestore already exists (this is fine)"

log_success "Firestore ready"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  🌐 Frontend:   ${FRONTEND_URL}"
echo -e "  🔧 Backend:    ${BACKEND_URL}"
echo -e "  🔌 WebSocket:  ${BACKEND_WS_URL}"
echo ""
echo -e "  Next steps:"
echo -e "  1. Open the frontend URL in your browser"
echo -e "  2. Allow camera and microphone access"
echo -e "  3. Point camera at homework and click Analyze!"
echo ""
