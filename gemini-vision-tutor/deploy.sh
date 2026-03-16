# ============================================================
# VisionTutor AI — Cloud Build + Cloud Run Deployment Config
# ============================================================
# Triggered automatically on git push, or run manually:
#   gcloud builds submit --config cloudbuild.yaml .
#
# Required Secret Manager secret:
#   gcloud secrets create GEMINI_API_KEY --data-file=- <<< "your_key"
#   gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
#     --member="serviceAccount:YOUR_PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
#     --role="roles/secretmanager.secretAccessor"
# ============================================================

steps:
  # ── Step 1: Build Docker image ─────────────────────────────
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/visiontutor-backend:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/visiontutor-backend:latest'
      - './backend'

  # ── Step 2: Push to Container Registry ────────────────────
  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-sha'
    args: ['push', 'gcr.io/$PROJECT_ID/visiontutor-backend:$COMMIT_SHA']
    waitFor: ['build']

  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-latest'
    args: ['push', 'gcr.io/$PROJECT_ID/visiontutor-backend:latest']
    waitFor: ['build']

  # ── Step 3: Deploy to Cloud Run ───────────────────────────
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    id: 'deploy'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'visiontutor-backend'
      - '--image'
      - 'gcr.io/$PROJECT_ID/visiontutor-backend:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--port'
      - '8080'
      - '--timeout'
      - '3600'
      - '--min-instances'
      - '1'
      - '--max-instances'
      - '10'
      - '--memory'
      - '512Mi'
      - '--cpu'
      - '1'
      - '--update-secrets'
      - 'GEMINI_API_KEY=GEMINI_API_KEY:latest'
    waitFor: ['push-sha']

images:
  - 'gcr.io/$PROJECT_ID/visiontutor-backend:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/visiontutor-backend:latest'

timeout: '1200s'

options:
  logging: CLOUD_LOGGING_ONLY