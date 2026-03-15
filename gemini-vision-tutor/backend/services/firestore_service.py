"""
Firestore Service - Conversation history and session storage
Persists tutoring sessions to Google Cloud Firestore
"""

import os
import logging
from datetime import datetime
from typing import Optional
from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Manages persistent conversation storage in Google Firestore.
    Falls back to in-memory storage if Firestore is unavailable (local dev).
    """

    def __init__(self):
        self.collection_name = os.getenv("FIRESTORE_COLLECTION", "tutor_sessions")
        self.db = None
        self._memory_store = {}  # Fallback for local development

        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if project_id:
                self.db = firestore.AsyncClient(project=project_id)
                logger.info(f"✅ Firestore connected to project: {project_id}")
            else:
                logger.warning("⚠️  GOOGLE_CLOUD_PROJECT not set - using in-memory storage")
        except (DefaultCredentialsError, Exception) as e:
            logger.warning(f"⚠️  Firestore unavailable ({e}) - using in-memory storage")

    async def create_session(self, session_id: str, metadata: dict = None) -> dict:
        """Create a new tutoring session."""
        session_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "conversation": [],
            "problem_info": {},
            "metadata": metadata or {}
        }

        if self.db:
            doc_ref = self.db.collection(self.collection_name).document(session_id)
            await doc_ref.set(session_data)
        else:
            self._memory_store[session_id] = session_data

        return session_data

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve an existing session."""
        if self.db:
            doc_ref = self.db.collection(self.collection_name).document(session_id)
            doc = await doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        else:
            return self._memory_store.get(session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None
    ) -> bool:
        """Add a message to the conversation history."""
        message = {
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        if self.db:
            doc_ref = self.db.collection(self.collection_name).document(session_id)
            await doc_ref.update({
                "conversation": firestore.ArrayUnion([message]),
                "updated_at": datetime.utcnow().isoformat()
            })
        else:
            if session_id not in self._memory_store:
                await self.create_session(session_id)
            self._memory_store[session_id]["conversation"].append(message)
            self._memory_store[session_id]["updated_at"] = datetime.utcnow().isoformat()

        return True

    async def update_problem_info(self, session_id: str, problem_info: dict) -> bool:
        """Update the current problem being worked on."""
        if self.db:
            doc_ref = self.db.collection(self.collection_name).document(session_id)
            await doc_ref.update({
                "problem_info": problem_info,
                "updated_at": datetime.utcnow().isoformat()
            })
        else:
            if session_id in self._memory_store:
                self._memory_store[session_id]["problem_info"] = problem_info

        return True

    async def get_conversation_history(self, session_id: str) -> list:
        """Get the full conversation history for a session."""
        session = await self.get_session(session_id)
        if session:
            return session.get("conversation", [])
        return []

    async def get_problem_info(self, session_id: str) -> dict:
        """Get the current problem info for a session."""
        session = await self.get_session(session_id)
        if session:
            return session.get("problem_info", {})
        return {}
