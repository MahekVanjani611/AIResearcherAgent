#!/usr/bin/env python3
"""
Chat Management System

Integrates with AuthManager and SessionManager to:
1. Create new chats per user
2. Save chat history (like ChatGPT)
3. Load previous chats
4. Store chat metadata
5. Manage multiple chats per user
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

from rag_module import RAGManager, SessionManager

CHATS_DIR = Path("chats_storage")
CHATS_DIR.mkdir(exist_ok=True)


class ChatManager:
    """Manages user chats and conversation history"""
    
    def __init__(self, username: str):
        self.username = username
        self.user_chats_dir = CHATS_DIR / username
        self.user_chats_dir.mkdir(exist_ok=True)
        self.chats_index_file = self.user_chats_dir / "chats_index.json"
        self._load_chats_index()
        
        # Initialize RAG manager for memory
        self.rag_manager = RAGManager(user_id=username)
        self.session_manager = SessionManager(user_id=username)
    
    def _load_chats_index(self):
        """Load chats index for user"""
        if self.chats_index_file.exists():
            with open(self.chats_index_file, 'r') as f:
                self.chats_index = json.load(f)
        else:
            self.chats_index = {}
    
    def _save_chats_index(self):
        """Save chats index"""
        with open(self.chats_index_file, 'w') as f:
            json.dump(self.chats_index, f, indent=2)
    
    def create_new_chat(self, title: str = "", initial_topic: str = "") -> str:
        """
        Create a new chat
        
        Returns:
            chat_id (str)
        """
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        chat_data = {
            "chat_id": chat_id,
            "title": title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": now,
            "last_updated": now,
            "messages": [],
            "metadata": {
                "initial_topic": initial_topic,
                "total_messages": 0,
                "research_mode": "interrupts",  # or "streaming"
            },
            "session_id": self.rag_manager.session_id
        }
        
        # Save chat file
        chat_file = self.user_chats_dir / f"{chat_id}.json"
        with open(chat_file, 'w') as f:
            json.dump(chat_data, f, indent=2)
        
        # Update index
        self.chats_index[chat_id] = {
            "title": chat_data["title"],
            "created_at": now,
            "last_updated": now,
            "message_count": 0
        }
        self._save_chats_index()
        
        return chat_id
    
    def load_chat(self, chat_id: str) -> Optional[Dict]:
        """Load a specific chat"""
        chat_file = self.user_chats_dir / f"{chat_id}.json"
        
        if not chat_file.exists():
            return None
        
        with open(chat_file, 'r') as f:
            return json.load(f)
    
    def save_chat(self, chat_id: str, chat_data: Dict):
        """Save chat data"""
        chat_file = self.user_chats_dir / f"{chat_id}.json"
        
        with open(chat_file, 'w') as f:
            json.dump(chat_data, f, indent=2)
        
        # Update index
        if chat_id in self.chats_index:
            self.chats_index[chat_id]["last_updated"] = datetime.now().isoformat()
            self.chats_index[chat_id]["message_count"] = len(chat_data.get("messages", []))
            self._save_chats_index()
    
    def add_message_to_chat(self, chat_id: str, role: str, content: str, 
                           timestamp: str = None, metadata: Dict = None):
        """Add a message to chat (like ChatGPT)"""
        chat = self.load_chat(chat_id)
        if not chat:
            return False
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,  # "user", "assistant", "system"
            "content": content,
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        chat["messages"].append(message)
        chat["last_updated"] = datetime.now().isoformat()
        chat["metadata"]["total_messages"] = len(chat["messages"])
        
        self.save_chat(chat_id, chat)
        
        # Also add to RAG memory for context
        self.rag_manager.session_manager.add_conversation_turn(
            self.rag_manager.session_id,
            role=role,
            content=content[:500],  # Limit for memory
            context_used=metadata.get("context_used", []) if metadata else []
        )
        
        return True
    
    def get_all_chats(self) -> List[Dict]:
        """Get all chats for user (like ChatGPT sidebar)"""
        chats = []
        for chat_id, meta in self.chats_index.items():
            chats.append({
                "chat_id": chat_id,
                "title": meta["title"],
                "created_at": meta["created_at"],
                "last_updated": meta["last_updated"],
                "message_count": meta.get("message_count", 0)
            })
        
        # Sort by last_updated (newest first)
        chats.sort(key=lambda x: x["last_updated"], reverse=True)
        return chats
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat"""
        chat_file = self.user_chats_dir / f"{chat_id}.json"
        
        if chat_file.exists():
            chat_file.unlink()
        
        if chat_id in self.chats_index:
            del self.chats_index[chat_id]
            self._save_chats_index()
            return True
        
        return False
    
    def get_chat_context(self, chat_id: str, last_n_messages: int = 10) -> str:
        """Get recent chat context for RAG"""
        chat = self.load_chat(chat_id)
        if not chat:
            return ""
        
        messages = chat.get("messages", [])[-last_n_messages:]
        context_parts = []
        
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:200]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def search_chats(self, query: str) -> List[Dict]:
        """Search chats by title or content"""
        results = []
        
        for chat_id in self.chats_index.keys():
            chat = self.load_chat(chat_id)
            if not chat:
                continue
            
            # Search in title
            if query.lower() in chat.get("title", "").lower():
                results.append({
                    "chat_id": chat_id,
                    "title": chat["title"],
                    "match": "title"
                })
                continue
            
            # Search in messages
            for msg in chat.get("messages", []):
                if query.lower() in msg["content"].lower():
                    results.append({
                        "chat_id": chat_id,
                        "title": chat["title"],
                        "match": "message",
                        "preview": msg["content"][:100]
                    })
                    break
        
        return results
    
    def get_chat_summary(self, chat_id: str) -> Optional[str]:
        """Get AI-generated summary of chat (uses RAG)"""
        chat = self.load_chat(chat_id)
        if not chat:
            return None
        
        # Get summary from session if available
        session = self.session_manager.load_session(chat.get("session_id"))
        if session and session.summary:
            return session.summary
        
        return None


# ============================================================================
# Memory Storage Strategies (Like ChatGPT)
# ============================================================================

class MemoryStrategy:
    """Base class for memory storage strategies"""
    
    def store(self, chat_id: str, data: Dict):
        raise NotImplementedError
    
    def retrieve(self, chat_id: str):
        raise NotImplementedError


class LocalMemory(MemoryStrategy):
    """Store chats locally (default)"""
    
    def __init__(self, username: str):
        self.chat_manager = ChatManager(username)
    
    def store(self, chat_id: str, data: Dict):
        """Store chat locally"""
        self.chat_manager.save_chat(chat_id, data)
    
    def retrieve(self, chat_id: str):
        """Retrieve chat from local storage"""
        return self.chat_manager.load_chat(chat_id)


class VectorDBMemory(MemoryStrategy):
    """Store chats in Vector DB (ChromaDB) for semantic search"""
    
    def __init__(self, username: str):
        self.chat_manager = ChatManager(username)
        self.rag = self.chat_manager.rag_manager
    
    def store(self, chat_id: str, data: Dict):
        """Store chat and index in VectorDB"""
        # Store locally first
        self.chat_manager.save_chat(chat_id, data)
        
        # Also add to vector store for semantic search
        chat_text = f"Title: {data['title']}\n"
        chat_text += "\n".join([f"{m['role']}: {m['content']}" for m in data["messages"]])
        
        self.rag.add_to_memory(
            text=chat_text,
            metadata={
                "chat_id": chat_id,
                "type": "chat_history",
                "source": "user_chat"
            }
        )
    
    def retrieve(self, chat_id: str):
        """Retrieve chat from local storage"""
        return self.chat_manager.load_chat(chat_id)
