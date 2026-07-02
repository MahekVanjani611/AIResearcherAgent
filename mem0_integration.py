#!/usr/bin/env python3
"""
Mem0 Integration Module

Integrates Mem0 long-term memory system with your AI Research Assistant.
Mem0 learns from user interactions and improves responses over time.

Features:
  - Persistent user memory across sessions
  - Automatic memory extraction from conversations
  - Context-aware responses based on historical behavior
  - Memory search and retrieval
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

from loguru import logger

# Configure logger
logger.add("logs/mem0_integration.log", rotation="500 MB")


class Mem0Manager:
    """Manages Mem0 integration with your chat system"""
    
    def __init__(self, api_key: Optional[str] = None, user_id: Optional[str] = None):
        """
        Initialize Mem0Manager
        
        Args:
            api_key: Mem0 API key (or use MEM0_API_KEY env var)
            user_id: Unique user identifier
        """
        if MemoryClient is None:
            raise ImportError("mem0ai not installed. Run: pip install mem0ai")
        
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            logger.warning("MEM0_API_KEY not set. Memory features will be limited.")
            self.client = None
        else:
            self.client = MemoryClient(api_key=self.api_key)
        
        self.user_id = user_id or "default_user"
        self.memory_cache_dir = Path("mem0_cache")
        self.memory_cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.memory_cache_dir / f"{self.user_id}_memories.json"
        
        self._load_memory_cache()
    
    def _load_memory_cache(self):
        """Load cached memories from local file"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.memory_cache = json.load(f)
        else:
            self.memory_cache = {"memories": []}
    
    def _save_memory_cache(self):
        """Save memories to local cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.memory_cache, f, indent=2)
    
    def add_memory_from_conversation(self, user_message: str, assistant_response: str) -> Dict:
        """
        Add memory from a conversation exchange
        
        Args:
            user_message: User's input
            assistant_response: Assistant's response
        
        Returns:
            Memory addition result
        """
        if not self.client:
            logger.warning("Mem0 client not configured. Skipping memory storage.")
            return {"status": "skipped", "reason": "No API key"}
        
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response}
            ]
            
            result = self.client.add(messages, user_id=self.user_id)
            
            # Cache locally
            self.memory_cache["memories"].append({
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "mem0_result": result
            })
            self._save_memory_cache()
            
            logger.info(f"Memory added for user {self.user_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {"status": "error", "error": str(e)}
    
    def add_memories_batch(self, conversations: List[Dict]) -> Dict:
        """
        Add multiple memories at once (batch operation)
        
        Args:
            conversations: List of {"user_message": "...", "assistant_response": "..."}
        
        Returns:
            Batch operation result
        """
        if not conversations:
            return {"status": "skipped", "reason": "Empty list"}
        
        results = {
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for conv in conversations:
            result = self.add_memory_from_conversation(
                conv.get("user_message", ""),
                conv.get("assistant_response", "")
            )
            
            if "error" not in result:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append(result)
        
        logger.info(f"Batch: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search user's memories
        
        Args:
            query: Search query
            limit: Max results to return
        
        Returns:
            List of matching memories
        """
        if not self.client:
            logger.warning("Mem0 client not configured. Returning cached results only.")
            return self._search_local_cache(query)
        
        try:
            results = self.client.search(query, user_id=self.user_id, limit=limit)
            logger.info(f"Memory search completed: {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return self._search_local_cache(query)
    
    def search_memories_batch(self, queries: List[str], limit: int = 5) -> Dict:
        """
        Search multiple queries at once
        
        Args:
            queries: List of search queries
            limit: Max results per query
        
        Returns:
            Dictionary with results for each query
        """
        results = {}
        
        for query in queries:
            results[query] = self.search_memories(query, limit)
        
        logger.info(f"Batch search completed for {len(queries)} queries")
        return results
    
    def _search_local_cache(self, query: str, limit: int = 5) -> List[Dict]:
        """Search local memory cache (fallback)"""
        results = []
        query_lower = query.lower()
        
        for memory in self.memory_cache.get("memories", []):
            if (query_lower in memory["user_message"].lower() or 
                query_lower in memory["assistant_response"].lower()):
                results.append({
                    "timestamp": memory["timestamp"],
                    "user_message": memory["user_message"],
                    "assistant_response": memory["assistant_response"],
                    "source": "local_cache"
                })
        
        return results[:limit]
    
    def get_user_context(self) -> str:
        """
        Get summarized user context from memories
        
        Returns:
            Formatted string of user preferences and history
        """
        try:
            memories = self.client.get(user_id=self.user_id) if self.client else []
            
            if not memories:
                return "No user memories found."
            
            # Format memories as context
            context_lines = ["=== User Context from Memory ==="]
            for i, memory in enumerate(memories[:10], 1):
                context_lines.append(f"{i}. {memory.get('memory', 'N/A')}")
            
            return "\n".join(context_lines)
        
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return "Error retrieving user context."
    
    def update_memory(self, memory_id: str, new_content: str) -> Dict:
        """Update existing memory"""
        if not self.client:
            logger.warning("Mem0 client not configured.")
            return {"status": "error", "reason": "No API key"}
        
        try:
            result = self.client.update(memory_id, new_content, user_id=self.user_id)
            logger.info(f"Memory {memory_id} updated")
            return result
        
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return {"status": "error", "error": str(e)}
    
    def delete_memory(self, memory_id: str) -> Dict:
        """Delete a memory"""
        if not self.client:
            logger.warning("Mem0 client not configured.")
            return {"status": "error", "reason": "No API key"}
        
        try:
            result = self.client.delete(memory_id, user_id=self.user_id)
            logger.info(f"Memory {memory_id} deleted")
            return result
        
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about user's memories"""
        stats = {
            "user_id": self.user_id,
            "cached_memories": len(self.memory_cache.get("memories", [])),
            "cache_file": str(self.cache_file),
            "api_configured": self.client is not None
        }
        return stats


# Example usage functions
def demonstrate_mem0_features():
    """Demonstrate Mem0 capabilities"""
    
    # Initialize
    mem0 = Mem0Manager(user_id="research_user_001")
    
    print("\n🧠 Mem0 Integration Demo\n")
    print("=" * 50)
    
    # 1. Add single memory
    print("\n1️⃣ Adding single memory...")
    mem0.add_memory_from_conversation(
        user_message="I'm a vegetarian and allergic to nuts.",
        assistant_response="Got it! I'll remember your dietary preferences."
    )
    
    # 2. Add multiple memories at once
    print("\n2️⃣ Adding multiple memories (batch)...")
    batch_conversations = [
        {
            "user_message": "I work in machine learning",
            "assistant_response": "Great! I'll remember your ML background"
        },
        {
            "user_message": "I prefer Python over JavaScript",
            "assistant_response": "Noted! Python is your preference"
        },
        {
            "user_message": "I like working with data",
            "assistant_response": "Got it! You enjoy data work"
        }
    ]
    batch_result = mem0.add_memories_batch(batch_conversations)
    print(f"   Successful: {batch_result['successful']}, Failed: {batch_result['failed']}")
    
    # 3. Search single memory
    print("\n3️⃣ Searching single memory...")
    results = mem0.search_memories("What are my dietary restrictions?")
    print(f"   Found: {len(results)} results")
    
    # 4. Search multiple queries at once
    print("\n4️⃣ Searching multiple queries (batch)...")
    queries = [
        "What's my professional background?",
        "What programming languages do I prefer?",
        "What are my food allergies?"
    ]
    batch_search = mem0.search_memories_batch(queries)
    for query, results in batch_search.items():
        print(f"   Query: {query}")
        print(f"   Results: {len(results)} found")
    
    # 5. Get user context
    print("\n5️⃣ Getting user context for better responses...")
    context = mem0.get_user_context()
    print(context)
    
    # 6. Memory stats
    print("\n6️⃣ Memory statistics...")
    stats = mem0.get_memory_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_mem0_features()
