#!/usr/bin/env python3
"""
Mem0 REST API Wrapper

Complete guide to using Mem0 REST API with Python.
All HTTP endpoints with examples.
"""

import os
import json
import requests
from typing import Dict, List, Optional
from loguru import logger

# Configure logger
logger.add("logs/mem0_api.log", rotation="500 MB")


class Mem0RestAPI:
    """Mem0 REST API Client"""
    
    BASE_URL = "https://api.mem0.ai/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mem0 REST API client
        
        Args:
            api_key: Mem0 API key (or use MEM0_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MEM0_API_KEY not found. Set it in .env or pass as parameter.\n"
                "Get key at: https://app.mem0.ai/dashboard/api-keys"
            )
        
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def add_memory(self, messages: List[Dict], user_id: str) -> Dict:
        """
        Add memory from conversation
        
        Args:
            messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            user_id: Unique user identifier
        
        Returns:
            API response
        """
        url = f"{self.BASE_URL}/memory/add"
        payload = {
            "messages": messages,
            "user_id": user_id
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Memory added for user {user_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding memory: {e}")
            return {"error": str(e)}
    
    def search_memories(self, query: str, user_id: str, limit: int = 5) -> Dict:
        """
        Search user's memories
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Max results
        
        Returns:
            Search results
        """
        url = f"{self.BASE_URL}/memory/search"
        payload = {
            "query": query,
            "user_id": user_id,
            "limit": limit
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Memory search completed for user {user_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching memories: {e}")
            return {"error": str(e)}
    
    def get_memories(self, user_id: str) -> Dict:
        """
        Get all memories for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            All memories for user
        """
        url = f"{self.BASE_URL}/memory/get"
        payload = {"user_id": user_id}
        
        try:
            response = requests.get(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Retrieved memories for user {user_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting memories: {e}")
            return {"error": str(e)}
    
    def update_memory(self, memory_id: str, content: str, user_id: str) -> Dict:
        """
        Update existing memory
        
        Args:
            memory_id: Memory ID to update
            content: New content
            user_id: User identifier
        
        Returns:
            API response
        """
        url = f"{self.BASE_URL}/memory/update"
        payload = {
            "id": memory_id,
            "content": content,
            "user_id": user_id
        }
        
        try:
            response = requests.put(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Memory {memory_id} updated")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating memory: {e}")
            return {"error": str(e)}
    
    def delete_memory(self, memory_id: str, user_id: str) -> Dict:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID to delete
            user_id: User identifier
        
        Returns:
            API response
        """
        url = f"{self.BASE_URL}/memory/delete"
        payload = {
            "id": memory_id,
            "user_id": user_id
        }
        
        try:
            response = requests.delete(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Memory {memory_id} deleted")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e)}


# ==============================================================================
# REST API ENDPOINT EXAMPLES (cURL)
# ==============================================================================

CURL_EXAMPLES = """
🌐 CURL EXAMPLES - Use these in terminal

1️⃣ ADD MEMORY
curl -X POST https://api.mem0.ai/v1/memory/add \\
  -H "Authorization: Token YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [
      {"role": "user", "content": "I like Python and AI"},
      {"role": "assistant", "content": "Got it! You like Python and AI"}
    ],
    "user_id": "user123"
  }'

2️⃣ SEARCH MEMORIES
curl -X POST https://api.mem0.ai/v1/memory/search \\
  -H "Authorization: Token YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What programming languages do I prefer?",
    "user_id": "user123",
    "limit": 5
  }'

3️⃣ GET ALL MEMORIES
curl -X GET https://api.mem0.ai/v1/memory/get \\
  -H "Authorization: Token YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "user123"}'

4️⃣ UPDATE MEMORY
curl -X PUT https://api.mem0.ai/v1/memory/update \\
  -H "Authorization: Token YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "id": "memory_id_here",
    "content": "Updated content",
    "user_id": "user123"
  }'

5️⃣ DELETE MEMORY
curl -X DELETE https://api.mem0.ai/v1/memory/delete \\
  -H "Authorization: Token YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "id": "memory_id_here",
    "user_id": "user123"
  }'
"""


# ==============================================================================
# PYTHON EXAMPLES
# ==============================================================================

def example_python_usage():
    """Complete Python example"""
    
    print("\n" + "="*70)
    print("🐍 PYTHON EXAMPLES")
    print("="*70)
    
    # Initialize API
    try:
        api = Mem0RestAPI()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    user_id = "demo_user_001"
    
    # 1. Add memory
    print("\n1️⃣ Adding memory...")
    messages = [
        {"role": "user", "content": "I'm a Python developer interested in AI"},
        {"role": "assistant", "content": "Great! I'll remember your interests"}
    ]
    result = api.add_memory(messages, user_id)
    print(json.dumps(result, indent=2))
    
    # 2. Search memories
    print("\n2️⃣ Searching memories...")
    results = api.search_memories(
        "What's my professional background?",
        user_id
    )
    print(json.dumps(results, indent=2))
    
    # 3. Get all memories
    print("\n3️⃣ Getting all memories...")
    all_memories = api.get_memories(user_id)
    print(json.dumps(all_memories, indent=2))
    
    # 4. Update memory (if memory_id exists)
    print("\n4️⃣ Update example (need memory_id from above)")
    print("   api.update_memory('memory_id', 'new content', user_id)")
    
    # 5. Delete memory (if memory_id exists)
    print("\n5️⃣ Delete example (need memory_id from above)")
    print("   api.delete_memory('memory_id', user_id)")


# ==============================================================================
# NODEJS/JAVASCRIPT EXAMPLES
# ==============================================================================

JAVASCRIPT_EXAMPLES = """
📱 JAVASCRIPT/NODE.JS EXAMPLES

1️⃣ Installation
npm install mem0ai

2️⃣ Add Memory
const { MemoryClient } = require('mem0ai');
const client = new MemoryClient({ apiKey: 'your-api-key' });

const messages = [
  { role: 'user', content: 'I prefer JavaScript' },
  { role: 'assistant', content: 'Got it!' }
];

await client.add(messages, { userId: 'user123' });

3️⃣ Search Memories
const results = await client.search(
  'What are my preferences?',
  { userId: 'user123', limit: 5 }
);

4️⃣ Get All Memories
const allMemories = await client.get({ userId: 'user123' });

5️⃣ Update Memory
await client.update('memory_id', 'new content', { userId: 'user123' });

6️⃣ Delete Memory
await client.delete('memory_id', { userId: 'user123' });
"""


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEM0 REST API COMPLETE GUIDE")
    print("="*70)
    
    print("\n📝 SETUP INSTRUCTIONS:")
    print("-" * 70)
    print("""
1. Get API Key:
   - Go to https://app.mem0.ai
   - Sign up (free)
   - Dashboard → Settings → API Keys → Create New Key

2. Save API Key to .env:
   echo "MEM0_API_KEY=your-key-here" >> .env

3. Run this script:
   python mem0_api_rest.py
    """)
    
    print("\n" + CURL_EXAMPLES)
    print("\n" + JAVASCRIPT_EXAMPLES)
    
    # Run Python examples
    example_python_usage()
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\n📚 Full API docs: https://docs.mem0.ai/api-reference")
    print("💬 Support: https://discord.gg/mem0ai\n")
