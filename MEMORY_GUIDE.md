# Memory Management System - Complete Guide

## 📚 Overview

Your AI Research Assistant now has a **ChatGPT-like memory system** that:
- ✅ Stores chat history per user
- ✅ Manages multiple conversations
- ✅ Implements user authentication
- ✅ Persists data with vector embeddings
- ✅ Supports semantic search across chats

---

## 🧠 How ChatGPT Stores Memory

### 1. **Session-Based Storage**
```
ChatGPT Architecture:
┌─────────────────┐
│   User Login    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Session Token      │ (JWT-like)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Chat Instance      │ (Current conversation)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Message History    │ (All messages in chat)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Long-term Memory   │ (Vector embeddings)
└─────────────────────┘
```

### 2. **Three Types of Memory**

#### **A. Short-term Memory (Current Chat)**
- Messages in current conversation
- Context window (last N messages)
- Used for immediate context

#### **B. Long-term Memory (All Chats)**
- Previous conversations stored in DB
- Vector embeddings for semantic search
- Retrievable across sessions

#### **C. System Memory (User Profile)**
- User preferences
- Authentication tokens
- Chat history index
- Usage statistics

---

## 🏗️ Your Implementation Architecture

### **File Structure**
```
LangchainProject/
├── auth_manager.py          # User authentication
├── chat_manager.py          # Chat & memory management
├── rag_module.py            # Vector embeddings (ChromaDB)
├── main.py                  # Streamlit UI
│
├── users.json               # User credentials
├── sessions.json            # Active sessions
├── chats_storage/           # Chat history
│   └── {username}/
│       ├── chats_index.json
│       ├── {chat_id}.json
│       └── {chat_id}.json
└── chromadb_storage/        # Vector embeddings
    ├── sessions/            # Session data
    └── kg_storage.json      # Knowledge graph
```

---

## 🔐 Authentication Flow (Like ChatGPT)

```python
from auth_manager import AuthManager

auth = AuthManager()

# 1. REGISTER
success, msg = auth.register_user("mahek", "password123", "mahek@example.com")
# Returns: (True, "User 'mahek' registered successfully")

# 2. LOGIN
success, msg, token = auth.login_user("mahek", "password123")
# Returns: (True, "Welcome mahek!", "abc123-session-token")

# 3. USE TOKEN TO VERIFY SESSION
is_valid, username = auth.verify_session(token)
# Returns: (True, "mahek")

# 4. LOGOUT
auth.logout_user(token)
```

### **Session Token Validity**
- **Duration**: 7 days
- **Auto-logout**: After 7 days of inactivity
- **Invalidation**: On logout or password reset

---

## 💬 Chat Management (Like ChatGPT)

### **Creating New Chat**
```python
from chat_manager import ChatManager

chat_mgr = ChatManager(username="mahek")

# Create new chat
chat_id = chat_mgr.create_new_chat(
    title="Healthcare vs AI",
    initial_topic="Impact of AI on healthcare"
)
# Returns: "550e8400-e29b-41d4-a716-446655440000"
```

### **Adding Messages (Persistent Storage)**
```python
# User message
chat_mgr.add_message_to_chat(
    chat_id=chat_id,
    role="user",
    content="How does AI impact healthcare?",
    metadata={"sources": ["source1", "source2"]}
)

# Assistant response
chat_mgr.add_message_to_chat(
    chat_id=chat_id,
    role="assistant",
    content="AI revolutionizes healthcare by...",
    metadata={"model": "gemini-2.5-flash"}
)
```

### **Loading Chat History**
```python
# Load specific chat
chat = chat_mgr.load_chat(chat_id)
print(chat["messages"])  # All messages

# Get all chats (sidebar list)
all_chats = chat_mgr.get_all_chats()
# Returns: List of chats sorted by last_updated

# Get recent chat context
context = chat_mgr.get_chat_context(chat_id, last_n_messages=5)
```

---

## 🧠 Memory Types & Where to Use Them

### **1. LOCAL MEMORY (Default) ✅**
```python
from chat_manager import LocalMemory

memory = LocalMemory(username="mahek")
memory.store(chat_id, chat_data)
chat = memory.retrieve(chat_id)
```

**Use for:**
- ✅ Quick access to recent chats
- ✅ Offline mode
- ✅ Privacy (all local)
- ❌ Semantic search across chats

**Storage**: `chats_storage/{username}/{chat_id}.json`

---

### **2. VECTOR DB MEMORY (Recommended) ⭐**
```python
from chat_manager import VectorDBMemory

memory = VectorDBMemory(username="mahek")
memory.store(chat_id, chat_data)
# Stores both locally AND in ChromaDB
```

**Use for:**
- ✅ Semantic search
- ✅ "Find all chats about healthcare"
- ✅ Knowledge graph integration
- ✅ Better context retrieval

**Storage**: 
- Local: `chats_storage/{username}/{chat_id}.json`
- Vector: `chromadb_storage/kg_user_{username}/`

---

### **3. SESSION MEMORY (Background)**
```python
from rag_module import SessionManager

session_mgr = SessionManager(user_id="mahek")
session_id = session_mgr.create_session()
session_mgr.add_conversation_turn(
    session_id,
    role="user",
    content="Research query"
)
```

**Use for:**
- ✅ During active research
- ✅ Conversation history
- ✅ Token tracking
- ✅ Performance metrics

**Storage**: `chromadb_storage/sessions/{user_id}/{session_id}.json`

---

## 📊 Memory Comparison Table

| Feature | Local | Vector DB | Session |
|---------|-------|-----------|---------|
| **Speed** | Fast ⚡ | Medium 🟡 | Medium 🟡 |
| **Search** | Basic | Semantic ⭐ | Basic |
| **Scalability** | Limited | Excellent ⭐ | Medium |
| **Privacy** | Offline ⭐ | Requires DB | Local |
| **Size Limit** | Disk | RAM + Disk | RAM |
| **Cost** | Free ⭐ | Storage | Free |

**Recommendation**: Use **VectorDBMemory** for production (combines speed + search)

---

## 🔄 Complete Integration Example

```python
import streamlit as st
from auth_manager import AuthManager
from chat_manager import ChatManager, VectorDBMemory

# ============================================================================
# STEP 1: AUTHENTICATION
# ============================================================================

auth = AuthManager()

# Login page (initial)
with st.sidebar:
    if "session_token" not in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state.show_login = True
        with col2:
            if st.button("Register"):
                st.session_state.show_register = True
        
        if st.session_state.get("show_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                success, msg, token = auth.login_user(username, password)
                if success:
                    st.session_state.session_token = token
                    st.session_state.username = username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

# ============================================================================
# STEP 2: GET USERNAME FROM SESSION TOKEN
# ============================================================================

if "session_token" in st.session_state:
    session_token = st.session_state.session_token
    is_valid, username = auth.verify_session(session_token)
    
    if not is_valid:
        st.error("Session expired. Please login again.")
        st.stop()
    
    # ====================================================================
    # STEP 3: INITIALIZE CHAT MANAGER
    # ====================================================================
    
    chat_mgr = ChatManager(username)
    memory = VectorDBMemory(username)
    
    # Sidebar: Chat history
    with st.sidebar:
        st.header(f"👤 {username}")
        
        if st.button("➕ New Chat"):
            chat_id = chat_mgr.create_new_chat(
                title=f"Chat {datetime.now().strftime('%H:%M')}"
            )
            st.session_state.current_chat_id = chat_id
            st.rerun()
        
        st.markdown("---")
        st.subheader("📝 Recent Chats")
        
        all_chats = chat_mgr.get_all_chats()
        for chat in all_chats[:10]:  # Last 10 chats
            if st.button(f"• {chat['title']}", key=chat['chat_id']):
                st.session_state.current_chat_id = chat['chat_id']
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            auth.logout_user(session_token)
            del st.session_state.session_token
            st.rerun()
    
    # ====================================================================
    # STEP 4: MAIN CHAT INTERFACE
    # ====================================================================
    
    # Get or create current chat
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = chat_mgr.create_new_chat()
    
    chat_id = st.session_state.current_chat_id
    chat = chat_mgr.load_chat(chat_id)
    
    st.title(chat["title"])
    
    # Display chat messages (persistent)
    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # ====================================================================
    # STEP 5: NEW MESSAGE INPUT
    # ====================================================================
    
    user_input = st.chat_input("Type your question...")
    
    if user_input:
        # Save user message
        chat_mgr.add_message_to_chat(
            chat_id,
            role="user",
            content=user_input
        )
        
        # Get context from memory
        context = chat_mgr.get_chat_context(chat_id, last_n_messages=5)
        
        # Generate response (call your research graph)
        response = "AI generated response..."
        
        # Save assistant message
        chat_mgr.add_message_to_chat(
            chat_id,
            role="assistant",
            content=response,
            metadata={"context_used": context}
        )
        
        # Also store in vector DB for semantic search
        memory.store(chat_id, chat_mgr.load_chat(chat_id))
        
        st.rerun()
```

---

## 🔍 Semantic Search Example

```python
from chat_manager import ChatManager

chat_mgr = ChatManager(username="mahek")

# Search across all chats
results = chat_mgr.search_chats("AI healthcare impact")

for result in results:
    print(f"Chat: {result['title']}")
    print(f"Match: {result['match']}")
    if 'preview' in result:
        print(f"Preview: {result['preview']}")
```

---

## 💾 Data Persistence

### **What Gets Saved Automatically**
1. ✅ Chat messages (local file)
2. ✅ Chat metadata (title, timestamps)
3. ✅ Vector embeddings (ChromaDB)
4. ✅ User profile (auth)
5. ✅ Session tokens
6. ✅ Research results per chat

### **What Gets Cleaned Up**
- ❌ Old sessions (7+ days inactive)
- ❌ Expired tokens
- ❌ Temporary data

---

## 🚀 Next Steps

1. **Integrate Auth into main.py** - Add login/register pages
2. **Add Chat Sidebar** - Show all chats
3. **Implement Vector Search** - Find related chats
4. **Add Export/Import** - Download chat history
5. **Implement Rate Limiting** - Prevent abuse

---

## 📋 Summary

| Component | Purpose | Storage |
|-----------|---------|---------|
| **AuthManager** | User login/register | `users.json` |
| **ChatManager** | Chat history | `chats_storage/` |
| **SessionManager** | Active sessions | `chromadb_storage/sessions/` |
| **RAGManager** | Vector search | `chromadb_storage/` |
| **LocalMemory** | Quick chat access | Local JSON |
| **VectorDBMemory** | Semantic search | ChromaDB + JSON |

Use **VectorDBMemory** for the best experience! ⭐
