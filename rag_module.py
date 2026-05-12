import os
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import sys

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# These are the preferred imports for LangChain v1+; they may vary by version.
try:
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    raise ImportError(
        "Missing dependencies for RAG module. Install: langchain, langchain-chroma, "
        "langchain-google-genai, chromadb, networkx" 
    ) from e

try:
    import networkx as nx
except ImportError:
    raise ImportError("Please install networkx for knowledge graph support: pip install networkx")

BASE_DIR = Path(os.getenv("CHROMA_DB_DIR", "chromadb_storage")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
KG_FILE = BASE_DIR / "kg_storage.json"
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_METRICS_FILE = BASE_DIR / "memory_metrics.json"

# ============================================================================
# Session & Memory Management Classes (Mem0-like functionality)
# ============================================================================

@dataclass
class MemoryMetric:
    """Tracks memory usage metrics for optimization"""
    timestamp: str
    query: str
    retrieved_count: int
    relevance_score: float
    response_time: float
    memory_footprint_mb: float
    session_id: str
    user_id: str

@dataclass
class SessionMemory:
    """Represents a conversation session with memory tracking"""
    session_id: str
    user_id: str
    created_at: str
    last_accessed: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    important_facts: List[str] = field(default_factory=list)
    queries_count: int = 0
    total_tokens_used: int = 0
    memory_metrics: List[Dict] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_id: str
    timestamp: str
    role: str  # "user" or "assistant"
    content: str
    context_used: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class SessionManager:
    """Manages user sessions with persistent storage"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.sessions_dir = SESSIONS_DIR / user_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_index_file = self.sessions_dir / "index.json"
        self._load_sessions_index()
    
    def _load_sessions_index(self):
        if self.sessions_index_file.exists():
            with open(self.sessions_index_file, 'r') as f:
                self.sessions_index = json.load(f)
        else:
            self.sessions_index = {}
    
    def _save_sessions_index(self):
        with open(self.sessions_index_file, 'w') as f:
            json.dump(self.sessions_index, f, indent=2)
    
    def create_session(self) -> str:
        """Create new session and return session_id"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        session = SessionMemory(
            session_id=session_id,
            user_id=self.user_id,
            created_at=now,
            last_accessed=now
        )
        self.sessions_index[session_id] = {
            "created_at": now,
            "last_accessed": now,
            "conversation_turns": 0
        }
        self._save_session(session)
        self._save_sessions_index()
        logger.info(f"Created new session {session_id} for user {self.user_id}")
        return session_id
    
    def _save_session(self, session: SessionMemory):
        session_file = self.sessions_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(session), f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[SessionMemory]:
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            logger.warning(f"Session {session_id} not found")
            return None
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            session = SessionMemory(**data)
            # Update last_accessed
            session.last_accessed = datetime.now().isoformat()
            self._save_session(session)
            return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def add_conversation_turn(self, session_id: str, role: str, content: str, 
                            context_used: List[str] = None):
        """Add a turn to session conversation history"""
        session = self.load_session(session_id)
        if not session:
            return
        
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            context_used=context_used or []
        )
        session.conversation_history.append(asdict(turn))
        session.queries_count += 1
        self._save_session(session)
    
    def get_session_context(self, session_id: str, max_turns: int = 5) -> str:
        """Get conversation context for current session"""
        session = self.load_session(session_id)
        if not session or not session.conversation_history:
            return ""
        
        recent_turns = session.conversation_history[-max_turns:]
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"{turn['role'].upper()}: {turn['content'][:200]}")
        return "\n".join(context_parts)
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions for user"""
        sessions = []
        for session_id in self.sessions_index.keys():
            session = self.load_session(session_id)
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "last_accessed": session.last_accessed,
                    "turns": len(session.conversation_history),
                    "summary": session.summary
                })
        return sorted(sessions, key=lambda x: x['last_accessed'], reverse=True)
    
    def summarize_session(self, session_id: str, llm=None):
        """Create summary of session using LLM"""
        session = self.load_session(session_id)
        if not session:
            return
        
        history_text = "\n".join([
            f"{t['role']}: {t['content']}" 
            for t in session.conversation_history
        ])
        
        if not llm:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        try:
            prompt = f"""Summarize this research session in 2-3 sentences, highlighting main topics and findings:

{history_text}"""
            response = llm.invoke([SystemMessage(content="You are a summarization expert."), 
                                  HumanMessage(content=prompt)])
            session.summary = response.content
            self._save_session(session)
            logger.info(f"Session {session_id} summarized")
        except Exception as e:
            logger.error(f"Error summarizing session: {e}")

class MemoryMetricsTracker:
    """Tracks memory usage and retrieval metrics"""
    
    def __init__(self):
        self.metrics_file = MEMORY_METRICS_FILE
        self._load_metrics()
    
    def _load_metrics(self):
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = []
    
    def _save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def record_retrieval(self, user_id: str, session_id: str, query: str, 
                        retrieved_count: int, relevance_score: float, response_time: float):
        """Record a retrieval operation"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "user_id": user_id,
            "session_id": session_id,
            "retrieved_count": retrieved_count,
            "relevance_score": relevance_score,
            "response_time": response_time,
            "memory_footprint_mb": memory_usage
        }
        self.metrics.append(metric)
        self._save_metrics()
    
    def get_user_stats(self, user_id: str, days: int = 30) -> Dict:
        """Get user statistics"""
        cutoff = datetime.now() - timedelta(days=days)
        user_metrics = [
            m for m in self.metrics 
            if m['user_id'] == user_id and 
            datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
        if not user_metrics:
            return {}
        
        return {
            "total_retrievals": len(user_metrics),
            "avg_relevance_score": sum(m['relevance_score'] for m in user_metrics) / len(user_metrics),
            "avg_response_time": sum(m['response_time'] for m in user_metrics) / len(user_metrics),
            "peak_memory_mb": max(m['memory_footprint_mb'] for m in user_metrics),
            "total_documents_retrieved": sum(m['retrieved_count'] for m in user_metrics)
        }
    
    def get_memory_summary(self) -> Dict:
        """Get overall memory system summary"""
        return {
            "total_operations": len(self.metrics),
            "total_memory_samples": len(self.metrics),
            "avg_memory_usage_mb": sum(m['memory_footprint_mb'] for m in self.metrics) / len(self.metrics) if self.metrics else 0
        }



class KnowledgeGraph:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph = nx.Graph()
        self.data_file = KG_FILE
        self.access_count = defaultdict(int)  # Track node access frequency
        self._load()

    def _load(self):
        if self.data_file.exists():
            try:
                loaded = json.loads(self.data_file.read_text(encoding="utf-8"))
                user_data = loaded.get(self.user_id, {})
                self.graph = nx.node_link_graph(user_data) if user_data else nx.Graph()
            except Exception as e:
                logger.warning(f"Failed to load KG file, starting new graph: {e}")
                self.graph = nx.Graph()

    def _save(self):
        store = {}
        if self.data_file.exists():
            try:
                store = json.loads(self.data_file.read_text(encoding="utf-8"))
            except Exception:
                store = {}

        store[self.user_id] = nx.node_link_data(self.graph)
        self.data_file.write_text(json.dumps(store, indent=2), encoding="utf-8")

    def add_triple(self, source: str, relation: str, target: str, metadata: Optional[Dict] = None):
        self.graph.add_node(source)
        self.graph.add_node(target)
        self.graph.add_edge(source, target, relation=relation, metadata=metadata or {})
        self._save()

    def get_related(self, node: str, depth: int = 2) -> List[str]:
        if node not in self.graph:
            return []
        self.access_count[node] += 1  # Track access
        neighbors = set([node])
        current = {node}
        for _ in range(depth):
            next_level = set()
            for n in current:
                next_level |= set(self.graph.neighbors(n))
            neighbors |= next_level
            current = next_level
        neighbors.discard(node)
        return list(neighbors)

    def build_from_text(self, text: str, source_label: Optional[str] = None):
        tokens = re.findall(r"[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*", text)
        tokens = [t.strip() for t in tokens if len(t.strip()) > 2]
        tokens = list(dict.fromkeys(tokens))  # keep order unique
        if source_label:
            self.graph.add_node(source_label)
        for i, t in enumerate(tokens[:10]):
            node = t
            self.graph.add_node(node)
            if source_label:
                self.graph.add_edge(source_label, node, relation="mentioned_in")
            if i > 0:
                self.graph.add_edge(tokens[i - 1], node, relation="related")
        self._save()

    def get_context(self, query: str, max_nodes: int = 12) -> str:
        entities = re.findall(r"[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*", query)
        entities = [e for e in entities if len(e) > 2]
        context = []
        for ent in entities:
            related = self.get_related(ent)
            if related:
                context.append(f"{ent} -> {', '.join(related[:max_nodes])}")
        return "\n".join(context)
    
    def get_top_nodes(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed nodes"""
        return sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_graph_stats(self) -> Dict:
        """Get knowledge graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "most_accessed": self.get_top_nodes(5)
        }




class RAGManager:
    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        logger.debug(f"Initializing RAGManager for user: {user_id}")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.kg = KnowledgeGraph(user_id)
        
        # Session management
        self.session_manager = SessionManager(user_id)
        self.session_id = session_id or self.session_manager.create_session()
        
        # Memory metrics tracking
        self.metrics_tracker = MemoryMetricsTracker()
        
        # Document cache for performance
        self._doc_cache = {}
        self._cache_timestamp = datetime.now()

        self.vector_store = Chroma(
            persist_directory=str(BASE_DIR),
            embedding_function=self.embeddings,
            collection_name=f"kg_user_{user_id}"
        )

    def add_to_memory(self, text: str, metadata: Optional[Dict] = None):
        """Add document to memory with session tracking"""
        if metadata is None:
            metadata = {}
        metadata["user_id"] = self.user_id
        metadata["session_id"] = self.session_id
        metadata["doc_id"] = metadata.get("doc_id", str(uuid.uuid4()))
        metadata["timestamp"] = datetime.now().isoformat()

        document = Document(page_content=text, metadata=metadata)
        self.vector_store.add_documents([document])
        self.kg.build_from_text(text, source_label=metadata["doc_id"])
        logger.info(f"Document added to memory for session {self.session_id}")

    def search_memory(self, query: str, k: int = 3) -> List[Document]:
        """Search memory with session context and metrics tracking"""
        import time
        start_time = time.time()
        
        query_clean = query.strip()
        if len(query_clean) > 300:
            query_clean = query_clean[:300]

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k, "filter": {"user_id": self.user_id}}
        )
        docs = retriever.invoke(query_clean)
        
        # Calculate relevance scores
        relevance_score = len(docs) / max(k, 1)
        response_time = time.time() - start_time
        
        # Record metrics
        self.metrics_tracker.record_retrieval(
            user_id=self.user_id,
            session_id=self.session_id,
            query=query_clean,
            retrieved_count=len(docs),
            relevance_score=relevance_score,
            response_time=response_time
        )
        
        # Add to session history
        self.session_manager.add_conversation_turn(
            self.session_id,
            role="user",
            content=query_clean,
            context_used=[doc.metadata.get("doc_id") for doc in docs]
        )
        
        logger.info(f"Retrieved {len(docs)} documents in {response_time:.2f}s")
        return docs

    def search_with_session_context(self, query: str, k: int = 3) -> Tuple[List[Document], str]:
        """Search that includes previous session context"""
        session_context = self.session_manager.get_session_context(self.session_id, max_turns=3)
        
        # Enhance query with session context if available
        if session_context:
            enhanced_query = f"{query}\nPrevious context: {session_context[:200]}"
        else:
            enhanced_query = query
        
        docs = self.search_memory(enhanced_query, k=k)
        return docs, session_context

    def get_all_user_docs(self) -> List[Dict]:
        """Get all documents for user"""
        try:
            return self.vector_store.get(where={"user_id": self.user_id})
        except Exception:
            return []

    def as_retriever(self, k: int = 4):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k, "filter": {"user_id": self.user_id}}
        )

    def graph_rag_answer(self, query: str, k: int = 5) -> str:
        """Generate RAG answer with knowledge graph context"""
        docs, session_ctx = self.search_with_session_context(query, k=k)
        rag_context = "\n\n".join([f"- {d.page_content[:300]}" for d in docs])
        kg_context = self.kg.get_context(query)

        full_context = "".join([
            "### Session Context:\n", session_ctx or "(No previous context)", "\n\n",
            "### RAG context from retrieval:\n", rag_context, "\n\n",
            "### Knowledge graph context:\n", kg_context or "(KG has no related nodes yet)", "\n\n",
            "### User question:\n", query
        ])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = f"You are a research assistant with memory of previous sessions. Use the provided context to answer succinctly.\n\n{full_context}"
        
        response = llm.invoke([
            SystemMessage(content="You are a research assistant with memory of previous sessions."),
            HumanMessage(content=prompt)
        ])
        
        answer = response.content

        # Record assistant response in session
        self.session_manager.add_conversation_turn(
            self.session_id,
            role="assistant",
            content=answer,
            context_used=[doc.metadata.get("doc_id") for doc in docs]
        )
        
        return answer

    def add_kg_fact(self, source: str, relation: str, target: str):
        """Add fact to knowledge graph"""
        self.kg.add_triple(source, relation, target)

    def get_kg_context(self, query: str, max_nodes: int = 8) -> str:
        return self.kg.get_context(query, max_nodes=max_nodes)
    
    # ========================================================================
    # Advanced Memory & Session Features
    # ========================================================================
    
    def get_session_info(self) -> Dict:
        """Get current session information"""
        session = self.session_manager.load_session(self.session_id)
        if not session:
            return {}
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "conversation_turns": len(session.conversation_history),
            "queries_count": session.queries_count,
            "summary": session.summary
        }
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions for current user"""
        return self.session_manager.get_all_sessions()
    
    def load_previous_session(self, session_id: str) -> bool:
        """Load previous session as current"""
        session = self.session_manager.load_session(session_id)
        if session:
            self.session_id = session_id
            logger.info(f"Loaded previous session {session_id}")
            return True
        return False
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        return self.metrics_tracker.get_user_stats(self.user_id)
    
    def get_kg_stats(self) -> Dict:
        """Get knowledge graph statistics"""
        return self.kg.get_graph_stats()
    
    def summarize_current_session(self):
        """Create summary of current session"""
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.session_manager.summarize_session(self.session_id, llm=llm)
        logger.info(f"Session {self.session_id} summarized")
    
    def extract_important_facts(self) -> List[str]:
        """Extract important facts from session"""
        session = self.session_manager.load_session(self.session_id)
        if not session or not session.conversation_history:
            return []
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        history_text = "\n".join([
            f"{t['role']}: {t['content']}" 
            for t in session.conversation_history[-10:]
        ])
        
        try:
            prompt = f"""Extract key facts and findings from this conversation (bullet points):

{history_text}"""
            response = llm.invoke([
                SystemMessage(content="Extract important facts, only list facts."),
                HumanMessage(content=prompt)
            ])
            facts = [f.strip() for f in response.content.split('\n') if f.strip()]
            session.important_facts = facts
            self.session_manager._save_session(session)
            return facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []
    
    def get_system_memory_summary(self) -> Dict:
        """Get system-wide memory metrics"""
        return self.metrics_tracker.get_memory_summary()
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        sessions = self.session_manager.get_all_sessions()
        
        cleaned = 0
        for session in sessions:
            last_accessed = datetime.fromisoformat(session['last_accessed'])
            if last_accessed < cutoff:
                session_file = self.session_manager.sessions_dir / f"{session['session_id']}.json"
                if session_file.exists():
                    session_file.unlink()
                    del self.session_manager.sessions_index[session['session_id']]
                    cleaned += 1
        
        self.session_manager._save_sessions_index()
        logger.info(f"Cleaned up {cleaned} old sessions")
    
    def get_memory_footprint(self) -> Dict:
        """Analyze memory footprint"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "sessions_count": len(self.session_manager.sessions_index),
            "docs_count": len(self.get_all_user_docs()),
            "kg_nodes": self.kg.graph.number_of_nodes(),
            "kg_edges": self.kg.graph.number_of_edges()
        }



