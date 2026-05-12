#!/usr/bin/env python3
"""
Simplified Example: Using Enhanced RAG with Minimal API Calls

This demonstrates:
1. Adding documents
2. Searching and retrieving documents
3. Monitoring memory metrics
4. Session management (without LLM calls)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from rag_module import RAGManager
from datetime import datetime

# ============================================================================
# QUICK SAMPLE DOCUMENTS
# ============================================================================

QUICK_DOCS = [
    ("Python is a high-level programming language known for simplicity.", "Python Basics"),
    ("React is a JavaScript library for building user interfaces.", "Frontend Dev"),
    ("MongoDB is a NoSQL database for flexible document storage.", "Databases"),
    ("Docker containers provide lightweight application packaging.", "DevOps"),
    ("Machine Learning enables computers to learn from data patterns.", "AI/ML"),
]

# ============================================================================
# QUICK TEST QUERIES
# ============================================================================

QUICK_QUERIES = [
    "Tell me about Python",
    "What is React used for?",
    "Explain MongoDB",
    "How do Docker containers work?",
    "What is Machine Learning?",
]


def print_section(title: str):
    """Print formatted section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_subsection(title: str):
    """Print formatted subsection"""
    print(f"\n➜ {title}")
    print("-" * 70)


def main():
    print_section("QUICK RAG EXAMPLE - NO API QUOTA ISSUES")
    
    # Initialize
    print_subsection("1️⃣  Initializing RAG System")
    user_id = "quick_test_user"
    rag = RAGManager(user_id=user_id)
    print(f"✓ User: {user_id}")
    print(f"✓ Session: {rag.session_id[:12]}...")
    
    # Add documents
    print_subsection("2️⃣  Adding Sample Documents")
    for text, title in QUICK_DOCS:
        rag.add_to_memory(text=text, metadata={"source": f"{title}.txt", "category": title})
        print(f"✓ Added: {title}")
    
    # Search documents (NO LLM CALLS)
    print_subsection("3️⃣  Searching Documents (Without LLM)")
    for query in QUICK_QUERIES:
        docs = rag.search_memory(query, k=2)
        print(f"\n🔍 Query: '{query}'")
        print(f"   Found: {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"   [{i}] {source}: {doc.page_content[:60]}...")
    
    # Session info
    print_subsection("4️⃣  Session Information")
    session = rag.get_session_info()
    print(f"Session ID: {session['session_id'][:12]}...")
    print(f"Created: {session['created_at']}")
    print(f"Conversation Turns: {session['conversation_turns']}")
    print(f"Queries: {session['queries_count']}")
    
    # Knowledge graph
    print_subsection("5️⃣  Knowledge Graph Statistics")
    kg_stats = rag.get_kg_stats()
    print(f"Nodes: {kg_stats['total_nodes']}")
    print(f"Edges: {kg_stats['total_edges']}")
    print(f"Density: {kg_stats['density']:.3f}")
    print(f"Most Accessed: {kg_stats['most_accessed'][:3]}")
    
    # Memory stats
    print_subsection("6️⃣  Memory & Performance Metrics")
    memory_stats = rag.get_memory_stats()
    if memory_stats:
        print(f"Total Retrievals: {memory_stats['total_retrievals']}")
        print(f"Avg Relevance Score: {memory_stats['avg_relevance_score']:.3f}")
        print(f"Avg Response Time: {memory_stats['avg_response_time']:.3f}s")
        print(f"Peak Memory: {memory_stats['peak_memory_mb']:.1f}MB")
    
    # Memory footprint
    print_subsection("7️⃣  System Memory Footprint")
    footprint = rag.get_memory_footprint()
    print(f"RSS Memory: {footprint['rss_mb']:.1f}MB")
    print(f"Memory Usage: {footprint['percent']:.1f}%")
    print(f"Sessions Count: {footprint['sessions_count']}")
    print(f"Documents: {footprint['docs_count']}")
    print(f"KG Nodes: {footprint['kg_nodes']}")
    print(f"KG Edges: {footprint['kg_edges']}")
    
    # Session history
    print_subsection("8️⃣  Conversation History")
    session = rag.session_manager.load_session(rag.session_id)
    if session and session.conversation_history:
        print(f"Total turns: {len(session.conversation_history)}")
        for turn in session.conversation_history[-3:]:
            role = turn['role'].upper()
            content = turn['content'][:50]
            print(f"  {role}: {content}...")
    
    # All sessions
    print_subsection("9️⃣  All Sessions for User")
    all_sessions = rag.get_all_sessions()
    print(f"Total sessions: {len(all_sessions)}")
    for sess in all_sessions[:3]:
        print(f"  • {sess['session_id'][:12]}... ({sess['turns']} turns)")
    
    print_section("✓ QUICK EXAMPLE COMPLETED")
    print("""
Next Steps:
1. Try: python example_with_uploader.py
   (Upload your own PDF/TXT files)

2. Use in your app:
   rag = RAGManager('user_id')
   docs = rag.search_memory('your question')
   
3. Monitor metrics:
   stats = rag.get_memory_stats()
   print(stats)

4. Manage sessions:
   rag.get_all_sessions()
   rag.load_previous_session(session_id)
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
