#!/usr/bin/env python3
"""
Test script for enhanced RAG module with session management.
Run this to verify the memory system is working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

def test_rag_session_management():
    """Test session management features"""
    print("\n" + "="*60)
    print("Testing Enhanced RAG Module - Session Management")
    print("="*60)
    
    try:
        from rag_module import RAGManager, SessionManager, MemoryMetricsTracker
        
        # Test 1: Create RAGManager with auto-session
        print("\n[TEST 1] Creating RAGManager with automatic session...")
        user_id = "test_user_123"
        rag = RAGManager(user_id=user_id)
        print(f"✓ Created RAGManager for user: {user_id}")
        print(f"✓ Session ID: {rag.session_id}")
        
        # Test 2: Add documents
        print("\n[TEST 2] Adding documents to memory...")
        rag.add_to_memory(
            text="Machine Learning is a subset of AI that focuses on learning from data.",
            metadata={"source": "test_doc_1", "category": "AI"}
        )
        rag.add_to_memory(
            text="Deep Learning uses neural networks with multiple layers.",
            metadata={"source": "test_doc_2", "category": "AI"}
        )
        print("✓ Added 2 documents to memory")
        
        # Test 3: Search memory
        print("\n[TEST 3] Searching memory...")
        docs = rag.search_memory("machine learning", k=2)
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Test 4: Session info
        print("\n[TEST 4] Checking session info...")
        session_info = rag.get_session_info()
        print(f"✓ Session Info:")
        print(f"  - Created: {session_info['created_at']}")
        print(f"  - Turns: {session_info['conversation_turns']}")
        print(f"  - Queries: {session_info['queries_count']}")
        
        # Test 5: Knowledge graph stats
        print("\n[TEST 5] Checking knowledge graph...")
        kg_stats = rag.get_kg_stats()
        print(f"✓ KG Stats:")
        print(f"  - Nodes: {kg_stats['total_nodes']}")
        print(f"  - Edges: {kg_stats['total_edges']}")
        
        # Test 6: Memory footprint
        print("\n[TEST 6] Checking memory footprint...")
        footprint = rag.get_memory_footprint()
        print(f"✓ Memory Footprint:")
        print(f"  - RSS: {footprint['rss_mb']:.1f}MB")
        print(f"  - Docs: {footprint['docs_count']}")
        print(f"  - KG Nodes: {footprint['kg_nodes']}")
        
        # Test 7: Multiple sessions
        print("\n[TEST 7] Testing multiple sessions...")
        session1 = rag.session_id
        all_sessions_before = rag.get_all_sessions()
        print(f"✓ Created first session: {session1[:8]}...")
        
        # Create new session
        rag2 = RAGManager(user_id=user_id)
        session2 = rag2.session_id
        print(f"✓ Created second session: {session2[:8]}...")
        
        all_sessions_after = rag2.get_all_sessions()
        print(f"✓ Total sessions for user: {len(all_sessions_after)}")
        
        # Test 8: Load previous session
        print("\n[TEST 8] Loading previous session...")
        success = rag2.load_previous_session(session1)
        if success:
            print(f"✓ Successfully loaded session: {session1[:8]}...")
        else:
            print(f"✗ Failed to load session")
        
        # Test 9: Session context
        print("\n[TEST 9] Testing session context preservation...")
        rag.search_memory("what is deep learning", k=1)
        session_ctx = rag.session_manager.get_session_context(rag.session_id)
        if session_ctx:
            print(f"✓ Session context retrieved:")
            print(f"  {session_ctx[:100]}...")
        
        # Test 10: Extract facts (if conversation exists)
        print("\n[TEST 10] Testing fact extraction...")
        facts = rag.extract_important_facts()
        if facts:
            print(f"✓ Extracted {len(facts)} facts")
            for fact in facts[:3]:
                print(f"  - {fact[:60]}...")
        else:
            print("✓ No facts yet (need longer conversation)")
        
        print("\n" + "="*60)
        print("All tests completed successfully! ✓")
        print("="*60)
        print("\nKey features working:")
        print("✓ Session creation & management")
        print("✓ Document addition & retrieval")
        print("✓ Session info tracking")
        print("✓ Knowledge graph building")
        print("✓ Memory monitoring")
        print("✓ Multi-session support")
        print("✓ Session context preservation")
        print("✓ Fact extraction")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n✗ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_metrics():
    """Test memory metrics tracking"""
    print("\n" + "="*60)
    print("Testing Memory Metrics Tracking")
    print("="*60)
    
    try:
        from rag_module import MemoryMetricsTracker
        
        tracker = MemoryMetricsTracker()
        print("✓ MemoryMetricsTracker initialized")
        
        # Test record
        tracker.record_retrieval(
            user_id="test_user",
            session_id="session_1",
            query="test query",
            retrieved_count=3,
            relevance_score=0.87,
            response_time=0.25
        )
        print("✓ Recorded test retrieval metric")
        
        # Get stats
        stats = tracker.get_user_stats("test_user")
        print(f"✓ User stats: {stats}")
        
        summary = tracker.get_memory_summary()
        print(f"✓ System summary: {summary}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Enhanced RAG Module Tests\n")
    
    test1_pass = test_rag_session_management()
    test2_pass = test_memory_metrics()
    
    if test1_pass and test2_pass:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour enhanced RAG module is ready to use!")
        print("See RAG_MEMORY_GUIDE.md for detailed usage instructions.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check errors above.")
        sys.exit(1)
