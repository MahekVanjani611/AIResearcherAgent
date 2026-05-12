"""
Example Integration: Using Enhanced RAG Memory in main.py

This shows how to integrate the new session-based RAG with memory metrics
directly into your Streamlit application.
"""

import streamlit as st
from rag_module import RAGManager
from datetime import datetime

def initialize_rag_session():
    """Initialize RAG manager with session support"""
    if "rag_manager" not in st.session_state:
        rag_manager = RAGManager(
            user_id=st.session_state.user_id,
            session_id=st.session_state.get("current_session_id")
        )
        st.session_state.rag_manager = rag_manager
        st.session_state.current_session_id = rag_manager.session_id
    
    return st.session_state.rag_manager


def show_session_sidebar():
    """Display session management in sidebar"""
    with st.sidebar:
        st.header("📊 Session & Memory")
        
        rag = st.session_state.rag_manager
        
        # Current session info
        session_info = rag.get_session_info()
        if session_info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversation Turns", session_info['conversation_turns'])
            with col2:
                st.metric("Queries", session_info['queries_count'])
            
            if session_info['summary']:
                st.caption("📝 Session Summary")
                st.write(session_info['summary'][:200] + "...")
        
        # Memory statistics
        st.subheader("💾 Memory Metrics")
        memory_stats = rag.get_memory_stats()
        if memory_stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Avg Relevance",
                    f"{memory_stats['avg_relevance_score']:.2f}",
                    delta="Higher is better",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Response Time",
                    f"{memory_stats['avg_response_time']:.3f}s",
                    delta="Lower is better",
                    delta_color="inverse"
                )
            
            st.metric("Total Retrievals", memory_stats['total_retrievals'])
            st.metric("Peak Memory", f"{memory_stats['peak_memory_mb']:.1f}MB")
        
        # Knowledge Graph stats
        st.subheader("🧠 Knowledge Graph")
        kg_stats = rag.get_kg_stats()
        if kg_stats:
            st.metric("Nodes", kg_stats['total_nodes'])
            st.metric("Edges", kg_stats['total_edges'])
            st.metric("Graph Density", f"{kg_stats['density']:.2f}")
            
            if kg_stats['most_accessed']:
                st.caption("Most Accessed Nodes")
                for node, count in kg_stats['most_accessed'][:5]:
                    st.write(f"• {node}: {count} times")
        
        # Session management
        st.divider()
        st.subheader("📂 Sessions")
        
        if st.button("📋 View All Sessions"):
            sessions = rag.get_all_sessions()
            st.session_state.show_sessions = True
        
        if st.button("✏️ Summarize Current"):
            rag.summarize_current_session()
            st.success("Session summarized!")
            st.rerun()
        
        if st.button("🧹 Cleanup Old Sessions (30+ days)"):
            rag.cleanup_old_sessions(days=30)
            st.success("Old sessions cleaned up!")
        
        # Memory footprint
        st.divider()
        st.subheader("🖥️ System Memory")
        footprint = rag.get_memory_footprint()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory Used", f"{footprint['rss_mb']:.1f}MB")
            st.metric("Sessions", footprint['sessions_count'])
        with col2:
            st.metric("Memory %", f"{footprint['percent']:.1f}%")
            st.metric("Docs", footprint['docs_count'])


def show_sessions_modal():
    """Display all sessions in a modal/expander"""
    if st.session_state.get("show_sessions"):
        st.divider()
        st.header("📂 All Sessions")
        
        rag = st.session_state.rag_manager
        sessions = rag.get_all_sessions()
        
        if not sessions:
            st.info("No sessions found")
            return
        
        for session in sessions:
            with st.expander(
                f"🔹 {session['session_id'][:8]}... "
                f"({session['turns']} turns) - "
                f"{session['created_at'].split('T')[0]}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Created", session['created_at'].split('T')[1][:5])
                with col2:
                    st.metric("Last Accessed", session['last_accessed'].split('T')[1][:5])
                with col3:
                    st.metric("Turns", session['turns'])
                
                if session['summary']:
                    st.write(f"**Summary:** {session['summary']}")
                
                # Load session button
                if st.button(
                    f"📂 Load Session {session['session_id'][:8]}",
                    key=f"load_{session['session_id']}"
                ):
                    rag.load_previous_session(session['session_id'])
                    st.session_state.current_session_id = session['session_id']
                    st.success("Session loaded!")
                    st.rerun()


def handle_user_query(query: str):
    """Handle user query with full memory context"""
    rag = st.session_state.rag_manager
    
    with st.spinner("🔍 Searching memory and knowledge graph..."):
        # Search with session context
        docs, session_context = rag.search_with_session_context(query, k=4)
        
        # Display context being used
        if session_context:
            with st.expander("📚 Using Session Context"):
                st.text(session_context[:500])
        
        # Generate answer
        answer = rag.graph_rag_answer(query, k=5)
        
        # Display answer
        st.write(answer)
        
        # Show sources
        if docs:
            with st.expander(f"📖 Sources ({len(docs)} documents)"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Document {i}:**")
                    st.write(doc.page_content[:300] + "...")
                    if doc.metadata:
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")


def extract_session_insights():
    """Extract and display key insights from session"""
    rag = st.session_state.rag_manager
    
    st.subheader("💡 Session Insights")
    
    facts = rag.extract_important_facts()
    
    if facts:
        st.write("**Key Facts Extracted:**")
        for fact in facts:
            st.write(f"• {fact}")
    else:
        st.info("Not enough conversation history to extract insights yet")


# ========================================================================
# Integration in main.py (simplified example)
# ========================================================================

"""
To integrate this into your main.py:

# At the top of main.py
from rag_integration_example import (
    initialize_rag_session,
    show_session_sidebar,
    show_sessions_modal,
    handle_user_query,
    extract_session_insights
)

# In your auth check
if st.session_state.authenticated:
    rag_manager = initialize_rag_session()
    
    # Sidebar
    show_session_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Research", "Sessions", "Insights"])
    
    with tab1:
        user_query = st.text_area("📝 Your Research Question:")
        if user_query and st.button("🔍 Search"):
            handle_user_query(user_query)
    
    with tab2:
        show_sessions_modal()
    
    with tab3:
        extract_session_insights()
"""
