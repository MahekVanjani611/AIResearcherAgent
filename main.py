import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from research_graph import graph, get_graph_visualization
from langchain_core.messages import HumanMessage
import uuid

# Load env variables
load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vanilla CSS)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-running { background-color: #f39c12; color: white; }
    .status-done { background-color: #2ecc71; color: white; }
    .status-error { background-color: #e74c3c; color: white; }
    
    /* Animation for the spinner-like feel */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .pulsing { animation: pulse 1.5s infinite; }
    
    .mermaid {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    tavily_key = st.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
    
    if st.button("Update Keys"):
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("API Keys updated.")

    st.markdown("---")
    st.header("📊 Graph Visualization")
    mermaid_code = get_graph_visualization()
    if mermaid_code:
        st.components.v1.html(
            f"""
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            """,
            height=400,
            scrolling=True
        )

    st.markdown("---")
    st.info("💡 **Tip**: Clear the session state to start a new research project.")
    if st.button("Clear History"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main App ---

st.title("🤖 Coordinated AI Research Assistant")
st.markdown("A multi-agent team collaborating with **Human-in-the-Loop** interrupts for step-by-step verification.")

# Session State Initialization
if "research_log" not in st.session_state:
    st.session_state.research_log = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "interrupted" not in st.session_state:
    st.session_state.interrupted = False

# Input Section
topic = st.text_area("Enter your research topic:", 
                    placeholder="e.g. The impact of LLMs on modern healthcare diagnostic tools...",
                    height=100)

def run_research(resume=False):
    config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 50}
    
    if not resume:
        initial_state = {
            "user_query": topic,
            "plan": [],
            "research_results": [],
            "insights": [],
            "fact_checks": [],
            "critic_score": 0,
            "critic_feedback": "",
            "final_report": "",
            "sources": []
        }
        stream_input = initial_state
        st.session_state.research_log = []
        st.session_state.final_report = None
        st.session_state.sources = []
    else:
        stream_input = None # Resuming from checkpoint
    
    st.session_state.is_running = True
    st.session_state.interrupted = False
    
    try:
        # Use a list to catch the last output before interrupt
        for output in graph.stream(stream_input, config=config, stream_mode="updates"):
            for node_name, node_output in output.items():
                st.session_state.research_log.append(f"✅ **{node_name.capitalize()}** completed.")
                
                if node_name == "writer" and "final_report" in node_output:
                    st.session_state.final_report = node_output["final_report"]
                if "sources" in node_output:
                    st.session_state.sources.extend(node_output["sources"])
        
        # After stream finishes, check if it's actually finished or just interrupted
        snapshot = graph.get_state(config)
        if snapshot.next:
            st.session_state.interrupted = True
            st.session_state.research_log.append(f"⏸️ **Paused**: Review state and click 'Continue' to proceed to `{snapshot.next[0]}`.")
        else:
            st.session_state.interrupted = False
            if st.session_state.final_report:
                st.session_state.research_log.append("🏁 **Finished**: Research report generated.")
        
        st.session_state.is_running = False
        
    except Exception as e:
        st.error(f"Execution Error: {e}")
        st.session_state.is_running = False
        st.session_state.interrupted = False

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Start Deep Research", use_container_width=True, disabled=st.session_state.is_running):
        if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            st.error("Please provide both API keys in the sidebar.")
        elif not topic:
            st.warning("Please enter a research topic.")
        else:
            run_research(resume=False)
            st.rerun()

with col2:
    if st.session_state.interrupted:
        if st.button("➡️ Continue to Next Step", use_container_width=True, type="primary"):
            run_research(resume=True)
            st.rerun()

# --- State Status Section ---
st.markdown("---")
st.subheader("📊 Current Agent State Status")

# Fetch latest state from checkpoint
config = {"configurable": {"thread_id": st.session_state.thread_id}}
snapshot = graph.get_state(config)
state_values = snapshot.values if snapshot.values else {}

# Create status cards
s_col0, s_col1, s_col2, s_col3, s_col4 = st.columns([1.5, 1, 1, 1, 1])

def count_items(val):
    if isinstance(val, list): return len(val)
    return 1 if val else 0

with s_col0:
    next_node = snapshot.next[0] if snapshot.next else "None (Finished)"
    st.metric("⏭️ Next Action", next_node)
with s_col1:
    st.metric("📋 Tasks", count_items(state_values.get("plan")))
with s_col2:
    st.metric("🔍 Sources", len(set(state_values.get("sources", []))))
with s_col3:
    st.metric("💡 Insights", count_items(state_values.get("insights")))
with s_col4:
    score = state_values.get("critic_score", 0)
    st.metric("⭐ Score", f"{score}/10" if score else "N/A")

# Detailed State View (Expander)
with st.expander("👁️ View Partial Data & Edit Plan", expanded=True):
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        current_plan = state_values.get("plan", [])
        st.write("**Research Plan:**")
        
        # If we are paused before researcher, allow editing the plan
        if st.session_state.interrupted and snapshot.next and snapshot.next[0] == "researcher":
            plan_str = "\n".join(current_plan)
            edited_plan_str = st.text_area("Edit tasks (one per line):", value=plan_str, height=150)
            if st.button("💾 Save Edited Plan"):
                new_plan = [t.strip() for t in edited_plan_str.split("\n") if t.strip()]
                # Update the graph state
                graph.update_state(config, {"plan": new_plan})
                st.success("Plan updated successfully!")
                st.rerun()
        else:
            for i, task in enumerate(current_plan):
                st.text(f"{i+1}. {task}")
    
    with d_col2:
        st.write("**Recent Feedback:**")
        st.info(state_values.get("critic_feedback", "No feedback yet."))

# Display Research Progress
st.markdown("---")
if st.session_state.research_log:
    st.subheader("🕵️ Live Research Feed")
    for log in st.session_state.research_log:
        st.info(log)

# Display Results
if st.session_state.final_report:
    st.markdown("---")
    st.subheader("📝 Final Research Report")
    
    tabs = st.tabs(["📖 Report", "🔗 Sources", "📊 Debug State"])
    
    with tabs[0]:
        st.markdown(st.session_state.final_report)
        st.download_button("Download Markdown Report", st.session_state.final_report, file_name="research_report.md")
        
    with tabs[1]:
        if st.session_state.sources:
            st.write("### Credible Sources")
            for src in set(st.session_state.sources):
                st.write(f"- {src}")
        else:
            st.info("No sources recorded.")
            
    with tabs[2]:
        st.json(st.session_state.research_log)
        snapshot = graph.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
        st.write("Current Graph State:")
        st.json(snapshot.values)

else:
    if not st.session_state.research_log and not st.session_state.is_running:
        st.info("Results will appear here after the agents complete their work.")
