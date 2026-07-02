import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from research_graph import graph, get_graph_visualization
from langchain_core.messages import HumanMessage
import uuid
import time
from datetime import datetime
from typing import Dict, Optional
import re
from auth_manager import AuthManager
from chat_manager import ChatManager, VectorDBMemory
from mem0_integration import Mem0Manager
from components import apply_custom_theme, render_badge, render_progress_bar, render_section_header
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit.components.v1 as components

# Load env variables
load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization (Must be before any other code)
if "research_log" not in st.session_state:
    st.session_state.research_log = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
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
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "allow_interrupts_flag" not in st.session_state:
    st.session_state.allow_interrupts_flag = True
if "streaming_active" not in st.session_state:
    st.session_state.streaming_active = False
if "session_token" not in st.session_state:
    st.session_state.session_token = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = None
if "auth_manager" not in st.session_state:
    st.session_state.auth_manager = AuthManager()
if "mem0_manager" not in st.session_state:
    st.session_state.mem0_manager = None
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = None  # Track for Mem0 pairing
if "progress_completed_stages" not in st.session_state:
    st.session_state.progress_completed_stages = []

RESEARCH_STAGE_ORDER = [
    "planner",
    "researcher",
    "analyst",
    "fact_checker",
    "critic",
    "writer",
]

RESEARCH_STAGE_LABELS = {
    "planner": "Planner",
    "researcher": "Researcher",
    "analyst": "Analyst",
    "fact_checker": "Fact Checker",
    "critic": "Critic",
    "writer": "Writer",
}


def get_stage_label(stage_key: str) -> str:
    return RESEARCH_STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").title())


def reset_research_progress():
    st.session_state.progress_completed_stages = []


def record_stage_completion(stage_key: str):
    if stage_key == "planner" and st.session_state.progress_completed_stages:
        if st.session_state.progress_completed_stages[-1] != "writer":
            st.session_state.progress_completed_stages = []

    if stage_key in RESEARCH_STAGE_ORDER and stage_key not in st.session_state.progress_completed_stages:
        st.session_state.progress_completed_stages.append(stage_key)


def render_research_progress(progress_slot=None, current_stage: Optional[str] = None, next_stage: Optional[str] = None):
    """Render a clear progress summary for the research pipeline."""
    total_stages = len(RESEARCH_STAGE_ORDER)
    completed_stages = st.session_state.progress_completed_stages

    target = progress_slot if progress_slot is not None else st
    with target.container():
        render_section_header("Research Progress", "📈")
        render_progress_bar(len(completed_stages), total_stages, "Pipeline completion")

        summary_bits = []
        if completed_stages:
            summary_bits.append("Completed: " + " → ".join(get_stage_label(stage) for stage in completed_stages))
        if current_stage:
            summary_bits.append(f"Current stage: {get_stage_label(current_stage)}")
        if next_stage:
            summary_bits.append(f"Next stage: {get_stage_label(next_stage)}")

        if summary_bits:
            for line in summary_bits:
                st.caption(line)
        elif st.session_state.is_running:
            st.caption("The pipeline will update here as each agent stage finishes.")
        else:
            st.info("Run a research job to see the planner, researcher, analyst, fact-checker, critic, and writer stages fill in here.")


def render_mermaid_graph(mermaid_code: str):
    """Render Mermaid safely inside the Streamlit sidebar."""
    cleaned_code = mermaid_code.strip()
    cleaned_code = re.sub(r"^```mermaid\s*", "", cleaned_code)
    cleaned_code = re.sub(r"\s*```$", "", cleaned_code)

    html_content = f"""
    <div style="background-color: white; padding: 12px; border-radius: 12px; overflow-x: auto;">
        <div class="mermaid">
{cleaned_code}
        </div>
    </div>
    <script>
        if (!window.mermaidInitialized) {{
            mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', theme: 'default' }});
            window.mermaidInitialized = true;
        }}
        mermaid.contentLoaded();
    </script>
    """
    components.html(
        f"""
        <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            </head>
            <body style="margin: 0; padding: 0; background: white;">
                {html_content}
            </body>
        </html>
        """,
        height=420,
        scrolling=True,
    )


# Apply the shared theme before local page styling so component styles are available.
apply_custom_theme()

# ============================================================================
# AUTHENTICATION & LOGIN
# ============================================================================

auth = st.session_state.auth_manager

# Check if user is logged in
if not st.session_state.session_token:
    # Show login/register page
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 50px auto;
            padding: 40px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AI Research Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        username_login = st.text_input("Username", key="login_user")
        password_login = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🔓 Login", use_container_width=True, type="primary"):
            if username_login and password_login:
                success, msg, token = auth.login_user(username_login, password_login)
                if success:
                    st.session_state.session_token = token
                    st.session_state.current_user = username_login
                    st.session_state.chat_manager = ChatManager(username_login)
                    st.session_state.mem0_manager = Mem0Manager(user_id=username_login)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please enter username and password")
    
    with col2:
        st.subheader("Register")
        username_reg = st.text_input("New Username", key="reg_user")
        email_reg = st.text_input("Email (optional)", key="reg_email")
        password_reg = st.text_input("New Password", type="password", key="reg_pass")
        password_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("📝 Register", use_container_width=True):
            if not username_reg or not password_reg:
                st.warning("Username and password required")
            elif password_reg != password_confirm:
                st.error("Passwords do not match")
            else:
                success, msg = auth.register_user(username_reg, password_reg, email_reg)
                if success:
                    st.success(msg)
                    st.info("✅ Now you can login!")
                else:
                    st.error(msg)
    
    st.stop()

else:
    # User is logged in - verify session
    is_valid, username = auth.verify_session(st.session_state.session_token)
    if not is_valid:
        st.error("❌ Session expired. Please login again.")
        st.session_state.session_token = None
        st.session_state.current_user = None
        st.rerun()
    
    st.session_state.current_user = username

# Custom Styling with Chat Interface (Vanilla CSS)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Chat Message Styles */
    .chat-message {
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    .chat-message.agent {
        background-color: rgba(30, 144, 255, 0.1);
        border-left-color: #1e90ff;
    }
    
    .chat-message.user {
        background-color: rgba(34, 139, 34, 0.1);
        border-left-color: #22aa22;
    }
    
    .chat-message.system {
        background-color: rgba(255, 165, 0, 0.1);
        border-left-color: #ffa500;
    }
    
    .chat-timestamp {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.3rem;
    }
    
    .chat-role {
        font-weight: bold;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
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
    
    .interrupt-flag {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .interrupt-flag.enabled {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
    }
    
    .interrupt-flag.disabled {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
    }
    
    .streaming-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #f39c12;
        animation: pulse 1s infinite;
        margin-right: 0.5rem;
    }
    
    .message-input-container {
        position: sticky;
        bottom: 0;
        background-color: #0e1117;
        padding: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: User Menu & Chat History
with st.sidebar:
    st.header(f"👤 {st.session_state.current_user}")
    
    # Chat Management
    col_new, col_config = st.columns(2)
    
    with col_new:
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            chat_id = st.session_state.chat_manager.create_new_chat(title=f"Chat {datetime.now().strftime('%H:%M')}")
            st.session_state.current_chat_id = chat_id
            st.session_state.chat_messages = []
            st.session_state.final_report = None
            st.session_state.sources = []
            st.rerun()
    
    with col_config:
        if st.button("⚙️ Settings", use_container_width=True, key="settings_btn"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
    
    st.markdown("---")
    
    # Settings Panel
    if st.session_state.get("show_settings"):
        st.subheader("⚙️ Settings")
        allow_interrupts = st.checkbox("🛑 Allow Interrupts", value=st.session_state.allow_interrupts_flag)
        st.session_state.allow_interrupts_flag = allow_interrupts
        
        st.markdown("---")
        st.subheader("API Keys")
        google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
        tavily_key = st.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
        
        if st.button("Update Keys"):
            os.environ["GOOGLE_API_KEY"] = google_key
            os.environ["TAVILY_API_KEY"] = tavily_key
            st.success("API Keys updated.")
    
    # Chat History (Like ChatGPT)
    st.markdown("---")
    st.subheader("📝 Recent Chats")
    
    all_chats = st.session_state.chat_manager.get_all_chats()
    
    if not all_chats:
        st.info("No chats yet. Create a new one!")
    else:
        for chat in all_chats[:15]:  # Show last 15 chats
            col_title, col_delete = st.columns([4, 1])
            
            with col_title:
                if st.button(
                    f"• {chat['title'][:25]}{'...' if len(chat['title']) > 25 else ''}",
                    key=f"chat_{chat['chat_id']}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat['chat_id']
                    # Load chat messages
                    loaded_chat = st.session_state.chat_manager.load_chat(chat['chat_id'])
                    st.session_state.chat_messages = loaded_chat.get("messages", [])
                    st.rerun()
            
            with col_delete:
                if st.button("🗑️", key=f"delete_{chat['chat_id']}", help="Delete chat"):
                    st.session_state.chat_manager.delete_chat(chat['chat_id'])
                    if st.session_state.current_chat_id == chat['chat_id']:
                        st.session_state.current_chat_id = None
                        st.session_state.chat_messages = []
                    st.rerun()
    
    # Graph Visualization
    st.markdown("---")
    st.subheader("📊 Graph Visualization")
    mermaid_code = get_graph_visualization()
    if mermaid_code:
        render_mermaid_graph(mermaid_code)
    else:
        st.info("Graph visualization will appear here after research.")
    
    # Logout
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        auth.logout_user(st.session_state.session_token)
        st.session_state.session_token = None
        st.session_state.current_user = None
        st.session_state.chat_manager = None
        st.rerun()

# --- Main App ---

st.title("🤖 AI Research Assistant")
st.markdown("Collaborative research with multi-agent AI teams")

# Initialize chat if none exists
if not st.session_state.current_chat_id:
    st.session_state.current_chat_id = st.session_state.chat_manager.create_new_chat()
    st.session_state.chat_messages = []

# Get current chat info
current_chat = st.session_state.chat_manager.load_chat(st.session_state.current_chat_id)
if current_chat:
    st.caption(f"Chat: {current_chat['title']} | Messages: {len(current_chat['messages'])}")

# Display Interrupt Flag Status
col_interrupt = st.columns(3)
with col_interrupt[0]:
    if st.session_state.allow_interrupts_flag:
        st.markdown('<div class="interrupt-flag enabled">✅ Interrupts: ENABLED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="interrupt-flag disabled">🚫 Interrupts: DISABLED</div>', unsafe_allow_html=True)

# Display Memory Stats (Sidebar)
with st.sidebar:
    if st.session_state.mem0_manager:
        st.divider()
        st.subheader("🧠 Memory Stats")
        mem_stats = st.session_state.mem0_manager.get_memory_stats()
        col_mem1, col_mem2 = st.columns(2)
        with col_mem1:
            st.metric("Cached Memories", mem_stats['cached_memories'])
        with col_mem2:
            st.metric("API Active", "✅" if mem_stats['api_configured'] else "❌")
        
        # Add memory search
        with st.expander("🔍 Search Memories", expanded=False):
            search_query = st.text_input("Search your memories:", key="mem_search")
            if search_query:
                results = st.session_state.mem0_manager.search_memories(search_query, limit=3)
                if results:
                    for i, result in enumerate(results, 1):
                        st.write(f"**Result {i}:** {str(result)[:200]}...")
                else:
                    st.info("No memories found matching this query.")

# Research Topic Input
st.markdown("---")
st.subheader("🔍 Research Topic")
topic = st.text_area("Enter your research topic:", 
                    placeholder="e.g. The impact of LLMs on modern healthcare diagnostic tools...",
                    height=100)

# Control Buttons
col1, col2, col3 = st.columns([1, 1, 1])

# Helper function to add messages and save to persistent storage
def add_message_to_chat(role: str, content: str, agent: str = None, metadata: Dict = None):
    """Add message to both session state and persistent storage with Mem0 integration"""
    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    
    if role == "user":
        msg["user"] = st.session_state.current_user
        st.session_state.last_user_message = content  # Store for Mem0 pairing
    elif role == "agent":
        msg["agent"] = agent or "Agent"
    
    st.session_state.chat_messages.append(msg)
    
    # Save to persistent storage
    if st.session_state.chat_manager:
        st.session_state.chat_manager.add_message_to_chat(
            st.session_state.current_chat_id,
            role=role,
            content=content,
            metadata=metadata
        )
    
    # Store in Mem0 for long-term memory (pair user message with assistant response)
    if st.session_state.mem0_manager:
        try:
            if role == "assistant" and st.session_state.last_user_message:
                # Store the Q&A pair when we have both user message and response
                st.session_state.mem0_manager.add_memory_from_conversation(
                    user_message=st.session_state.last_user_message,
                    assistant_response=content
                )
                st.session_state.last_user_message = None  # Reset after storing
        except Exception as e:
            logger.debug(f"Mem0 storage note: {e}")  # Debug level, not critical

def store_in_mem0(user_question: str, assistant_answer: str):
    """
    Direct helper to store Q&A pair in Mem0
    Use when you have both user question and response ready
    """
    if st.session_state.mem0_manager:
        try:
            st.session_state.mem0_manager.add_memory_from_conversation(
                user_message=user_question,
                assistant_response=assistant_answer
            )
        except Exception as e:
            logger.debug(f"Mem0 note: {e}")


def build_counter_question_response(counter_question: str) -> str:
    """Answer a follow-up question using the current chat context plus saved memory."""
    context_sections = []

    if st.session_state.chat_manager and st.session_state.current_chat_id:
        chat_context = st.session_state.chat_manager.get_chat_context(st.session_state.current_chat_id, last_n_messages=12)
        if chat_context:
            context_sections.append(f"Recent chat context:\n{chat_context}")

    if st.session_state.final_report:
        context_sections.append(f"Latest research report:\n{st.session_state.final_report[:8000]}")

    memory_hits = []
    if st.session_state.mem0_manager:
        try:
            memory_context = st.session_state.mem0_manager.get_user_context()
            if memory_context and "No user memories found." not in memory_context and "Error retrieving user context." not in memory_context:
                context_sections.append(memory_context)

            memory_hits = st.session_state.mem0_manager.search_memories(counter_question, limit=5)
        except Exception as e:
            logger.debug(f"Mem0 context lookup note: {e}")

    if memory_hits:
        hit_lines = []
        for hit in memory_hits:
            if isinstance(hit, dict):
                hit_lines.append(
                    f"- User: {hit.get('user_message', '')}\n  Assistant: {hit.get('assistant_response', '')}"
                )
            else:
                hit_lines.append(f"- {str(hit)}")
        context_sections.append("Relevant memory hits:\n" + "\n".join(hit_lines))

    prompt = f"""Answer the follow-up question using the provided chat context, research report, and saved memory.

Rules:
- Be concise and directly address the question.
- If the context is insufficient, say what is missing instead of inventing details.
- Prefer the user's prior saved preferences, past questions, and recent research context.

Follow-up question:
{counter_question}

Context:
{chr(10).join(context_sections) if context_sections else 'No additional context available.'}
"""

    try:
        if not os.getenv("GOOGLE_API_KEY"):
            return "I can respond, but I need the Google API key to generate a context-aware answer."

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        response = llm.invoke([
            SystemMessage(content="You answer follow-up questions using prior chat context and saved memory."),
            HumanMessage(content=prompt),
        ])
        return str(response.content)
    except Exception as e:
        logger.error(f"Counter question response generation failed: {e}")
        fallback = "I could not generate a model response, but I did load your saved chat and memory context. Please retry once the model is available."
        return fallback

def run_research_with_interrupts(config, initial_state):
    """Pipeline with interrupts enabled - pauses at each node"""
    st.session_state.is_running = True
    st.session_state.streaming_active = True
    st.session_state.interrupted = False
    progress_slot = st.empty()
    reset_research_progress()
    render_research_progress(progress_slot, current_stage="planner")
    
    try:
        # Process all nodes but pause between them
        all_outputs = []
        for output in graph.stream(initial_state, config=config, stream_mode="updates"):
            all_outputs.append(output)
            for node_name, node_output in output.items():
                record_stage_completion(node_name)
                agent_msg = f"✅ **{get_stage_label(node_name)}** completed."
                st.session_state.chat_messages.append({
                    "role": "agent",
                    "content": agent_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": node_name
                })
                
                st.session_state.research_log.append(agent_msg)
                render_research_progress(progress_slot, current_stage=node_name)
                
                if node_name == "writer" and "final_report" in node_output:
                    st.session_state.final_report = node_output["final_report"]
                if "sources" in node_output:
                    st.session_state.sources.extend(node_output.get("sources", []))
        
        # After all nodes complete, check final state
        snapshot = graph.get_state(config)
        final_state_values = snapshot.values if snapshot.values else {}
        
        # Check if pipeline is complete
        if snapshot.next:
            st.session_state.interrupted = True
            pause_msg = f"⏸️ **Paused**: Review state and click 'Continue' to proceed to `{snapshot.next[0]}`."
            st.session_state.chat_messages.append({
                "role": "system",
                "content": pause_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.research_log.append(pause_msg)
        else:
            st.session_state.interrupted = False
            if st.session_state.final_report:
                finish_msg = "🏁 **Finished**: Research report generated."
                st.session_state.chat_messages.append({
                    "role": "system",
                    "content": finish_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.session_state.research_log.append(finish_msg)
                render_research_progress(progress_slot, current_stage="writer", next_stage=None)
                
                # Store research Q&A in Mem0
                store_in_mem0(
                    user_question=topic,
                    assistant_answer=st.session_state.final_report
                )
                
                # Display final results
                st.markdown("---")
                st.markdown("### 📋 Research Report")
                st.markdown(st.session_state.final_report)
                
                # Display insights
                insights = final_state_values.get("insights", [])
                if insights:
                    st.markdown("---")
                    st.markdown("### 💡 Key Insights")
                    for insight in insights:
                        st.info(insight)
                
                # Display sources
                if st.session_state.sources:
                    st.markdown("---")
                    st.markdown("### 🔗 Sources")
                    unique_sources = list(set(st.session_state.sources))
                    for idx, src in enumerate(unique_sources, 1):
                        st.write(f"**{idx}.** {src}")
        
        st.session_state.is_running = False
        st.session_state.streaming_active = False
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Execution Error: {str(e)}"
        st.session_state.chat_messages.append({
            "role": "system",
            "content": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.error(error_msg)
        st.error(traceback.format_exc())
        st.session_state.is_running = False
        st.session_state.streaming_active = False


def run_research_without_interrupts(config, initial_state):
    """Pipeline without interrupts - runs all nodes silently, streams final output"""
    st.session_state.is_running = True
    st.session_state.streaming_active = True
    st.session_state.interrupted = False
    progress_slot = st.empty()
    reset_research_progress()
    render_research_progress(progress_slot, current_stage="planner")
    
    try:
        # Add system message about processing
        processing_msg = "🔄 Processing your research query (all stages running)..."
        st.session_state.chat_messages.append({
            "role": "system",
            "content": processing_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display the chat messages so far
        chat_display = st.container()
        with chat_display:
            for msg in st.session_state.chat_messages:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-role">👤 User: {msg.get('user', 'researcher')}</div>
                        <div>{content}</div>
                        <div class="chat-timestamp">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif role == "system":
                    st.markdown(f"""
                    <div class="chat-message system">
                        <div class="chat-role">⚙️ System</div>
                        <div>{content}</div>
                        <div class="chat-timestamp">{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Run all stages silently without interruption - keep resuming until complete
        final_state_values = {}
        stream_input = initial_state
        
        while True:
            # Stream the current stage
            has_output = False
            for output in graph.stream(stream_input, config=config, stream_mode="updates"):
                has_output = True
                for node_name, node_output in output.items():
                    record_stage_completion(node_name)
                    if node_name == "writer" and "final_report" in node_output:
                        st.session_state.final_report = node_output["final_report"]
                    if "sources" in node_output:
                        st.session_state.sources.extend(node_output.get("sources", []))
                    final_state_values.update(node_output)
                    render_research_progress(progress_slot, current_stage=node_name)
            
            # Check if there are more nodes to run
            snapshot = graph.get_state(config)
            if not snapshot.next or not has_output:
                # Pipeline is complete
                if snapshot.values:
                    final_state_values.update(snapshot.values)
                break
            
            # Resume from the next node
            stream_input = None
        
        # Stream the final report like ChatGPT
        st.markdown("---")
        render_research_progress()
        
        if st.session_state.final_report:
            # Stream the report with typing effect
            st.markdown("### 📋 Research Report")
            report_placeholder = st.empty()
            streamed_content = ""
            
            for i in range(0, len(st.session_state.final_report), 15):
                streamed_content += st.session_state.final_report[i:i+15]
                report_placeholder.markdown(streamed_content + "▌")  # Cursor effect
                time.sleep(0.01)  # Streaming delay
            
            # Final display without cursor
            report_placeholder.markdown(streamed_content)
            
            # Add to chat history
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": st.session_state.final_report,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        # Display key insights
        st.markdown("---")
        insights = final_state_values.get("insights", [])
        if insights:
            st.markdown("### 💡 Key Insights")
            for insight in insights:
                st.info(insight)
        
        # Display sources
        st.markdown("---")
        sources = st.session_state.sources
        if sources:
            st.markdown("### 🔗 Research Sources")
            unique_sources = list(set(sources))
            for idx, src in enumerate(unique_sources, 1):
                st.write(f"**{idx}.** {src}")
        
        # Display fact checks if available
        fact_checks = final_state_values.get("fact_checks", [])
        if fact_checks:
            st.markdown("---")
            st.markdown("### ✅ Fact Checks")
            for check in fact_checks:
                st.write(f"- {check}")
        
        # Display critic feedback if available
        critic_feedback = final_state_values.get("critic_feedback", "")
        critic_score = final_state_values.get("critic_score", 0)
        if critic_feedback or critic_score:
            st.markdown("---")
            st.markdown("### 🎯 Quality Assessment")
            st.write(f"**Score:** {critic_score}/10")
            if critic_feedback:
                st.info(critic_feedback)
        
        # Show completion message
        finish_msg = "✅ **Research Complete**: All stages processed. Final report streamed above."
        st.session_state.chat_messages.append({
            "role": "system",
            "content": finish_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.research_log.append(finish_msg)
        
        # Store research Q&A in Mem0
        if st.session_state.final_report:
            store_in_mem0(
                user_question=topic,
                assistant_answer=st.session_state.final_report
            )
        
        st.session_state.is_running = False
        st.session_state.streaming_active = False
        st.success("✅ Research completed and streamed!")
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Execution Error: {str(e)}"
        st.session_state.chat_messages.append({
            "role": "system",
            "content": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.error(error_msg)
        st.error(traceback.format_exc())
        st.session_state.is_running = False
        st.session_state.streaming_active = False


def run_research(resume=False):
    """Main research runner - routes to interrupt or non-interrupt pipeline"""
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
        st.session_state.research_log = []
        st.session_state.chat_messages = []
        st.session_state.final_report = None
        st.session_state.sources = []
        
        # Add initial user message to chat (both in memory and persistent storage)
        user_msg = {
            "role": "user",
            "content": topic,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": st.session_state.current_user
        }
        st.session_state.chat_messages.append(user_msg)
        
        # Save to persistent storage
        st.session_state.chat_manager.add_message_to_chat(
            st.session_state.current_chat_id,
            role="user",
            content=topic
        )
        
        # Route based on interrupt flag
        if st.session_state.allow_interrupts_flag:
            # Pipeline with interrupts enabled
            run_research_with_interrupts(config, initial_state)
        else:
            # Pipeline without interrupts - streams like ChatGPT
            run_research_without_interrupts(config, initial_state)
    else:
        # Resume from checkpoint with interrupts
        run_research_with_interrupts(config, None)

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
    if st.session_state.interrupted and st.session_state.allow_interrupts_flag:
        if st.button("➡️ Continue", use_container_width=True, type="primary"):
            run_research(resume=True)
            st.rerun()

with col3:
    if st.session_state.is_running: st.markdown('<div class="streaming-indicator"></div><span>Streaming pipeline...</span>', unsafe_allow_html=True)

# Progress overview
st.markdown("---")
render_research_progress()

# --- Chat Interface Section ---
st.markdown("---")
st.subheader("💬 Chat & Streaming Messages")

# Display chat messages
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="chat-role">👤 User: {msg.get('user', 'researcher')}</div>
                <div>{content}</div>
                <div class="chat-timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        elif role == "agent":
            agent_name = msg.get("agent", "agent").capitalize()
            st.markdown(f"""
            <div class="chat-message agent">
                <div class="chat-role">🤖 {agent_name}</div>
                <div>{content}</div>
                <div class="chat-timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:  # system
            st.markdown(f"""
            <div class="chat-message system">
                <div class="chat-role">⚙️ System</div>
                <div>{content}</div>
                <div class="chat-timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

# Counter Question Input (Only when interrupts enabled and paused or running)
if st.session_state.allow_interrupts_flag and (st.session_state.interrupted or st.session_state.is_running):
    st.markdown("---")
    st.subheader("❓ Ask Counter Question")
    
    counter_question = st.text_input(
        "Ask a follow-up or counter question (uses saved memory + recent chat context):",
        placeholder="e.g., Can you elaborate on the limitations of this approach?",
        key="counter_question_input"
    )
    
    if st.button("📤 Send Counter Question", use_container_width=True):
        if counter_question.strip():
            # Add user's counter question to chat
            st.session_state.chat_messages.append({
                "role": "user",
                "content": counter_question,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user": st.session_state.current_user
            })

            system_response = build_counter_question_response(counter_question)
            st.session_state.chat_messages.append({
                "role": "system",
                "content": system_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Store follow-up Q&A in Mem0 so later questions can reuse the context
            store_in_mem0(counter_question, system_response)
            
            st.success("Counter question added to research context!")
            st.rerun()
        else:
            st.warning("Please enter a question.")
elif not st.session_state.allow_interrupts_flag and not st.session_state.is_running:
    st.info("💡 **Note**: Interrupts are disabled. Research runs continuously without pauses. Enable interrupts in the sidebar to use counter questions.")

# --- State Status Section ---
st.markdown("---")
st.subheader("📊 Current Agent State Status")
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
