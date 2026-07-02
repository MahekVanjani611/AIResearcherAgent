# AI Research Assistant - Enhancement Guide

## Part 1: MCP (Model Context Protocol) Server Integration

### What is MCP?
Model Context Protocol (MCP) is a standard for extending AI capabilities with external tools. It allows:
- **External tool integration** (APIs, databases, web services)
- **Real-time data access** (weather, stock prices, live web content)
- **Custom capabilities** (domain-specific tools)
- **System interoperability** (Cursor, VS Code, other AI tools)

### Recommended MCP Tools for Your Project

#### 1. **Web Search & Scraping Tool**
```python
# mcp_tools/web_search_tool.py
from pydantic import BaseModel, Field
from typing import Optional
import requests
from bs4 import BeautifulSoup

class WebSearchRequest(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results")

class WebScrapingRequest(BaseModel):
    url: str = Field(description="URL to scrape")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for content")

def web_search(request: WebSearchRequest) -> dict:
    """Search the web and return results with snippets"""
    # Integrate with Tavily API (already in use)
    pass

def web_scrape(request: WebScrapingRequest) -> dict:
    """Scrape webpage content"""
    try:
        response = requests.get(request.url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        if request.css_selector:
            content = soup.select(request.css_selector)
        else:
            content = soup.find_all(['p', 'h1', 'h2', 'h3'])
        return {"success": True, "content": [str(c) for c in content]}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### 2. **Database Query Tool**
```python
# mcp_tools/database_tool.py
from typing import Optional, List
import sqlite3

class DatabaseQueryRequest(BaseModel):
    query: str = Field(description="SQL query")
    database: str = Field(default="research.db", description="Database file")

def execute_query(request: DatabaseQueryRequest) -> dict:
    """Execute SQL queries on research database"""
    try:
        conn = sqlite3.connect(request.database)
        cursor = conn.cursor()
        cursor.execute(request.query)
        results = cursor.fetchall()
        conn.close()
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### 3. **Document Processing Tool**
```python
# mcp_tools/document_tool.py
from typing import List
import PyPDF2
import json

class DocumentProcessRequest(BaseModel):
    file_path: str = Field(description="Path to document")
    operation: str = Field(description="extract|summarize|analyze")

def process_document(request: DocumentProcessRequest) -> dict:
    """Process various document formats"""
    if request.file_path.endswith('.pdf'):
        return extract_pdf(request.file_path)
    elif request.file_path.endswith('.txt'):
        return extract_text(request.file_path)
    elif request.file_path.endswith('.json'):
        return extract_json(request.file_path)
```

#### 4. **Real-Time Data Tool**
```python
# mcp_tools/realtime_tool.py
# Access: weather, stock prices, news, social media

class RealtimeDataRequest(BaseModel):
    data_type: str = Field(description="weather|stocks|news|crypto")
    query: str = Field(description="Search term or location")

def get_realtime_data(request: RealtimeDataRequest) -> dict:
    """Fetch real-time data from various sources"""
    if request.data_type == "weather":
        return get_weather_data(request.query)
    elif request.data_type == "stocks":
        return get_stock_data(request.query)
    elif request.data_type == "news":
        return get_news_data(request.query)
```

### MCP Server Implementation Structure

```
mcp_server/
├── __init__.py
├── server.py                 # Main MCP server
├── tools/
│   ├── __init__.py
│   ├── web_search.py
│   ├── database.py
│   ├── documents.py
│   ├── realtime.py
│   └── code_execution.py
├── resources/
│   ├── research_db.py       # DB schema
│   └── config.py
└── tests/
    └── test_tools.py
```

### Step-by-Step MCP Implementation

#### Step 1: Create MCP Server
```python
# mcp_server/server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

server = Server("research-assistant-mcp")

# Register tools
@server.define_tool(
    name="web_search",
    description="Search the web for information",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer"}
        },
        "required": ["query"]
    }
)
async def web_search_tool(query: str, num_results: int = 5):
    # Implementation
    pass

@server.define_tool(
    name="database_query",
    description="Query the research database",
    inputSchema={...}
)
async def database_tool(query: str):
    # Implementation
    pass

if __name__ == "__main__":
    server.run()
```

#### Step 2: Integrate MCP into Research Graph
```python
# research_graph.py (modified)
from mcp.client import Client

mcp_client = Client("research-assistant-mcp")

def researcher_node(state):
    """Use MCP tools for research"""
    # Can now call:
    # - mcp_client.call_tool("web_search", {"query": "..."})
    # - mcp_client.call_tool("database_query", {"query": "..."})
    pass
```

#### Step 3: Setup MCP Requirements
```
pip install mcp
pip install pydantic
pip install requests beautifulsoup4
pip install PyPDF2
pip install yfinance
pip install newsapi-python
```

---

## Part 2: Advanced UI/UX Improvements

### 1. **Enhanced CSS Styling**

Create a new file: `assets/styles.css`

```python
# assets/custom_theme.py
CUSTOM_CSS = """
<style>
    /* Root Variables */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border: #475569;
        --radius: 12px;
        --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, #1a1f3a 100%);
    }

    /* Chat Container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px;
        background: var(--surface);
        border-radius: var(--radius);
        border: 1px solid var(--border);
    }

    /* Message Bubbles - Enhanced */
    .message-bubble {
        padding: 12px 16px;
        border-radius: var(--radius);
        margin-bottom: 8px;
        animation: slideIn 0.3s ease-out;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all var(--transition);
    }

    .message-bubble:hover {
        transform: translateX(2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    .message-user {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        margin-left: auto;
        max-width: 80%;
        border-bottom-right-radius: 4px;
    }

    .message-agent {
        background: var(--surface-light);
        color: var(--text-primary);
        margin-right: auto;
        max-width: 80%;
        border: 1px solid var(--border);
        border-bottom-left-radius: 4px;
    }

    .message-system {
        background: rgba(10, 176, 176, 0.1);
        color: #00d4d4;
        margin: 8px 0;
        border-left: 3px solid #00d4d4;
        border-radius: 0;
    }

    /* Animation */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Sidebar Improvements */
    .stSidebar {
        background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
        border-right: 1px solid var(--border);
    }

    .sidebar-section {
        padding: 16px;
        background: var(--surface-light);
        border-radius: var(--radius);
        margin-bottom: 12px;
        border: 1px solid var(--border);
    }

    /* Buttons - Modern */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: var(--radius);
        font-weight: 600;
        transition: all var(--transition);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Input Fields */
    .stTextInput input, .stTextArea textarea {
        background: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius) !important;
        transition: all var(--transition);
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all var(--transition);
    }

    .card:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: var(--success);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--warning);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: var(--danger);
    }

    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--surface);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .message-user, .message-agent {
            max-width: 95%;
        }
        .stSidebar {
            width: 100% !important;
        }
    }
</style>
"""

def apply_custom_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
```

### 2. **Enhanced Components**

```python
# components/enhanced_ui.py

def render_message_with_reactions(message: dict, message_id: str):
    """Render message with reactions and copy buttons"""
    col1, col2 = st.columns([1, 0.15])
    
    with col1:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="message-bubble message-user">
                    <small><b>{message['user']}</b> • {message['timestamp']}</small>
                    <p>{message['content']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-bubble message-agent">
                    <small><b>{message.get('agent', 'Agent')}</b> • {message['timestamp']}</small>
                    <p>{message['content']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        reaction_col1, reaction_col2 = st.columns(2)
        with reaction_col1:
            if st.button("👍", key=f"like_{message_id}"):
                st.toast("Marked as helpful!")
        with reaction_col2:
            if st.button("📋", key=f"copy_{message_id}", help="Copy"):
                st.write(message['content'])

def render_source_card(source: dict):
    """Render enhanced source card"""
    with st.container():
        st.markdown(f"""
            <div class="card">
                <h4>📄 {source.get('title', 'Untitled')}</h4>
                <p style="font-size: 0.9em; color: var(--text-secondary);">
                    {source.get('snippet', 'No preview available')}
                </p>
                <div style="margin-top: 10px;">
                    <span class="badge badge-success">{source.get('type', 'Document')}</span>
                    <a href="{source.get('url', '#')}" target="_blank" style="margin-left: 10px;">
                        🔗 View Source
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_research_progress(nodes_completed: int, total_nodes: int):
    """Render progress bar with animation"""
    progress = nodes_completed / total_nodes
    st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="background: var(--surface-light); border-radius: 10px; overflow: hidden; height: 8px;">
                <div style="
                    width: {progress * 100}%;
                    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
                    height: 100%;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <small style="color: var(--text-secondary);">
                {nodes_completed}/{total_nodes} nodes completed
            </small>
        </div>
    """, unsafe_allow_html=True)

def render_stats_card(title: str, value: str, icon: str, color: str = "primary"):
    """Render stats card"""
    st.markdown(f"""
        <div class="card" style="border-left: 4px solid var(--{color});">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9em;">{title}</p>
                    <h3 style="margin: 5px 0 0 0;">{value}</h3>
                </div>
                <div style="font-size: 2em;">{icon}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
```

### 3. **Animated Chat Interface**

```python
# components/animated_chat.py

def render_animated_chat():
    """Render enhanced chat interface with animations"""
    st.markdown("""
        <script>
        function autoScroll() {
            var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        window.addEventListener('load', autoScroll);
        </script>
    """, unsafe_allow_html=True)

def render_typing_indicator():
    """Show AI is typing"""
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.8em; color: var(--text-secondary);">AI is thinking</span>
            <div class="loading" style="display: flex; gap: 4px;">
                <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%;"></div>
                <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%;"></div>
                <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_toast_notification(message: str, notification_type: str = "info"):
    """Show elegant notification"""
    icon_map = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    st.toast(f"{icon_map.get(notification_type)} {message}")
```

### 4. **Dashboard Layout**

```python
# pages/dashboard.py

def render_dashboard():
    """Render user dashboard with stats and analytics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_stats_card("Total Chats", "12", "💬", "primary")
    
    with col2:
        render_stats_card("Messages", "486", "💭", "secondary")
    
    with col3:
        render_stats_card("Research Tasks", "23", "🔍", "accent")
    
    with col4:
        render_stats_card("Sources Used", "156", "📚", "success")
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📊 Research Analytics")
    
    # Create charts
    tab1, tab2, tab3 = st.tabs(["Activity", "Topics", "Performance"])
    
    with tab1:
        st.line_chart({...})
    with tab2:
        st.bar_chart({...})
    with tab3:
        st.area_chart({...})
```

---

## Implementation Priority

### **Phase 1: Quick Wins (Week 1-2)** 🚀
- [ ] Apply custom CSS theme
- [ ] Add animated chat bubbles
- [ ] Implement message reactions
- [ ] Add typing indicators

### **Phase 2: MCP Foundation (Week 3-4)** 🔌
- [ ] Setup MCP server framework
- [ ] Implement web search tool
- [ ] Add database query tool
- [ ] Integrate into research graph

### **Phase 3: Advanced UI (Week 5-6)** ✨
- [ ] Dashboard with analytics
- [ ] Chat search and filtering
- [ ] Export functionality
- [ ] Dark/light theme switcher

### **Phase 4: Polish (Week 7-8)** 🎯
- [ ] Performance optimization
- [ ] Mobile responsiveness
- [ ] Accessibility improvements
- [ ] User feedback system

---

## Required Dependencies

```txt
# UI Enhancements
streamlit>=1.30.0
streamlit-option-menu>=0.3.2
streamlit-analytics>=0.4.1

# MCP Integration
mcp>=0.1.0
pydantic>=2.0

# Additional Tools
requests>=2.31.0
beautifulsoup4>=4.12.0
PyPDF2>=4.0.0
yfinance>=0.2.30
newsapi-python>=1.1
python-dotenv>=1.0.0
chromadb>=0.4.0
langchain>=0.1.0
```

---

## Installation & Next Steps

1. **Apply Styling**: Copy CSS to your main.py
2. **Create Components**: Add enhanced_ui.py to project
3. **Setup MCP**: Create mcp_server/ directory
4. **Test**: Run with `streamlit run main.py`
5. **Deploy**: Use Streamlit Cloud or Docker

