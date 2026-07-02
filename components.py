# Enhanced UI Components for Streamlit Research Assistant

import streamlit as st
from datetime import datetime
import json
from typing import Dict, Optional, List

# ============================================================================
# CUSTOM THEME & STYLING
# ============================================================================

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

    /* Sidebar Enhancement */
    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, var(--surface) 0%, var(--background) 100%);
    }

    /* Message Bubbles */
    .message-bubble {
        padding: 14px 16px;
        border-radius: var(--radius);
        margin-bottom: 10px;
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
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
        border-bottom-right-radius: 4px;
    }

    .message-agent {
        background: var(--surface-light);
        color: var(--text-primary);
        margin-right: auto;
        margin-left: 0;
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
        padding-left: 12px;
    }

    /* Message Timestamp */
    .message-time {
        font-size: 0.75em;
        color: var(--text-secondary);
        opacity: 0.7;
        margin-bottom: 4px;
    }

    /* Animations */
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

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    .bounce {
        animation: bounce 1s ease-in-out infinite;
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
        transform: translateY(-2px);
    }

    .card-header {
        font-weight: 600;
        margin-bottom: 8px;
        color: var(--text-primary);
    }

    .card-body {
        color: var(--text-secondary);
        font-size: 0.9em;
        line-height: 1.6;
    }

    /* Buttons - Modern */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: var(--radius);
        font-weight: 600;
        transition: all var(--transition);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius) !important;
        transition: all var(--transition);
        padding: 10px 12px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
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

    .badge-info {
        background: rgba(99, 102, 241, 0.2);
        color: var(--primary);
    }

    /* Sections */
    .section-header {
        font-size: 1.2em;
        font-weight: 600;
        color: var(--text-primary);
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--primary);
    }

    /* Stats Card */
    .stats-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px;
        text-align: center;
    }

    .stats-value {
        font-size: 2em;
        font-weight: 700;
        color: var(--primary);
        margin: 8px 0;
    }

    .stats-label {
        font-size: 0.9em;
        color: var(--text-secondary);
    }

    /* Scrollbar */
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

    /* Progress Bar */
    .progress-container {
        background: var(--surface-light);
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 10px 0;
    }

    .progress-bar {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        height: 100%;
        transition: width 0.3s ease;
        border-radius: 10px;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--text-secondary);
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--surface-light);
        color: var(--text-primary);
        text-align: center;
        border-radius: var(--radius);
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity var(--transition);
        font-size: 0.85em;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .message-user, .message-agent {
            max-width: 95%;
        }
    }
</style>
"""

# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================

def apply_custom_theme():
    """Apply the custom theme to the app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_stats_card(title: str, value: str, icon: str, subtitle: str = ""):
    """Render a statistics card"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
            <div class="card">
                <div class="stats-label">{title}</div>
                <div class="stats-value">{value}</div>
                {f'<div class="stats-label" style="font-size: 0.8em; margin-top: 4px;">{subtitle}</div>' if subtitle else ''}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div style='font-size: 2.5em; text-align: center; margin-top: 8px;'>{icon}</div>", unsafe_allow_html=True)

def render_message_bubble(message: Dict, show_reactions: bool = True):
    """Render an enhanced message bubble"""
    role = message.get('role', 'agent')
    content = message.get('content', '')
    timestamp = message.get('timestamp', '')
    sender = message.get('user') if role == 'user' else message.get('agent', 'AI')
    
    bubble_class = f"message-{role}"
    
    col1, col2 = st.columns([20, 1]) if show_reactions else (st.columns([1])[0], None)
    
    with col1:
        st.markdown(f"""
            <div class="message-bubble {bubble_class}">
                <div class="message-time"><b>{sender}</b> • {timestamp}</div>
                <p style="margin: 8px 0 0 0;">{content}</p>
            </div>
        """, unsafe_allow_html=True)
    
    if show_reactions and col2:
        with col2:
            reaction_col1, reaction_col2 = st.columns(2)
            with reaction_col1:
                if st.button("👍", key=f"like_{id(message)}", help="Helpful"):
                    st.toast("✨ Thanks for the feedback!")
            with reaction_col2:
                if st.button("📋", key=f"copy_{id(message)}", help="Copy"):
                    st.info(content)

def render_source_card(source: Dict):
    """Render a source/document card"""
    st.markdown(f"""
        <div class="card">
            <div class="card-header">📄 {source.get('title', 'Untitled')}</div>
            <div class="card-body">
                {source.get('snippet', 'No preview available')}
            </div>
            <div style="margin-top: 12px; display: flex; gap: 8px;">
                <span class="badge badge-info">{source.get('type', 'Document')}</span>
                <a href="{source.get('url', '#')}" target="_blank" style="
                    color: var(--primary);
                    text-decoration: none;
                    font-size: 0.9em;
                    font-weight: 500;
                ">🔗 View Source</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_progress_bar(current: int, total: int, label: str = "Progress"):
    """Render an animated progress bar"""
    percentage = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
        <div style="margin: 12px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.9em; color: var(--text-secondary);">{label}</span>
                <span style="font-size: 0.9em; color: var(--primary); font-weight: 600;">{current}/{total}</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {percentage}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_typing_indicator():
    """Show AI is typing"""
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin: 10px 0;">
            <span style="font-size: 0.9em; color: var(--text-secondary);">AI is thinking</span>
            <div style="display: flex; gap: 3px;">
                <div style="width: 6px; height: 6px; background: var(--primary); border-radius: 50%;" class="bounce"></div>
                <div style="width: 6px; height: 6px; background: var(--primary); border-radius: 50%; animation-delay: 0.1s;" class="bounce"></div>
                <div style="width: 6px; height: 6px; background: var(--primary); border-radius: 50%; animation-delay: 0.2s;" class="bounce"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_badge(text: str, badge_type: str = "info"):
    """Render a badge"""
    st.markdown(f'<span class="badge badge-{badge_type}">{text}</span>', unsafe_allow_html=True)

def render_section_header(title: str, icon: str = ""):
    """Render a section header"""
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def render_notification(message: str, notification_type: str = "info", dismissible: bool = True):
    """Render a notification"""
    icon_map = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    
    bg_colors = {
        "success": "rgba(16, 185, 129, 0.1)",
        "error": "rgba(239, 68, 68, 0.1)",
        "warning": "rgba(245, 158, 11, 0.1)",
        "info": "rgba(99, 102, 241, 0.1)"
    }
    
    border_colors = {
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b",
        "info": "#6366f1"
    }
    
    icon = icon_map.get(notification_type, "ℹ️")
    bg = bg_colors.get(notification_type, "rgba(99, 102, 241, 0.1)")
    border = border_colors.get(notification_type, "#6366f1")
    
    st.markdown(f"""
        <div style="
            background: {bg};
            border-left: 4px solid {border};
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 1.2em;">{icon}</span>
            <span style="color: var(--text-primary);">{message}</span>
        </div>
    """, unsafe_allow_html=True)

def render_divider():
    """Render a styled divider"""
    st.markdown("""
        <div style="
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            margin: 20px 0;
        "></div>
    """, unsafe_allow_html=True)

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def render_dashboard_header(username: str, chat_count: int = 0, message_count: int = 0):
    """Render dashboard header with user info"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
            <div style="padding: 20px;">
                <h1 style="margin: 0;">👤 Welcome, {username}!</h1>
                <p style="color: var(--text-secondary); margin: 8px 0 0 0;">
                    Your AI Research Assistant
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        render_stats_card("Chats", str(chat_count), "💬")
    
    with col3:
        render_stats_card("Messages", str(message_count), "💭")

def render_code_block(code: str, language: str = "python"):
    """Render a syntax-highlighted code block"""
    st.markdown(f"""
        ```{language}
        {code}
        ```
    """)

# Export all functions
__all__ = [
    'apply_custom_theme',
    'render_stats_card',
    'render_message_bubble',
    'render_source_card',
    'render_progress_bar',
    'render_typing_indicator',
    'render_badge',
    'render_section_header',
    'render_notification',
    'render_divider',
    'render_dashboard_header',
    'render_code_block',
    'CUSTOM_CSS'
]
