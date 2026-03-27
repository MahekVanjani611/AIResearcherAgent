import os
from typing import TypedDict, List, Annotated, cast
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
from dotenv import load_dotenv
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("research_logs.log", rotation="10 MB")

load_dotenv()

# Define the state object
class AgentState(TypedDict):
    user_query: str
    plan: List[str]
    research_results: Annotated[List[str], operator.add]
    insights: Annotated[List[str], operator.add]
    fact_checks: Annotated[List[str], operator.add]
    critic_score: int
    critic_feedback: str
    final_report: str
    sources: Annotated[List[str], operator.add]

# Initialize LLM and Tavily
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- Node Implementations ---

def planner_node(state: AgentState):
    logger.info(f"Starting Planner for query: {state['user_query'][:50]}...")
    query = state["user_query"]
    prompt = f"As a Research Lead, decompose the following topic into 3-4 specific research sub-tasks: {query}. Return ONLY a bulleted list."
    response = llm.invoke([SystemMessage(content="You are a planning agent."), HumanMessage(content=prompt)])
    tasks = [t.strip("- ") for t in response.content.strip().split("\n") if t.strip()]
    logger.debug(f"Planner generated {len(tasks)} tasks.")
    return {"plan": tasks}

def researcher_node(state: AgentState):
    logger.info("Starting Researcher...")
    tasks = state["plan"]
    results: List[str] = []
    sources: List[str] = []
    
    for task in tasks[:3]:
        logger.debug(f"Searching for: {task}")
        try:
            search = tavily.search(query=task, max_results=3)
            search_results = search.get("results", [])
            for r in search_results:
                url = r.get('url', '')
                content = r.get('content', '')
                results.append(f"Source: {url}\nContent: {content}")
                sources.append(url)
        except Exception as e:
            logger.error(f"Search error for '{task}': {e}")
            
    logger.info(f"Researcher found {len(results)} results from {len(set(sources))} sources.")
    return {"research_results": results, "sources": sources}

def analyst_node(state: AgentState):
    logger.info("Starting Analyst...")
    res_list = state["research_results"]
    data = "\n\n".join(res_list)
    prompt = f"Analyze the following research results and extract 3 key data-driven insights: \n\n{data}"
    response = llm.invoke([SystemMessage(content="You are an analyst agent."), HumanMessage(content=prompt)])
    logger.debug("Analyst finished extracting insights.")
    return {"insights": [str(response.content)]}

def fact_checker_node(state: AgentState):
    logger.info("Starting Fact-Checker...")
    res_list = state["research_results"]
    data = "\n\n".join(res_list)
    prompt = f"Cross-verify these claims against the following context. Highlight any discrepancies or confirm validity: \n\n{data}"
    response = llm.invoke([SystemMessage(content="You are a fact-checker."), HumanMessage(content=prompt)])
    logger.debug("Fact-checker finished verification.")
    return {"fact_checks": [str(response.content)]}

def critic_node(state: AgentState):
    logger.info("Starting Critic...")
    insights = "\n".join(state["insights"])
    facts = "\n".join(state["fact_checks"])
    prompt = f"Critique the following research synthesis. Score it from 1-10 and provide feedback.\n\nInsights: {insights}\n\nFacts: {facts}\n\nFormat: Score: [X], Feedback: [Feedback]"
    response = llm.invoke([SystemMessage(content="You are a quality controller."), HumanMessage(content=prompt)])
    content = str(response.content)
    score = 7 
    if "Score:" in content:
        try:
            # Extract only the numeric part before any slash or non-digit (except digits)
            parts = content.split("Score:")[1].strip().split()[0]
            # Take only the part before a '/' if it exists
            score_val = parts.split('/')[0]
            # Extract digits only
            score_str = "".join(filter(str.isdigit, score_val))
            if score_str:
                score = int(score_str)
        except Exception as e:
            logger.warning(f"Failed to parse score accurately from '{content[:50]}...': {e}")
    
    logger.info(f"Critic Score: {score}")
    return {"critic_score": score, "critic_feedback": content}

def writer_node(state: AgentState):
    logger.info("Starting Writer...")
    insights = "\n".join(state["insights"])
    sources = "\n".join(set(state["sources"]))
    prompt = f"Write a professional, structured markdown report on: {state['user_query']}.\n\nInsights:\n{insights}\n\nSources:\n{sources}"
    response = llm.invoke([SystemMessage(content="You are a professional report writer."), HumanMessage(content=prompt)])
    logger.info("Writer finished final report.")
    return {"final_report": str(response.content)}

# --- Graph Construction ---

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("researcher", "fact_checker")
    workflow.add_edge("analyst", "critic")
    workflow.add_edge("fact_checker", "critic")

    def routing_logic(state: AgentState) -> str:
        score = int(state.get("critic_score", 0))
        if score >= 7:
            logger.info("Satisfactory score achieved. Proceeding to writer.")
            return "writer"
        logger.warning(f"Score {score} too low. Re-planning...")
        return "planner"

    workflow.add_conditional_edges(
        "critic",
        routing_logic,
        {
            "writer": "writer",
            "planner": "planner"
        }
    )

    workflow.add_edge("writer", END)
    
    # Initialize memory saver for checkpointing / interrupts
    checkpointer = MemorySaver()
    
    # Compile with interrupt capability
    # We interrupt BEFORE every step except the start, to allow the user to check each stage
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["researcher", "analyst", "fact_checker", "critic", "writer"]
    )

graph = build_graph()

def get_graph_visualization():
    """Generates a Mermaid graph representation."""
    try:
        # returns the mermaid string
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.error(f"Failed to generate graph visualization: {e}")
        return None

