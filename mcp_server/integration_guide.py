# Integration guide: How to use MCP server with your research assistant

from mcp_server import call_mcp_tool, mcp_server
import asyncio
from typing import Optional

# ============================================================================
# EXAMPLE 1: Enhanced Researcher Node with MCP
# ============================================================================

async def researcher_node_with_mcp(state: dict) -> dict:
    """
    Example researcher node that uses MCP tools
    Add this to your research_graph.py
    """
    query = state.get("research_query", "")
    
    print(f"[Researcher] Starting research on: {query}")
    
    # Step 1: Web search using MCP
    print("[Researcher] Searching the web...")
    search_results = await call_mcp_tool(
        "web_search",
        query=query,
        num_results=5
    )
    
    if not search_results.get("success"):
        print(f"[Researcher] Search failed: {search_results.get('error')}")
        return state
    
    # Step 2: Scrape top results
    top_sources = []
    for result in search_results.get("results", [])[:2]:
        print(f"[Researcher] Scraping: {result['url']}")
        scraped = await call_mcp_tool(
            "web_scrape",
            url=result['url'],
            extract_type="text"
        )
        
        if scraped.get("success"):
            top_sources.append({
                "title": result['title'],
                "url": result['url'],
                "content": " ".join(scraped.get("content", [])[:5])
            })
    
    # Step 3: Get real-time data if relevant
    if any(keyword in query.lower() for keyword in ["stock", "crypto", "market"]):
        print("[Researcher] Fetching market data...")
        market_data = await call_mcp_tool(
            "realtime_data",
            data_type="stocks",
            query=query.split()[-1]
        )
        state["market_data"] = market_data
    
    # Update state with research findings
    state["research_findings"] = {
        "query": query,
        "sources_count": len(search_results.get("results", [])),
        "top_sources": top_sources,
        "raw_results": search_results.get("results", [])
    }
    
    return state


# ============================================================================
# EXAMPLE 2: Document Analysis with MCP
# ============================================================================

async def document_analyzer_with_mcp(document_path: str) -> dict:
    """
    Analyze a document using MCP tools
    """
    print(f"[Analyzer] Processing document: {document_path}")
    
    # Process document
    result = await call_mcp_tool(
        "document_process",
        file_path=document_path,
        operation="extract"
    )
    
    if not result.get("success"):
        return {"error": result.get("error")}
    
    return {
        "file_type": result.get("file_type"),
        "content": result.get("content"),
        "metadata": {
            "pages": result.get("page_count"),
            "chars": result.get("char_count"),
            "words": result.get("word_count")
        }
    }


# ============================================================================
# EXAMPLE 3: Database Query with MCP
# ============================================================================

async def query_research_db(sql_query: str) -> dict:
    """
    Query the research database using MCP
    """
    print(f"[DB] Executing query: {sql_query[:50]}...")
    
    result = await call_mcp_tool(
        "database_query",
        query=sql_query,
        database="research.db"
    )
    
    return result


# ============================================================================
# EXAMPLE 4: Integration into main.py
# ============================================================================

def update_main_with_mcp():
    """
    Steps to integrate MCP into main.py:
    
    1. Add import at top:
        from mcp_server import call_mcp_tool
        import asyncio
    
    2. Modify run_research_without_interrupts function:
        
        def run_research_without_interrupts(config, initial_state):
            # ... existing code ...
            for event in graph.stream(initial_state, config=config):
                # ... existing code ...
                
                # Add MCP tool calls here if needed
                if event_type == "researcher":
                    # Enhanced with MCP
                    pass
    
    3. Create new function for MCP-enhanced research:
        
        async def enhanced_research_with_mcp(topic: str):
            # Call MCP tools
            search_results = await call_mcp_tool("web_search", query=topic, num_results=5)
            # Process results
            return search_results
    
    4. Call from UI:
        if st.button("Research with MCP"):
            results = asyncio.run(enhanced_research_with_mcp(topic))
            st.json(results)
    """
    pass


# ============================================================================
# EXAMPLE 5: Custom Tool Implementation
# ============================================================================

async def create_custom_mcp_tool():
    """
    Template for creating custom MCP tools
    """
    from pydantic import BaseModel, Field
    
    class CustomRequest(BaseModel):
        param1: str = Field(description="First parameter")
        param2: int = Field(default=10, description="Second parameter")
    
    async def custom_tool_function(request: CustomRequest) -> dict:
        """Your custom tool logic"""
        try:
            # Implement your tool logic here
            result = f"Processed {request.param1} with param {request.param2}"
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Add to mcp_server.tools
    mcp_server.tools["custom_tool"] = custom_tool_function
    
    # Use it
    result = await mcp_server.execute_tool("custom_tool", {
        "param1": "test",
        "param2": 20
    })
    return result


# ============================================================================
# QUICK START
# ============================================================================

def quick_start_mcp():
    """
    Quick start guide for using MCP
    
    Step 1: Install dependencies
    ```
    pip install requests beautifulsoup4 PyPDF2 yfinance
    ```
    
    Step 2: Set environment variables
    ```
    TAVILY_API_KEY=your_key
    NEWS_API_KEY=your_key
    ```
    
    Step 3: Use in your code
    ```
    from mcp_server import call_mcp_tool
    import asyncio
    
    async def main():
        result = await call_mcp_tool(
            "web_search",
            query="AI research",
            num_results=5
        )
        print(result)
    
    asyncio.run(main())
    ```
    
    Step 4: Available Tools
    - web_search: Search the web
    - web_scrape: Scrape webpage content
    - database_query: Query SQL database
    - document_process: Process documents (PDF, TXT, JSON)
    - realtime_data: Get stocks, news, etc.
    """
    pass


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    async def test_all():
        print("Testing MCP Server Tools...\n")
        
        # List tools
        print("Available Tools:")
        for tool_name, tool_info in mcp_server.get_available_tools().items():
            print(f"  ✓ {tool_name}")
        
        print("\n" + "="*50 + "\n")
        
        # Test web search
        print("1. Testing Web Search...")
        result = await call_mcp_tool(
            "web_search",
            query="machine learning",
            num_results=3
        )
        if result.get("success"):
            print(f"   Found {len(result.get('results', []))} results ✓")
        else:
            print(f"   Error: {result.get('error')}")
        
        print("\n" + "="*50 + "\n")
        
        # Test document processing
        print("2. Testing Document Processing...")
        # Create a test file
        test_file = "test_doc.txt"
        with open(test_file, "w") as f:
            f.write("This is a test document for MCP.")
        
        result = await call_mcp_tool(
            "document_process",
            file_path=test_file,
            operation="extract"
        )
        if result.get("success"):
            print(f"   Processed {result.get('file_type')} file ✓")
        else:
            print(f"   Error: {result.get('error')}")
        
        print("\nMCP Server Tests Complete!")
    
    asyncio.run(test_all())
