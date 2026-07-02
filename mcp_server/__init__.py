# MCP Server for AI Research Assistant
# Extends capabilities with external tools and real-time data

import asyncio
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
import json
import os

# ============================================================================
# REQUEST MODELS
# ============================================================================

class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results (1-10)")

class WebScrapingRequest(BaseModel):
    """Web scraping request"""
    url: str = Field(description="URL to scrape")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for content")
    extract_type: str = Field(default="text", description="text|html|metadata")

class DatabaseQueryRequest(BaseModel):
    """Database query request"""
    query: str = Field(description="SQL query")
    database: str = Field(default="research.db", description="Database file")

class DocumentProcessRequest(BaseModel):
    """Document processing request"""
    file_path: str = Field(description="Path to document")
    operation: str = Field(description="extract|summarize|analyze")

class RealtimeDataRequest(BaseModel):
    """Real-time data request"""
    data_type: str = Field(description="weather|stocks|news|crypto")
    query: str = Field(description="Search term or location")

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

class WebSearchTool:
    """Web search tool using Tavily API"""
    
    @staticmethod
    async def search(request: WebSearchRequest) -> Dict[str, Any]:
        """Search the web and return results"""
        from tavily import TavilyClient
        
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return {"success": False, "error": "Tavily API key not configured"}
            
            client = TavilyClient(api_key=api_key)
            results = client.search(
                query=request.query,
                max_results=request.num_results
            )
            
            return {
                "success": True,
                "results": [
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("snippet", ""),
                        "content": r.get("content", "")
                    }
                    for r in results.get("results", [])
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class WebScrapingTool:
    """Web scraping tool"""
    
    @staticmethod
    async def scrape(request: WebScrapingRequest) -> Dict[str, Any]:
        """Scrape webpage content"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(request.url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if request.extract_type == "metadata":
                return {
                    "success": True,
                    "title": soup.title.string if soup.title else "N/A",
                    "meta_description": soup.find("meta", {"name": "description"}).get("content") if soup.find("meta", {"name": "description"}) else "",
                    "url": request.url
                }
            
            if request.css_selector:
                elements = soup.select(request.css_selector)
            else:
                elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            
            if request.extract_type == "html":
                content = [str(elem) for elem in elements]
            else:
                content = [elem.get_text(strip=True) for elem in elements]
            
            return {
                "success": True,
                "content": content[:100],  # Limit to 100 items
                "url": request.url
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class DatabaseTool:
    """Database query tool"""
    
    @staticmethod
    async def query(request: DatabaseQueryRequest) -> Dict[str, Any]:
        """Execute database query"""
        try:
            import sqlite3
            
            if not os.path.exists(request.database):
                return {"success": False, "error": f"Database not found: {request.database}"}
            
            conn = sqlite3.connect(request.database)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(request.query)
            results = cursor.fetchall()
            conn.close()
            
            return {
                "success": True,
                "results": [dict(row) for row in results],
                "row_count": len(results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class DocumentTool:
    """Document processing tool"""
    
    @staticmethod
    async def process(request: DocumentProcessRequest) -> Dict[str, Any]:
        """Process document"""
        try:
            if not os.path.exists(request.file_path):
                return {"success": False, "error": f"File not found: {request.file_path}"}
            
            if request.file_path.endswith('.pdf'):
                return await DocumentTool._process_pdf(request)
            elif request.file_path.endswith('.txt'):
                return await DocumentTool._process_text(request)
            elif request.file_path.endswith('.json'):
                return await DocumentTool._process_json(request)
            else:
                return {"success": False, "error": "Unsupported file type"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def _process_pdf(request: DocumentProcessRequest) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            import PyPDF2
            
            with open(request.file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            return {
                "success": True,
                "content": text,
                "page_count": len(reader.pages),
                "file_type": "PDF"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def _process_text(request: DocumentProcessRequest) -> Dict[str, Any]:
        """Process text file"""
        try:
            with open(request.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "char_count": len(content),
                "word_count": len(content.split()),
                "file_type": "TXT"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def _process_json(request: DocumentProcessRequest) -> Dict[str, Any]:
        """Process JSON file"""
        try:
            with open(request.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "success": True,
                "content": data,
                "file_type": "JSON"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class RealtimeDataTool:
    """Real-time data retrieval tool"""
    
    @staticmethod
    async def get_data(request: RealtimeDataRequest) -> Dict[str, Any]:
        """Get real-time data"""
        if request.data_type == "stocks":
            return await RealtimeDataTool._get_stocks(request.query)
        elif request.data_type == "news":
            return await RealtimeDataTool._get_news(request.query)
        else:
            return {"success": False, "error": f"Unsupported data type: {request.data_type}"}
    
    @staticmethod
    async def _get_stocks(symbol: str) -> Dict[str, Any]:
        """Get stock data"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def _get_news(query: str) -> Dict[str, Any]:
        """Get news data"""
        try:
            api_key = os.getenv("NEWS_API_KEY")
            if not api_key:
                return {"success": False, "error": "NewsAPI key not configured"}
            
            import requests
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=10"
            response = requests.get(url, headers={"X-Api-Key": api_key})
            data = response.json()
            
            return {
                "success": True,
                "articles": [
                    {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name"),
                        "publishedAt": article.get("publishedAt"),
                        "description": article.get("description")
                    }
                    for article in data.get("articles", [])
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# ============================================================================
# MCP SERVER CLASS
# ============================================================================

class ResearchAssistantMCPServer:
    """Main MCP server for research assistant"""
    
    def __init__(self):
        self.tools = {
            "web_search": WebSearchTool.search,
            "web_scrape": WebScrapingTool.scrape,
            "database_query": DatabaseTool.query,
            "document_process": DocumentTool.process,
            "realtime_data": RealtimeDataTool.get_data,
        }
    
    async def execute_tool(self, tool_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        
        try:
            tool_func = self.tools[tool_name]
            result = await tool_func(**request_data)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        return {
            "web_search": {
                "description": "Search the web for information",
                "input_schema": WebSearchRequest.model_json_schema()
            },
            "web_scrape": {
                "description": "Scrape content from a webpage",
                "input_schema": WebScrapingRequest.model_json_schema()
            },
            "database_query": {
                "description": "Execute SQL queries on research database",
                "input_schema": DatabaseQueryRequest.model_json_schema()
            },
            "document_process": {
                "description": "Process and extract content from documents",
                "input_schema": DocumentProcessRequest.model_json_schema()
            },
            "realtime_data": {
                "description": "Get real-time data (stocks, news, etc.)",
                "input_schema": RealtimeDataRequest.model_json_schema()
            },
        }

# ============================================================================
# INTEGRATION WITH RESEARCH GRAPH
# ============================================================================

# Singleton instance
mcp_server = ResearchAssistantMCPServer()

async def call_mcp_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Call an MCP tool from research graph"""
    return await mcp_server.execute_tool(tool_name, kwargs)

# Example usage in research_graph.py:
# from mcp_server import call_mcp_tool
# 
# async def researcher_node(state):
#     # Use MCP tools for enhanced research
#     search_results = await call_mcp_tool(
#         "web_search",
#         query=state.get("research_query"),
#         num_results=5
#     )

if __name__ == "__main__":
    # Test the MCP server
    import asyncio
    
    async def test():
        server = ResearchAssistantMCPServer()
        
        # List available tools
        tools = server.get_available_tools()
        print("Available tools:")
        for tool_name, tool_info in tools.items():
            print(f"  - {tool_name}: {tool_info['description']}")
        
        # Test web search
        result = await server.execute_tool("web_search", {
            "query": "Python programming",
            "num_results": 3
        })
        print("\nWeb search result:", json.dumps(result, indent=2)[:500])
    
    asyncio.run(test())
