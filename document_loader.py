#!/usr/bin/env python3
"""
Document Loader & Uploader Tool

Allows you to:
1. Load documents from files (TXT, PDF, etc.)
2. Add them to RAG memory
3. Search across all documents
4. Manage document library
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
import json

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from rag_module import RAGManager
import uuid

# Try importing PDF support (optional)
try:
    import PyPDF2
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    print("⚠️  PyPDF2 not installed. PDF support disabled.")
    print("   To enable: pip install PyPDF2")


class DocumentLoader:
    """Load and manage documents for RAG system"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.rag = RAGManager(user_id=user_id)
        self.loaded_docs = {}
    
    def load_txt_file(self, file_path: str) -> str:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"✗ Error loading TXT: {e}")
            return None
    
    def load_pdf_file(self, file_path: str) -> str:
        """Load PDF file"""
        if not HAS_PDF_SUPPORT:
            print("✗ PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
        
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"✗ Error loading PDF: {e}")
            return None
    
    def load_document(self, file_path: str, category: str = "General") -> bool:
        """Load document from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False
        
        # Determine file type
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            content = self.load_txt_file(file_path)
        elif suffix == '.pdf':
            content = self.load_pdf_file(file_path)
        else:
            print(f"✗ Unsupported file type: {suffix}")
            print("   Supported: .txt, .pdf")
            return False
        
        if not content:
            return False
        
        # Add to memory
        doc_id = str(uuid.uuid4())
        self.rag.add_to_memory(
            text=content,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
                "category": category,
                "doc_id": doc_id,
                "file_size": len(content)
            }
        )
        
        self.loaded_docs[doc_id] = {
            "name": file_path.name,
            "path": str(file_path),
            "size": len(content),
            "category": category,
            "loaded_at": str(Path(file_path).stat().st_mtime)
        }
        
        print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
        return True
    
    def load_from_directory(self, dir_path: str, pattern: str = "*.txt") -> int:
        """Load all matching files from directory"""
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            print(f"✗ Directory not found: {dir_path}")
            return 0
        
        files = list(dir_path.glob(pattern))
        if not files:
            print(f"✗ No files matching '{pattern}' in {dir_path}")
            return 0
        
        print(f"\nLoading {len(files)} files from {dir_path}...")
        
        loaded = 0
        for file in files:
            if self.load_document(file):
                loaded += 1
        
        return loaded
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents"""
        docs = self.rag.search_memory(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            })
        
        return results
    
    def get_loaded_docs_info(self) -> Dict:
        """Get info about loaded documents"""
        return {
            "total_docs": len(self.loaded_docs),
            "documents": self.loaded_docs,
            "rag_stats": self.rag.get_memory_footprint()
        }
    
    def list_documents(self):
        """List all loaded documents"""
        if not self.loaded_docs:
            print("✗ No documents loaded")
            return
        
        print(f"\nLoaded Documents ({len(self.loaded_docs)}):")
        for doc_id, info in self.loaded_docs.items():
            print(f"  • {info['name']}")
            print(f"    Size: {info['size']} chars")
            print(f"    Category: {info['category']}")


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def interactive_demo():
    """Interactive document loader demo"""
    print("\n" + "="*70)
    print("  DOCUMENT LOADER & RAG SYSTEM")
    print("="*70)
    
    user_id = "doc_loader_user"
    loader = DocumentLoader(user_id)
    
    print(f"\nUser: {user_id}")
    print(f"Session: {loader.rag.session_id[:12]}...\n")
    
    menu_options = {
        "1": ("Load TXT file", load_single_file),
        "2": ("Load all TXT files from folder", load_from_folder),
        "3": ("List loaded documents", list_docs),
        "4": ("Search documents", search_docs),
        "5": ("Show stats", show_stats),
        "6": ("Load example documents", load_examples),
        "0": ("Exit", None)
    }
    
    while True:
        print("\nOptions:")
        for key, (desc, _) in menu_options.items():
            print(f"  {key}. {desc}")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == "0":
            print("✓ Goodbye!")
            break
        elif choice in menu_options:
            func = menu_options[choice][1]
            if func:
                func(loader)
        else:
            print("✗ Invalid option")


def load_single_file(loader):
    """Load single file"""
    file_path = input("Enter file path: ").strip()
    category = input("Enter category (default: General): ").strip() or "General"
    loader.load_document(file_path, category=category)


def load_from_folder(loader):
    """Load from folder"""
    folder_path = input("Enter folder path: ").strip()
    pattern = input("File pattern (default: *.txt): ").strip() or "*.txt"
    count = loader.load_from_directory(folder_path, pattern=pattern)
    print(f"✓ Loaded {count} files")


def list_docs(loader):
    """List documents"""
    loader.list_documents()
    info = loader.get_loaded_docs_info()
    print(f"\nMemory: {info['rag_stats']['rss_mb']:.1f}MB")


def search_docs(loader):
    """Search documents"""
    query = input("Enter search query: ").strip()
    if not query:
        return
    
    k = input("Number of results (default: 5): ").strip()
    k = int(k) if k.isdigit() else 5
    
    results = loader.search(query, k=k)
    
    print(f"\n🔍 Results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['source']}")
        print(f"    {result['content']}...")


def show_stats(loader):
    """Show statistics"""
    session = loader.rag.get_session_info()
    kg = loader.rag.get_kg_stats()
    memory = loader.rag.get_memory_footprint()
    
    print("\n" + "="*50)
    print("SESSION STATS")
    print("="*50)
    print(f"Turns: {session['conversation_turns']}")
    print(f"Queries: {session['queries_count']}")
    
    print("\nKNOWLEDGE GRAPH")
    print("="*50)
    print(f"Nodes: {kg['total_nodes']}")
    print(f"Edges: {kg['total_edges']}")
    print(f"Density: {kg['density']:.3f}")
    
    print("\nMEMORY")
    print("="*50)
    print(f"RSS: {memory['rss_mb']:.1f}MB")
    print(f"Usage: {memory['percent']:.1f}%")
    print(f"Documents: {memory['docs_count']}")


def load_examples(loader):
    """Load example documents"""
    examples = [
        ("Python.txt", "Python is a high-level programming language..."),
        ("JavaScript.txt", "JavaScript is a versatile scripting language for web development..."),
        ("Database.txt", "Databases store and manage data efficiently..."),
    ]
    
    # Create temp files
    temp_dir = Path("temp_docs")
    temp_dir.mkdir(exist_ok=True)
    
    for filename, content in examples:
        filepath = temp_dir / filename
        filepath.write_text(content)
        loader.load_document(str(filepath))
    
    print(f"✓ Loaded {len(examples)} example documents from temp_docs/")


# ============================================================================
# QUICK EXAMPLE
# ============================================================================

def quick_example():
    """Quick non-interactive example"""
    print("\n" + "="*70)
    print("  QUICK DOCUMENT LOADER EXAMPLE")
    print("="*70)
    
    loader = DocumentLoader("example_user")
    
    # Create sample files
    print("\nCreating sample documents...")
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    documents = {
        "ai_basics.txt": "Artificial Intelligence is transforming industries...",
        "python_guide.txt": "Python is a versatile programming language...",
        "web_dev.txt": "Web development involves creating interactive websites...",
    }
    
    for filename, content in documents.items():
        filepath = sample_dir / filename
        filepath.write_text(content)
    
    # Load documents
    print(f"\nLoading documents from {sample_dir}...")
    count = loader.load_from_directory(str(sample_dir), pattern="*.txt")
    
    # Show loaded docs
    print("\nLoaded Documents:")
    loader.list_documents()
    
    # Search
    print("\n" + "="*70)
    print("Search Examples:")
    print("="*70)
    
    for query in ["Python", "AI", "Web development"]:
        print(f"\n🔍 Query: '{query}'")
        results = loader.search(query, k=2)
        for i, result in enumerate(results, 1):
            print(f"   [{i}] {result['source']}: {result['content'][:60]}...")
    
    # Stats
    print("\n" + "="*70)
    print("Stats:")
    print("="*70)
    info = loader.get_loaded_docs_info()
    print(f"Total documents: {info['total_docs']}")
    print(f"Memory usage: {info['rag_stats']['rss_mb']:.1f}MB")
    print(f"Documents: {info['rag_stats']['docs_count']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_example()
    else:
        interactive_demo()
