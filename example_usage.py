#!/usr/bin/env python3
"""
Complete Example: Using Enhanced RAG with Real Documents and Test Cases

This script demonstrates:
1. Adding sample documents to memory
2. Running queries with context
3. Monitoring memory metrics
4. Extracting facts
5. Managing sessions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from rag_module import RAGManager
from datetime import datetime

# ============================================================================
# SAMPLE DOCUMENTS - Real-world examples
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "content": """
        Machine Learning Fundamentals
        
        Machine Learning is a subset of Artificial Intelligence that enables computers to learn 
        and improve from experience without being explicitly programmed. It focuses on the development 
        of algorithms and statistical models that can learn patterns from data.
        
        Types of Machine Learning:
        1. Supervised Learning: Training data includes labeled examples (input-output pairs)
        2. Unsupervised Learning: Finding hidden patterns in unlabeled data
        3. Reinforcement Learning: Learning through interaction and rewards
        
        Key Applications:
        - Image Recognition and Computer Vision
        - Natural Language Processing
        - Recommendation Systems
        - Predictive Analytics
        - Autonomous Vehicles
        """,
        "metadata": {
            "source": "ml_fundamentals.txt",
            "category": "Machine Learning",
            "date": "2024-01-15"
        }
    },
    {
        "content": """
        Deep Learning and Neural Networks
        
        Deep Learning is a specialized branch of Machine Learning that uses artificial neural 
        networks with multiple layers (deep architectures) to process data. These networks are 
        inspired by the biological neural networks in animal brains.
        
        Architecture Components:
        - Input Layer: Receives raw data
        - Hidden Layers: Process and transform data through weighted connections
        - Output Layer: Produces final prediction or classification
        
        Popular Deep Learning Models:
        1. Convolutional Neural Networks (CNN) - Image processing
        2. Recurrent Neural Networks (RNN) - Sequence data and time series
        3. Transformers - Natural language processing, state-of-the-art performance
        4. Generative Adversarial Networks (GANs) - Generate synthetic data
        
        Deep Learning has achieved breakthrough results in:
        - Computer Vision (image classification, object detection)
        - Natural Language Understanding (translation, sentiment analysis)
        - Speech Recognition
        - Game Playing (AlphaGo, Chess engines)
        """,
        "metadata": {
            "source": "deep_learning.txt",
            "category": "Deep Learning",
            "date": "2024-02-20"
        }
    },
    {
        "content": """
        Natural Language Processing (NLP)
        
        Natural Language Processing is a field of AI that focuses on enabling computers to understand, 
        interpret, and generate human language in a meaningful and useful way.
        
        Core NLP Tasks:
        1. Tokenization: Breaking text into individual words or tokens
        2. Named Entity Recognition: Identifying people, places, organizations
        3. Sentiment Analysis: Determining emotional tone (positive, negative, neutral)
        4. Machine Translation: Converting text from one language to another
        5. Question Answering: Extracting answers from documents
        6. Text Summarization: Creating concise summaries of long documents
        
        Modern NLP Approaches:
        - Word Embeddings (Word2Vec, GloVe): Representing words as vectors
        - Transformer Models (BERT, GPT): Context-aware language understanding
        - Transfer Learning: Using pre-trained models for specific tasks
        
        Real-world Applications:
        - Chatbots and Virtual Assistants (Alexa, Google Assistant)
        - Email Spam Detection
        - Social Media Monitoring
        - Healthcare: Clinical Note Analysis
        """,
        "metadata": {
            "source": "nlp_guide.txt",
            "category": "NLP",
            "date": "2024-03-10"
        }
    },
    {
        "content": """
        Computer Vision Basics
        
        Computer Vision is the field of AI that enables computers to interpret and understand 
        visual information from images and videos, mimicking human vision capabilities.
        
        Key Computer Vision Tasks:
        1. Image Classification: Assigning labels to images
        2. Object Detection: Locating and identifying objects within images
        3. Semantic Segmentation: Classifying each pixel in an image
        4. Instance Segmentation: Distinguishing individual objects of the same class
        5. Face Recognition: Identifying individuals from facial features
        6. Optical Character Recognition (OCR): Extracting text from images
        
        Common Datasets:
        - ImageNet: 14 million labeled images for classification
        - COCO: Object detection and segmentation dataset
        - Cityscapes: Autonomous driving scene understanding
        
        Techniques:
        - Feature Extraction: Identifying important visual patterns
        - Convolutional Neural Networks (CNNs): Processing visual data
        - Edge Detection: Finding object boundaries
        - Region-based Methods: Focusing on specific image regions
        
        Applications:
        - Medical Imaging: Disease detection and diagnosis
        - Autonomous Vehicles: Road scene understanding
        - Retail: Inventory management and customer analytics
        - Manufacturing: Quality control and defect detection
        """,
        "metadata": {
            "source": "computer_vision.txt",
            "category": "Computer Vision",
            "date": "2024-03-25"
        }
    },
    {
        "content": """
        Large Language Models (LLMs) and GPT Architecture
        
        Large Language Models are deep neural networks trained on massive amounts of text data 
        that can generate human-like text and perform various language tasks.
        
        Evolution of LLMs:
        - GPT-1 (2018): 117 million parameters
        - GPT-2 (2019): 1.5 billion parameters - showed impressive text generation
        - GPT-3 (2020): 175 billion parameters - in-context learning capabilities
        - GPT-4 (2023): Advanced reasoning and multimodal understanding
        
        Key Concepts:
        1. Transformer Architecture: Parallel processing with attention mechanisms
        2. Tokenization: Breaking text into subword units (BPE, SentencePiece)
        3. Attention Mechanism: Weighting different parts of input for importance
        4. Context Window: Maximum length of text the model can process
        5. Temperature: Controls randomness in output (0=deterministic, 1=very random)
        
        Training Process:
        - Pre-training: Learning from unlabeled text corpus
        - Fine-tuning: Adapting to specific tasks with labeled data
        - RLHF: Reinforcement Learning from Human Feedback for alignment
        
        Capabilities:
        - Zero-shot Learning: Performing new tasks without examples
        - Few-shot Learning: Learning from minimal examples
        - Chain-of-Thought Reasoning: Step-by-step problem solving
        - Code Generation: Writing and debugging code
        
        Limitations:
        - Hallucination: Generating false information
        - Context Length: Limited ability to process very long documents
        - Knowledge Cutoff: Training data has temporal limits
        - Computational Cost: Expensive to train and deploy
        """,
        "metadata": {
            "source": "llm_guide.txt",
            "category": "LLMs",
            "date": "2024-04-05"
        }
    }
]

# ============================================================================
# TEST QUERIES - Real-world use cases
# ============================================================================

TEST_QUERIES = [
    {
        "query": "What is machine learning and its main types?",
        "expected_topics": ["Machine Learning", "Supervised Learning", "Unsupervised Learning"],
        "description": "Basic ML fundamentals"
    },
    {
        "query": "Explain the difference between deep learning and traditional machine learning",
        "expected_topics": ["Deep Learning", "Neural Networks", "Layers"],
        "description": "Comparing ML approaches"
    },
    {
        "query": "What are the applications of NLP in real-world scenarios?",
        "expected_topics": ["Chatbots", "Spam Detection", "Machine Translation"],
        "description": "NLP applications"
    },
    {
        "query": "Tell me about computer vision tasks and their applications",
        "expected_topics": ["Image Classification", "Object Detection", "Medical Imaging"],
        "description": "Computer Vision overview"
    },
    {
        "query": "What is transformer architecture and why is it important?",
        "expected_topics": ["Transformer", "Attention Mechanism", "Parallel Processing"],
        "description": "Transformer explanation"
    },
    {
        "query": "How do large language models like GPT work?",
        "expected_topics": ["LLM", "Tokenization", "Attention", "Pre-training"],
        "description": "LLM mechanics"
    },
    {
        "query": "What are the limitations of current AI models?",
        "expected_topics": ["Hallucination", "Context Length", "Computational Cost"],
        "description": "AI limitations"
    },
    {
        "query": "Explain semantic segmentation and instance segmentation",
        "expected_topics": ["Segmentation", "Classification", "Computer Vision"],
        "description": "Advanced CV concepts"
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n➜ {title}")
    print("-" * 70)


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    print_section("ENHANCED RAG SYSTEM - COMPLETE EXAMPLE")
    
    # Step 1: Initialize RAGManager
    print_subsection("Step 1: Initialize RAG Manager")
    user_id = "demo_user_001"
    rag = RAGManager(user_id=user_id)
    print(f"✓ RAGManager initialized for user: {user_id}")
    print(f"✓ Session ID: {rag.session_id[:8]}...")
    
    # Step 2: Add sample documents
    print_subsection("Step 2: Adding Sample Documents to Memory")
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        rag.add_to_memory(
            text=doc["content"],
            metadata=doc["metadata"]
        )
        print(f"✓ Added document {i}/{len(SAMPLE_DOCUMENTS)}: {doc['metadata']['source']}")
    
    # Step 3: Get system stats before queries
    print_subsection("Step 3: System Status Before Queries")
    kg_stats = rag.get_kg_stats()
    footprint = rag.get_memory_footprint()
    print(f"Knowledge Graph:")
    print(f"  • Nodes: {kg_stats['total_nodes']}")
    print(f"  • Edges: {kg_stats['total_edges']}")
    print(f"Memory:")
    print(f"  • RSS: {footprint['rss_mb']:.1f}MB")
    print(f"  • Docs: {footprint['docs_count']}")
    
    # Step 4: Run test queries
    print_subsection("Step 4: Running Test Queries")
    print(f"Will run {len(TEST_QUERIES)} queries with different topics...\n")
    
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}] {test['description']}")
        print(f"Question: {test['query']}")
        
        # Search with context
        docs, session_context = rag.search_with_session_context(
            query=test['query'],
            k=3
        )
        
        print(f"Retrieved: {len(docs)} documents")
        for j, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  [{j}] {source}: {doc.page_content[:80]}...")
        
        # Generate answer
        answer = rag.graph_rag_answer(test['query'], k=3)
        print(f"Answer: {answer[:200]}...")
    
    # Step 5: Session Analysis
    print_subsection("Step 5: Session Analysis")
    session_info = rag.get_session_info()
    print(f"Session ID: {session_info['session_id'][:12]}...")
    print(f"Created: {session_info['created_at'].split('T')[0]}")
    print(f"Conversation Turns: {session_info['conversation_turns']}")
    print(f"Total Queries: {session_info['queries_count']}")
    
    # Step 6: Extract facts
    print_subsection("Step 6: Extracting Important Facts")
    facts = rag.extract_important_facts()
    if facts:
        print(f"Extracted {len(facts)} key facts:")
        for i, fact in enumerate(facts[:5], 1):
            print(f"  {i}. {fact[:100]}...")
    
    # Step 7: Memory metrics
    print_subsection("Step 7: Memory & Performance Metrics")
    memory_stats = rag.get_memory_stats()
    print(f"Retrieval Statistics:")
    print(f"  • Total Retrievals: {memory_stats['total_retrievals']}")
    print(f"  • Avg Relevance Score: {memory_stats['avg_relevance_score']:.3f}")
    print(f"  • Avg Response Time: {memory_stats['avg_response_time']:.3f}s")
    print(f"  • Peak Memory: {memory_stats['peak_memory_mb']:.1f}MB")
    print(f"  • Total Docs Retrieved: {memory_stats['total_documents_retrieved']}")
    
    # Step 8: Session summarization
    print_subsection("Step 8: Auto-Summarizing Session")
    rag.summarize_current_session()
    updated_session = rag.get_session_info()
    print(f"Summary: {updated_session['summary'][:200]}...")
    
    # Step 9: System overview
    print_subsection("Step 9: System Overview")
    system_summary = rag.get_system_memory_summary()
    kg_stats = rag.get_kg_stats()
    footprint = rag.get_memory_footprint()
    
    print(f"Knowledge Graph:")
    print(f"  • Total Nodes: {kg_stats['total_nodes']}")
    print(f"  • Total Edges: {kg_stats['total_edges']}")
    print(f"  • Density: {kg_stats['density']:.3f}")
    print(f"  • Top Nodes: {kg_stats['most_accessed'][:3]}")
    
    print(f"\nMemory Footprint:")
    print(f"  • RSS: {footprint['rss_mb']:.1f}MB")
    print(f"  • VMS: {footprint['vms_mb']:.1f}MB")
    print(f"  • Memory %: {footprint['percent']:.1f}%")
    print(f"  • Sessions: {footprint['sessions_count']}")
    print(f"  • KG Nodes: {footprint['kg_nodes']}")
    print(f"  • KG Edges: {footprint['kg_edges']}")
    
    print(f"\nSystem Operations:")
    print(f"  • Total Operations: {system_summary['total_operations']}")
    print(f"  • Avg Memory Usage: {system_summary['avg_memory_usage_mb']:.1f}MB")
    
    # Step 10: Session management demo
    print_subsection("Step 10: Session Management Demo")
    all_sessions = rag.get_all_sessions()
    print(f"Total sessions for user: {len(all_sessions)}")
    for session in all_sessions[:3]:
        print(f"  • {session['session_id'][:8]}... - {session['turns']} turns - {session['created_at'].split('T')[0]}")
    
    print_section("✓ EXAMPLE COMPLETED SUCCESSFULLY")
    print(f"""
You can now:
1. Query the system with your own questions
2. Monitor memory metrics in real-time
3. Switch between sessions
4. Extract facts from conversations
5. Analyze knowledge graph growth

For more details, see: RAG_MEMORY_GUIDE.md
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
