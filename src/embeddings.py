"""
Embedding Engine module for semantic search and similarity calculations.

This module provides:
1. Text embedding generation using pre-trained models
2. Semantic similarity calculations
3. Relevant document retrieval
4. Context ranking and selection

The module uses sentence-transformers for generating embeddings and
cosine similarity for calculating semantic similarity between texts.
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingEngine:
    """
    Embedding Engine class for semantic search and similarity calculations.
    
    This class provides:
    1. Text embedding generation
    2. Semantic similarity scoring
    3. Relevant document retrieval
    4. Context ranking
    
    The class uses sentence-transformers for generating high-quality
    text embeddings that capture semantic meaning.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Embedding Engine with a pre-trained model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                      Default is 'all-MiniLM-L6-v2' which provides a good
                      balance between performance and speed
        """
        print(f"Using CPU...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy array of embeddings
        """
        print("Creating embeddings...")
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        print(f"Created embeddings shape: {embeddings.shape}")
        return embeddings.cpu().numpy()
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build a FAISS index from documents.
        given a set of vectors, we index them using Faiss — 
        then using another vector (the query vector), 
        we search for the most similar vectors within the index
        
        Args:
            documents: List of document dictionaries containing text and metadata
        """
        print("\nBuilding FAISS index...")
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        embeddings = self.create_embeddings(texts)
        
        dimension = embeddings.shape[1]
        print(f"Index dimension: {dimension}")
        
        # Create CPU index
        self.index = faiss.IndexFlatL2(dimension)
        print("Created FAISS index")
        
        self.index.add(embeddings.astype('float32'))
        print(f"Added {len(documents)} vectors to index")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents given a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        print("\nPerforming similarity search...")
        query_embedding = self.create_embeddings([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(1 / (1 + distance))
                results.append(doc)
        
        print(f"Found {len(results)} matches")
        return results
