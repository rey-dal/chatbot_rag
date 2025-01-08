"""
RAG (Retrieval-Augmented Generation) Engine for the chatbot.

This module implements a RAG system that combines:
1. Information retrieval from structured data (JSON files)
2. LLM-based text generation for natural language responses
3. Embedding-based semantic search for finding relevant context

The system uses a pre-trained language model to generate responses based on:
- Customer information from customer_list.json
- Module documentation from modules_documentation.json
- Domain-specific keywords from keywords.json
"""

from typing import List, Dict, Any
import os
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from data_loader import DataLoader
from embeddings import EmbeddingEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    """
    Main RAG Engine class that handles query processing and response generation.
    
    This class combines several components:
    1. Language Model: For generating natural language responses
    2. Data Loader: For accessing and preprocessing structured data
    3. Embedding Engine: For semantic search and context retrieval
    
    The workflow is:
    1. Receive user query
    2. Find relevant context using embeddings
    3. Create a prompt with the context
    4. Generate response using LLM
    """
    
    def __init__(self):
        """
        Initialize the RAG Engine with necessary components.
        
        Sets up:
        1. Language model and tokenizer
        2. Text generation pipeline
        3. Data loader for structured data
        4. Embedding engine for semantic search
        """
        # Initialize model and tokenizer
        model_name = "facebook/opt-350m"  # Using a smaller model for testing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None  # Load model to CPU
        )
        
        # Create pipeline with specific generation parameters
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=2048,  # Maximum length of generated text
            do_sample=True,   # Enable sampling-based generation
            temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = very random)
            top_p=0.95,  # Nucleus sampling parameter (higher = more diverse responses)
            repetition_penalty=1.15  # Penalize token repetition (>1.0 reduces repetition)
            #this will give you creative responses

            ################
            # Creative approach (current):
            # do_sample=True,      # Enable sampling for creative responses
            # temperature=0.7,     # Medium randomness
            # top_p=0.95,         # Allow diverse token selection
            # max_length=2048,     # Longer responses allowed
            # repetition_penalty=1.15,  # Light repetition control

            # Deterministic approach:
            # do_sample=False,     # Disable sampling for consistent outputs
            # temperature=0.1,     # Very low randomness for more precise outputs
            # top_p=0.5,          # Strict token selection for consistency
            # num_beams=4,         # Use beam search for better quality
            # max_length=1024,     # Shorter, more focused responses
            # repetition_penalty=1.2,  # Stronger repetition control
            # early_stopping=True  # Stop when good response found
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.data_loader = DataLoader()
        self.embedding_engine = EmbeddingEngine()
        
    def _create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM using the query and retrieved context.
        
        Args:
            query: User's natural language query
            context: List of relevant context documents
            
        Returns:
            Formatted prompt string for the LLM
        """
        prompt_template = """<s>[INST] You are an AI assistant for Product Managers, helping them understand analytics data.
        
Context information:
{context}

User Question: {query}

Please provide a clear and concise answer based on the context provided. If the information in the context is insufficient
or if you're unsure about something, please state that clearly. Focus on:
1. Direct answers to the question
2. Relevant metrics and trends
3. Any important caveats or limitations [/INST]

Answer: """
        
        context_str = "\n".join([f"- {doc['text']}" for doc in context])
        return prompt_template.format(context=context_str, query=query)
    
    def _format_response(self, llm_response: str, confidence: float) -> Dict[str, Any]:
        """
        Format the LLM response with metadata.
        
        Args:
            llm_response: Raw response from the LLM
            confidence: Confidence score for the response (embeddings.py will calculate this
            using the L2 (Euclidean) distance from FAISS and converting it 
            to a similarity score through the formula 1/(1 + distance))

        Returns:
            Dictionary containing the formatted response and metadata
        """
        return {
            "answer": llm_response,
            "confidence": confidence,
            "data_sources": [
                {"type": "analytics", "description": "Usage data"},
                {"type": "configuration", "description": "Configuration data"}
            ]
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using RAG.
        
        Workflow:
        1. Preprocess the query
        2. Get time context if needed
        3. Retrieve relevant datasets
        4. Get context using embeddings
        5. Generate and format response
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Preprocess query
        processed_query = self.data_loader.preprocess_query(query)
        
        # Get time context
        time_context = self.data_loader.get_time_context(query)
        
        # Get relevant datasets
        relevant_datasets = self.data_loader.get_relevant_datasets(query)
        
        # Retrieve relevant context using embeddings
        # Note: In a real implementation, you would load actual documents here
        context = [{"text": f"Sample context for {dataset}"} for dataset in relevant_datasets]
        
        # Create prompt and get LLM response
        prompt = self._create_prompt(processed_query, context)
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
        response = chain.run(query=processed_query)
        
        # Calculate confidence based on context relevance
        confidence = 0.8  # Placeholder - would be calculated based on embedding similarities
        
        return self._format_response(response, confidence)


if __name__ == "__main__":
    print("Starting AI Assistant...")
    print("Loading models and data...")
    rag_engine = RAGEngine()
    print("\nAI Assistant is ready! Ask me anything about our products and analytics.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nYour question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Thank you for using AI Assistant. Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("\nProcessing your question...")
            response = rag_engine.process_query(query)
            print("\nAnswer:", response['answer'])
            print(f"\nConfidence: {response['confidence']:.2%}")
            
        except KeyboardInterrupt:
            print("\nThank you for using AI Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking your question in a different way.")
