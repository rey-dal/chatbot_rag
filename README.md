# Chatbot

A chatbot system that helps Product Managers understand data through natural language queries. The system uses RAG (Retrieval-Augmented Generation) to combine structured data with language model capabilities.

## Architecture

The system combines three key components to provide accurate and contextual responses:

#### a. Vector Search (Embeddings + FAISS)
- **Text Embeddings**: Uses sentence-transformers to convert text into high-dimensional vectors
  - Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
  - Captures semantic meaning of text in vector space
  - Similar texts have similar vector representations

- **FAISS Index**:
  - Facebook AI Similarity Search (FAISS) for efficient vector search
  - Creates an optimized index structure for fast similarity lookups
  - Uses L2 distance or cosine similarity for vector comparison
  - Can handle millions of vectors with sub-second query time

- **Search Process**:
  1. Convert user query to vector
  2. Search FAISS index for similar vectors
  3. Retrieve top-k most similar documents
  4. Calculate confidence scores based on similarity

#### b. Language Model Integration
- Uses a pre-trained language model for natural language understanding
- Current implementation: OPT-350M (can be upgraded to larger models)

### 2. Data Structure

#### a. Customer Data (`data/customers/customer_list.json`)
```json
{
    "metadata": {
        "last_updated": "2024-01-08"
    },
    "customers": [
        {
            "id": "C001",
            "name": "Company Name",
            "type": "Business Type",
            "modules_used": ["Module1", "Module2"],
            "usage_statistics": {
                "monthly_searches": 1000,
                "active_users": 50
            }
        }
    ]
}
```

#### b. Module Documentation (`data/modules/modules_documentation.json`)
```json
{
    "modules": [
        {
            "name": "Module Name",
            "version": "1.0",
            "features": ["Feature1", "Feature2"],
            "metrics": {
                "performance": ["metric1", "metric2"],
                "business": ["metric3", "metric4"]
            }
        }
    ]
}
```

#### c. Keywords Configuration (`config/keywords.json`)
```json
{
    "core_concepts": {
        "concept1": "definition1",
        "concept2": "definition2"
    },
    "metrics": {
        "performance": ["metric1", "metric2"],
        "business": ["metric3", "metric4"]
    }
}
```

### 3. Implementation Details

#### a. Core Components
1. **RAG Engine** (`src/rag_engine.py`)
   - Manages the overall query processing workflow
   - Combines retrieved context with language model responses
   - Handles response generation and formatting

2. **Embedding Engine** (`src/embeddings.py`)
   - Manages vector embeddings and similarity search
   - Implements FAISS indexing and search
   - Calculates relevance scores
   - Optimizes search performance

3. **Data Loader** (`src/data_loader.py`)
   - Handles structured data access
   - Preprocesses queries and documents
   - Manages file paths and configurations
   - Extracts time context from queries



## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the chatbot:
```bash
python src/rag_engine.py
```

## Example Usage

### 1. Customer Queries

The chatbot can answer questions like:

- "What modules does AMEX use?"
- "Which customers are using Flight Search?"
- "When did Expedia become our customer?"
...

Examples:

Q: "What modules does AMEX use?"
A: "AMEX currently uses three core modules:
    - All Fares
    - Hotel Booking
    - Flight Search
   They've been an active customer since 2020."

Q: "Tell me about the Hotel Booking module"
A: "The Hotel Booking module offers:
    - Global hotel inventory access
    - Real-time room availability
    - Rate comparison
    - Booking modification capabilities"

# Model Parameters

- `do_sample` (True): Controls generation strategy
  - True = Use sampling for creative responses
  - False = Use deterministic decoding (greedy/beam search)

- `temperature` (0.7): Controls response randomness
  - 0.0 = deterministic, exact responses
  - 1.0 = random, creative responses
  
- `top_p` (0.95): Nucleus sampling parameter
  - Controls diversity of generated text
  - Higher values (>0.9) = more diverse responses
  - Lower values (<0.9) = more focused responses

- `num_beams`: Number of beams for beam search
  - Higher values = more thorough search
  - Typically 4-5 beams for good results
  - Only used when do_sample=False

- `early_stopping`: Stop generation early
  - True = Stop when all beams reach end token
  - Helps prevent unnecessary computation
  - Used with beam search for efficiency

- `max_length`: Maximum length of generated text
  - Longer texts allow more detailed responses
  - Should match model's context window

- `repetition_penalty` (1.15): Prevents repetitive text
  - Values > 1.0 reduce repetition
  - Higher values = stronger penalty
