"""
Data Loader module for managing and preprocessing structured data.

This module handles:
1. Loading and parsing JSON configuration files
2. Query preprocessing and enhancement
3. Time-based context extraction
4. Dataset relevance scoring

The module works with three main data sources:
- keywords.json: Domain-specific vocabulary and concepts
- customer_list.json: Customer profiles and usage data
- modules_documentation.json: Technical documentation for modules
"""

import json
import os
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

class DataLoader:
    """
    Data Loader class for managing structured data access and preprocessing.
    
    This class handles:
    1. Loading configuration files
    2. Query preprocessing
    3. Time context extraction
    4. Dataset relevance scoring
    
    The class uses absolute paths based on the script location to ensure
    reliable file access regardless of where the script is run from.
    """
    
    def __init__(self):
        """
        Initialize the Data Loader with correct file paths.
        
        Sets up absolute paths for:
        - Base directory
        - Config directory
        - Data directory
        
        Also loads the keywords configuration on initialization.
        """
        # Get absolute paths
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, "config")
        self.data_path = os.path.join(self.base_path, "data")
        self.keywords = self._load_keywords()
        
    def _load_keywords(self) -> Dict[str, Any]:
        """
        Load keywords and domain-specific vocabulary from keywords.json.
        
        Returns:
            Dictionary containing keywords configuration including:
            - Core concepts and their definitions
            - Module-specific terminology
            - Business metrics and their descriptions
        """
        with open(os.path.join(self.config_path, "keywords.json"), "r") as f:
            return json.load(f)
    
    def get_relevant_datasets(self, query: str) -> List[str]:
        """
        Identify relevant datasets based on the query and keywords.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of relevant dataset names
        """
        query = query.lower()
        relevant_datasets = []
        
        # Check for module-specific queries
        for module in self.keywords["modules"]:
            if module.lower() in query:
                relevant_datasets.append(f"module_{module.lower().replace(' ', '_')}")
        
        # Check for metric-related queries
        for metric in self.keywords["metrics"]:
            if metric.lower() in query:
                relevant_datasets.append(f"metrics_{metric}")
        
        # Check for time-based analysis
        for term in self.keywords["time_related_terms"]:
            if term.lower() in query:
                relevant_datasets.append("time_series_data")
        
        # Add performance data if performance indicators are mentioned
        for indicator in self.keywords["performance_indicators"]:
            if indicator.lower() in query:
                relevant_datasets.append("performance_metrics")
        
        return list(set(relevant_datasets))
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query by expanding domain-specific terms.
        
        Args:
            query: Original user query
            
        Returns:
            Preprocessed query with expanded terms
        """
        query = query.lower()
        
        # Replace abbreviations and domain terms with their full forms
        for concept, description in self.keywords["core_concepts"].items():
            query = query.replace(concept.lower(), f"{concept} ({description})")
        
        return query

    def get_time_context(self, query: str) -> Dict[str, Any]:
        """
        Extract time-related context from the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing time context information
        """
        current_time = datetime.now()
        time_context = {
            "has_time_reference": False,
            "time_frame": None,
            "reference_date": current_time
        }
        
        # Check for time-related terms
        for term in self.keywords["time_related_terms"]:
            if term.lower() in query.lower():
                time_context["has_time_reference"] = True
                time_context["time_frame"] = term
                break
        
        return time_context
