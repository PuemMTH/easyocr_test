"""
Configuration settings for OCR evaluation
"""
import easyocr
import pandas as pd
from typing import List, Dict, Any


def get_model_configurations(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get OCR model configurations for testing"""
    if not models:
        print("No models provided for configuration.")
        return []

    print(f"Loaded {len(models)} models for testing:")
    for model in models:
        print(f"- {model['name']}")
    
    return models


def get_dataset_configurations(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get dataset configurations for evaluation"""
    if not datasets:
        print("No datasets provided for configuration.")
        return []
    
    print(f"Loaded {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"- {dataset['name']} ({dataset['type']}) - {len(dataset['df'])} files")
    
    return datasets