"""
Configuration settings for OCR evaluation
"""
import easyocr
import pandas as pd
from typing import List, Dict, Any


def get_model_configurations() -> List[Dict[str, Any]]:
    """Get OCR model configurations for testing"""
    models = [
        {
            'name': 'base_model',
            'reader': easyocr.Reader(
                lang_list=['en', 'th'],
                detector=False,
                gpu=True,
                download_enabled=False
            )
        },
        {
            'name': 'custom_thai_1500_iteration',
            'reader': easyocr.Reader(
                lang_list=['en', 'th'],
                detector=False,
                gpu=True,
                download_enabled=False,
                recog_network='custom_thai_1500_iteration',
                model_storage_directory='./custom_example/model',
                user_network_directory='./custom_example/user_network'
            )
        },
        {
            'name': 'custom_thai_15000_iteration',
            'reader': easyocr.Reader(
                lang_list=['en', 'th'],
                detector=False,
                gpu=True,
                download_enabled=False,
                recog_network='custom_thai_15000_iteration',
                model_storage_directory='./custom_example/model',
                user_network_directory='./custom_example/user_network'
            )
        }
    ]
    
    print(f"Loaded {len(models)} models for testing:")
    for model in models:
        print(f"- {model['name']}")
    
    return models


def get_dataset_configurations() -> List[Dict[str, Any]]:
    """Get dataset configurations for evaluation"""
    datasets = [
        {
            'name': 'data_from_outsource',
            'type': 'bbox_json',
            'folder_path': '/Volumes/BACKUP/data/processed',
            'df': pd.read_csv('/Volumes/BACKUP/data/processed/file_mapping.csv'),
        },
        {
            'name': 'data_from_web',
            'type': 'full_image',
            'folder_path': './assests/input/test/',
            'df': pd.read_csv('./assests/input/test/labels.csv'),
        }
    ]
    
    print(f"Loaded {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"- {dataset['name']} ({dataset['type']}) - {len(dataset['df'])} files")
    
    return datasets
