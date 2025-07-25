"""
Dataset loading utilities for OCR evaluation
"""
import os
import pandas as pd
from PIL import Image
from typing import List, Dict, Any
import tqdm
from .ocr_data_models import OCRData


def load_dataset(dataset: Dict, type_dataset: str) -> List[Dict[str, Any]]:
    """
    Load dataset based on the specified type and return cropped images with ground truth text.

    Args:
        dataset (Dict): The dataset configuration containing 'df' and 'folder_path'.
        type_dataset (str): The type of dataset to load ('bbox_json' or 'full_image').

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing 'cropped_image' and 'ground_truth' keys.
    """
    results = []
    df = dataset['df']
    folder_path = dataset['folder_path']
    
    # Check if there's a target sample size limit
    target_sample_size = dataset.get('target_sample_size', -1)
    
    print(f"Loading dataset: {dataset['name']} (type: {type_dataset})")
    print(f"DataFrame shape: {df.shape}")
    
    if type_dataset == 'bbox_json':
        results = _load_bbox_json_dataset(df, folder_path, dataset['name'], target_sample_size)
    elif type_dataset == 'full_image':
        results = _load_full_image_dataset(df, folder_path, dataset['name'])
    elif type_dataset == 'lmdb_format':
        results = _load_lmdb_format_dataset(df, folder_path, dataset['name'])
    else:
        raise ValueError(f"Unknown dataset type: {type_dataset}")
    
    print(f"Loaded {len(results)} samples")
    return results


def _load_bbox_json_dataset(df: pd.DataFrame, folder_path: str, dataset_name: str, target_sample_size: int = -1) -> List[Dict[str, Any]]:
    """Load dataset with bounding box annotations from JSON files"""
    results = []
    
    # Find appropriate columns
    json_columns = [col for col in df.columns if 'json' in col.lower()]
    image_columns = [col for col in df.columns if 'image' in col.lower()]
    
    json_file_column = 'new_json_name' if 'new_json_name' in json_columns else json_columns[0]
    image_file_column = 'new_image_name' if 'new_image_name' in image_columns else image_columns[0]
    
    print(f"Using JSON column: '{json_file_column}', Image column: '{image_file_column}'")
    
    # Filter JSON files
    pd_files = df[df[json_file_column].str.contains('.json', na=False)]
    
    for index, row in tqdm.tqdm(pd_files.iterrows(), total=len(pd_files), desc="Processing bbox_json"):
        json_path = os.path.join(folder_path, row[json_file_column])
        image_path = os.path.join(folder_path, row[image_file_column])
        
        if not os.path.exists(json_path) or not os.path.exists(image_path):
            continue
            
        try:
            ocr_data = OCRData.from_json(json_path=json_path)
            image = Image.open(image_path)
            
            if len(ocr_data.programs) == 0:
                continue
                
            for i, box in enumerate(ocr_data.programs[0].frames[0].text_regions):
                # Stop if we've reached the target sample size
                if target_sample_size > 0 and len(results) >= target_sample_size:
                    return results
                    
                cropped_image = image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
                ground_truth = box.text.replace("\n", "")
                
                results.append({
                    'cropped_image': cropped_image,
                    'ground_truth': ground_truth,
                    'source_file': row[json_file_column],
                    'box_index': i
                })
        except Exception as e:
            print(f"Error processing {row[json_file_column]}: {e}")
            continue
    
    return results


def _load_full_image_dataset(df: pd.DataFrame, folder_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load dataset with full images and ground truth text"""
    results = []
    
    # Find appropriate columns
    file_path_column = None
    for col in df.columns:
        if 'file' in col.lower() or 'name' in col.lower():
            file_path_column = col
            break
    
    text_column = None
    for col in df.columns:
        if col.lower() in ['text', 'label', 'ground_truth', 'gt', 'caption', 'words']:
            text_column = col
            break
    
    if file_path_column is None:
        file_path_column = df.columns[0]
    if text_column is None:
        text_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    print(f"Using file column: '{file_path_column}', text column: '{text_column}'")
    
    # Filter image files
    pd_files = df[df[file_path_column].str.contains('.jpg|.png|.jpeg', case=False, na=False)]
    
    for index, row in tqdm.tqdm(pd_files.iterrows(), total=len(pd_files), desc="Processing full_image"):
        image_path = os.path.join(folder_path, row[file_path_column])
        
        if not os.path.exists(image_path):
            continue
            
        try:
            image = Image.open(image_path)
            ground_truth = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            results.append({
                'cropped_image': image,
                'ground_truth': ground_truth,
                'source_file': row[file_path_column],
                'box_index': 0
            })
        except Exception:
            continue
    
    return results


def _load_lmdb_format_dataset(df: pd.DataFrame, folder_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load dataset in LMDB format with separate image files and CSV labels"""
    results = []
    
    # Find appropriate columns - typically 'filename' and 'words' for LMDB format
    filename_column = None
    for col in df.columns:
        if col.lower() in ['filename', 'file', 'image', 'name', 'path']:
            filename_column = col
            break
    
    text_column = None
    for col in df.columns:
        if col.lower() in ['words', 'text', 'label', 'ground_truth', 'gt']:
            text_column = col
            break
    
    if filename_column is None:
        filename_column = df.columns[0]
    if text_column is None:
        text_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    print(f"Using filename column: '{filename_column}', text column: '{text_column}'")
    
    # Process each row in the dataframe
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing lmdb_format"):
        filename = row[filename_column]
        if pd.isna(filename):
            continue
            
        # Try different image extensions if the filename doesn't have one
        image_path = None
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
            # Try common extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                test_path = os.path.join(folder_path, filename + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        else:
            image_path = os.path.join(folder_path, filename)
        
        if image_path is None or not os.path.exists(image_path):
            print(f"Warning: Image not found for {filename}")
            continue
            
        try:
            image = Image.open(image_path)
            ground_truth = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            results.append({
                'cropped_image': image,
                'ground_truth': ground_truth,
                'source_file': filename,
                'box_index': 0  # LMDB format typically has one text per image
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return results


def create_sample_dataset(dataset: Dict, sample_size: int) -> Dict:
    """
    Create a smaller sample of the dataset for testing
    
    Args:
        dataset: Original dataset configuration
        sample_size: Number of samples to take (-1 for full dataset)
    
    Returns:
        Dataset configuration with reduced sample size
    """
    # if -1, return the full dataset
    if sample_size == -1:
        return dataset
    
    sample_dataset = dataset.copy()
    sample_dataset['df'] = dataset['df'].head(sample_size)
    sample_dataset['target_sample_size'] = sample_size  # Track target for reporting
    return sample_dataset


def create_balanced_sample_dataset(dataset: Dict, target_samples: int) -> Dict:
    """
    Create a sample that aims for a specific number of final samples
    
    For bbox_json datasets, this estimates how many images are needed
    to get approximately target_samples text regions.
    
    Args:
        dataset: Original dataset configuration
        target_samples: Target number of final samples
    
    Returns:
        Dataset configuration optimized for target sample count
    """
    if target_samples == -1:
        return dataset
    
    sample_dataset = dataset.copy()
    
    if dataset.get('type') == 'bbox_json':
        # For bbox datasets, be more conservative with estimation
        # Use more images but limit the final sample count
        if target_samples <= 10:
            # For small samples, use more images to ensure we get enough
            estimated_images_needed = max(target_samples, 3)
        else:
            # For larger samples, estimate ~3-4 text regions per image on average
            estimated_images_needed = max(1, target_samples // 3)
        
        sample_dataset['df'] = dataset['df'].head(estimated_images_needed)
        sample_dataset['target_sample_size'] = target_samples
        sample_dataset['estimated_images'] = estimated_images_needed
        
        print(f"🎯 Balanced sampling for bbox_json: {estimated_images_needed} images → target {target_samples} samples")
    else:
        # For full image datasets, 1 image = 1 sample
        sample_dataset['df'] = dataset['df'].head(target_samples)
        sample_dataset['target_sample_size'] = target_samples
    
    return sample_dataset
