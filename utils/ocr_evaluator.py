"""
OCR evaluation utilities
"""
import numpy as np
from typing import List, Dict, Any
import tqdm
import json
import os
from .metrics import OCRMetrics

# Global metrics instance
_metrics_instance = None

def get_metrics_instance():
    """Get or create global metrics instance with offline support"""
    global _metrics_instance
    if _metrics_instance is None:
        print("🔄 Initializing OCR metrics...")
        # Try local model first, then online, then offline mode
        local_model_path = './models/distiluse-base-multilingual-cased'
        if os.path.exists(local_model_path):
            _metrics_instance = OCRMetrics(local_model_path=local_model_path)
        else:
            try:
                # Try with local_files_only=True first
                _metrics_instance = OCRMetrics(local_files_only=True)
            except Exception as e:
                print(f"⚠️  Local files only failed, trying online mode: {e}")
                try:
                    _metrics_instance = OCRMetrics()
                except Exception as e2:
                    print(f"⚠️  Using offline mode (semantic model disabled): {e2}")
                    _metrics_instance = OCRMetrics(semantic_model=None)
    return _metrics_instance


def run_ocr_evaluation(models: List[Dict], dataset_results: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    """
    Run OCR evaluation on dataset with multiple models
    
    Args:
        models: List of model configurations with 'name' and 'reader'
        dataset_results: List of dataset items with 'cropped_image' and 'ground_truth'
        dataset_name: Name of the dataset
        
    Returns:
        List of evaluation results
    """
    all_results = []
    
    for model in models:
        print(f"  Testing model: {model['name']}")
        model_results = []
        
        for data_item in tqdm.tqdm(dataset_results, desc=f"OCR with {model['name']}"):
            try:
                # Perform OCR on the cropped image
                ocr_matrix = model['reader'].recognize(np.array(data_item['cropped_image']))
                ocr_text = ocr_matrix[0][1] if ocr_matrix else "No text"
                
                result = {
                    'dataset_name': dataset_name,
                    'model_name': model['name'],
                    'source_file': data_item['source_file'],
                    'box_index': data_item['box_index'],
                    'cropped_image': data_item['cropped_image'],
                    'ground_truth': data_item['ground_truth'],
                    'ocr_text': ocr_text
                }
                model_results.append(result)
                
            except Exception as e:
                print(f"Error processing {data_item['source_file']}: {e}")
                continue
                
        all_results.extend(model_results)
        print(f"  Completed {len(model_results)} images")
    
    return all_results


def calculate_metrics(results: List[Dict[str, Any]], output_path: str = None) -> List[Dict[str, Any]]:
    """
    Calculate OCR metrics for evaluation results
    
    Args:
        results: List of OCR evaluation results
        output_path: Optional path to save metrics JSON file
        
    Returns:
        List of individual metrics
    """
    metrics = get_metrics_instance()  # Use global instance
    individual_metrics = []
    
    # Remove existing file if it exists
    if output_path and os.path.exists(output_path):
        os.remove(output_path)
    
    # Group results by dataset and model
    results_by_model = {}
    for result in results:
        key = f"{result['dataset_name']}_{result['model_name']}"
        if key not in results_by_model:
            results_by_model[key] = []
        results_by_model[key].append(result)
    
    # Calculate metrics for each model-dataset combination
    for key, model_results in results_by_model.items():
        print(f"Calculating metrics for {key}")
        
        for result in tqdm.tqdm(model_results, desc=f"Metrics for {key}"):
            metric_result = metrics.evaluate(
                reference=result['ground_truth'], 
                hypothesis=result['ocr_text']
            )
            
            # Add additional information to the metric result
            metric_result.update({
                'dataset_name': result['dataset_name'],
                'model_name': result['model_name'],
                'source_file': result['source_file'],
                'box_index': result['box_index'],
                'ground_truth': result['ground_truth'],
                'ocr_text': result['ocr_text']
            })
            
            individual_metrics.append(metric_result)
    
    # Save metrics to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(individual_metrics, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(individual_metrics)} metric results to {output_path}")
    
    return individual_metrics


def print_summary_statistics(individual_metrics: List[Dict[str, Any]]):
    """Print summary statistics for the evaluation"""
    # Group by model-dataset combination
    results_by_key = {}
    for metric in individual_metrics:
        key = f"{metric['dataset_name']}_{metric['model_name']}"
        if key not in results_by_key:
            results_by_key[key] = []
        results_by_key[key].append(metric)
    
    print("\n=== EVALUATION SUMMARY ===")
    for key, metrics in results_by_key.items():
        if metrics:
            # Calculate character accuracy from CER (Character Error Rate)
            avg_cer = np.mean([m['cer_percent'] for m in metrics])
            avg_character_accuracy = 100 - avg_cer  # Accuracy = 100% - Error Rate
            
            avg_wer = np.mean([m['wer_percent'] for m in metrics])
            avg_wer_pythainlp = np.mean([m['wer_pythainlp_percent'] for m in metrics])
            avg_semantic_sim = np.mean([m['semantic_similarity'] for m in metrics])
            
            print(f"{key}:")
            print(f"  - Character Accuracy: {avg_character_accuracy:.2f}%")
            print(f"  - Character Error Rate: {avg_cer:.2f}%")
            print(f"  - Word Error Rate: {avg_wer:.2f}%")
            print(f"  - Word Error Rate (Thai): {avg_wer_pythainlp:.2f}%")
            print(f"  - Semantic Similarity: {avg_semantic_sim:.4f}")
            print(f"  - Total samples: {len(metrics)}")
            print()
