import os
import random
import matplotlib.pyplot as plt
import easyocr
import csv
import pandas as pd
import torch
from datetime import datetime
from utils.config import get_model_configurations, get_dataset_configurations
from utils.dataset_loader import load_dataset, create_sample_dataset, create_balanced_sample_dataset
from utils.ocr_evaluator import run_ocr_evaluation, calculate_metrics, print_summary_statistics

print("Program..start")

os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Pre-load semantic model with local path
print("🔄 Pre-loading semantic model...")
try:
    from utils.ocr_evaluator import get_metrics_instance
    # Use global metrics instance with offline support
    _ = get_metrics_instance()
    print("✅ OCR metrics initialized (with offline support)")
except Exception as e:
    print(f"⚠️  OCR metrics initialization failed: {e}")

models_to_test = get_model_configurations(models=[
    {
        'name': 'base_model',
        'reader': easyocr.Reader(
            lang_list=['en', 'th'],
            detector=False,
            gpu=True,
            verbose=False,
            download_enabled=False
        )
    },
    {
        'name': 'out_source_merge_kaggle_data_3gpus',
        'reader': easyocr.Reader(
            lang_list=['en','th'],
            detector=False,
            gpu=True,
            verbose=False,
            download_enabled=False,
            recog_network='out_source_merge_kaggle_data_3gpus',
            model_storage_directory='./custom_example/model',
            user_network_directory='./custom_example/user_network'
        )
    },
    {
        'name': 'out_source_merge_kaggle_data_freeze_sequence_config_3gpus',
        'reader': easyocr.Reader(
            lang_list=['en','th'],
            detector=False,
            gpu=True,
            verbose=False,
            download_enabled=False,
            recog_network='out_source_merge_kaggle_data_freeze_sequence_config_3gpus',
            model_storage_directory='./custom_example/model',
            user_network_directory='./custom_example/user_network'
        )
    },
    {
        'name': 'out_source_only_data_3gpus',
        'reader': easyocr.Reader(
            lang_list=['en','th'],
            detector=False,
            gpu=True,
            verbose=False,
            download_enabled=False,
            recog_network='out_source_only_data_3gpus',
            model_storage_directory='./custom_example/model',
            user_network_directory='./custom_example/user_network'
        )
    }
])
datasets_to_test = get_dataset_configurations(datasets=[
    {
        'name': 'lmdb_test',
        'type': 'lmdb_format',
        'folder_path': '/project/lt200384-ff_bio/puem/ocr/dataset/processed_lmdb_format/test',
        'df': pd.read_csv('/project/lt200384-ff_bio/puem/ocr/dataset/processed_lmdb_format/test/labels.csv'),
    },
    {
        'name': 'kaggle_test',
        'type': 'lmdb_format',
        'folder_path': '/project/lt200384-ff_bio/puem/ocr/dataset/kaggle_test',
        'df': pd.read_csv('/project/lt200384-ff_bio/puem/ocr/dataset/kaggle_test/labels.csv'),
    }
])

EVALUATION_SAMPLE_SIZE = -1
USE_BALANCED_SAMPLING = True

all_results = []
for dataset in datasets_to_test:
    print(f"\nProcessing dataset: {dataset['name']}")
    
    if USE_BALANCED_SAMPLING and EVALUATION_SAMPLE_SIZE != -1:
        test_dataset = create_balanced_sample_dataset(dataset, EVALUATION_SAMPLE_SIZE)
        if 'estimated_images' in test_dataset:
            print(f"🎯 Balanced sampling: Using ~{test_dataset['estimated_images']} images to target {EVALUATION_SAMPLE_SIZE} samples")
    else:
        test_dataset = create_sample_dataset(dataset, EVALUATION_SAMPLE_SIZE)
    dataset_results = load_dataset(test_dataset, test_dataset['type'])
    
    if len(dataset_results) == 0:
        print(f"⚠️  No data loaded for {dataset['name']}")
        continue
    
    expected_msg = "full dataset" if EVALUATION_SAMPLE_SIZE == -1 else f"~{EVALUATION_SAMPLE_SIZE} samples"
    print(f"📊 Using {len(dataset_results)} samples for evaluation (target: {expected_msg})")
    
    # Run optimized OCR evaluation with all models
    print("🚀 Starting optimized OCR evaluation...")
    model_results = run_ocr_evaluation(models_to_test, dataset_results, dataset['name'])
    all_results.extend(model_results)
    

print(f"\n✓ Total processed results: {len(all_results)}")
sampling_type = "Balanced" if USE_BALANCED_SAMPLING else "Standard"
print(f"📈 {sampling_type} sampling for fairer comparison across datasets")


# print(all_results)
types_list = [result['dataset_name'] for result in all_results]
print(f"🔍 Types of results collected: {set(types_list)}")

img = [x for x in all_results if x['dataset_name'] == 'data_from_outsource' and x['model_name'] == 'base_model']
random.seed(47)
img = random.sample(img, min(5, len(img)))
# for i, in enumerate(img):
for index, val in enumerate(img):
    plt.figure(figsize=(10, 3))
    plt.imshow(val['cropped_image'])
    plt.axis('off')
    os.makedirs("tmp", exist_ok=True)
    plt.savefig(f"output/sample_image_{index+1}_{val['dataset_name']}_{val['model_name']}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    
print("=== METRICS CALCULATION ===")
if len(all_results) > 0:
    metrics_output_path = f'./output/evaluation_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    individual_metrics = calculate_metrics(all_results, metrics_output_path)
    print_summary_statistics(individual_metrics)
    print(f"\n✓ Evaluation complete! Results saved to {metrics_output_path}")
else:
    print("⚠️  No results to calculate metrics")