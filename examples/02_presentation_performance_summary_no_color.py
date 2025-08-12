"""
OCR Performance Summary Presentation (No Color Version)
การแสดงผลสรุปประสิทธิภาพของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import numpy as np

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics for each model-dataset combination"""
    performance_data = []

    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            # Calculate metrics
            total_samples = len(subset)
            mean_cer = subset['cer'].mean()
            std_cer = subset['cer'].std()
            min_cer = subset['cer'].min()
            max_cer = subset['cer'].max()
            median_cer = subset['cer'].median()

            # Calculate accuracy distribution
            perfect_matches = len(subset[subset['cer'] == 0])
            high_accuracy = len(subset[subset['cer'] < 0.1])
            medium_accuracy = len(subset[(subset['cer'] >= 0.1) & (subset['cer'] < 0.5)])
            low_accuracy = len(subset[subset['cer'] >= 0.5])

            performance_data.append({
                'model_name': model,
                'dataset_name': dataset,
                'total_samples': total_samples,
                'mean_cer': mean_cer,
                'std_cer': std_cer,
                'min_cer': min_cer,
                'max_cer': max_cer,
                'median_cer': median_cer,
                'perfect_matches': perfect_matches,
                'perfect_matches_pct': (perfect_matches / total_samples) * 100,
                'high_accuracy': high_accuracy,
                'high_accuracy_pct': (high_accuracy / total_samples) * 100,
                'medium_accuracy': medium_accuracy,
                'medium_accuracy_pct': (medium_accuracy / total_samples) * 100,
                'low_accuracy': low_accuracy,
                'low_accuracy_pct': (low_accuracy / total_samples) * 100
            })

    return pd.DataFrame(performance_data)

def present_performance_summary(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present performance summary for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    performance_df = calculate_performance_metrics(df)

    print("="*100)
    print("OCR PERFORMANCE SUMMARY - PRESENTATION")
    print("="*100)

    for _, row in performance_df.iterrows():
        print(f"\nModel: {row['model_name']}")
        print(f"Dataset: {row['dataset_name']}")
        print(f"Total Samples: {row['total_samples']}")

        # CER Statistics
        print(f"\nCER Statistics:")
        print(f"\tMean CER: {row['mean_cer']*100:.2f}%")
        print(f"\tStd CER: {row['std_cer']*100:.2f}%")
        print(f"\tMin CER: {row['min_cer']*100:.2f}%")
        print(f"\tMax CER: {row['max_cer']*100:.2f}%")
        print(f"\tMedian CER: {row['median_cer']*100:.2f}%")

        # Accuracy Distribution
        print(f"\nAccuracy Distribution:")
        print(f"\tPerfect Matches (CER=0%): {row['perfect_matches']} ({row['perfect_matches_pct']:.1f}%)")
        print(f"\tHigh Accuracy (CER<10%): {row['high_accuracy']} ({row['high_accuracy_pct']:.1f}%)")
        print(f"\tMedium Accuracy (10%≤CER<50%): {row['medium_accuracy']} ({row['medium_accuracy_pct']:.1f}%)")
        print(f"\tLow Accuracy (CER≥50%): {row['low_accuracy']} ({row['low_accuracy_pct']:.1f}%)")

        # Performance Rating
        mean_cer = row['mean_cer']
        if mean_cer < 0.1:
            rating = "EXCELLENT"
        elif mean_cer < 0.2:
            rating = "GOOD"
        elif mean_cer < 0.3:
            rating = "FAIR"
        else:
            rating = "POOR"

        print(f"\nOverall Performance Rating: {rating}")

def find_best_performers(df):
    """Find and present the best performing models"""
    performance_df = calculate_performance_metrics(df)

    print("\n" + "="*80)
    print("BEST PERFORMING MODELS BY DATASET")
    print("="*80)

    for dataset in performance_df['dataset_name'].unique():
        dataset_performance = performance_df[performance_df['dataset_name'] == dataset]
        best_model = dataset_performance.loc[dataset_performance['mean_cer'].idxmin()]

        print(f"\nDataset: {dataset}")
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best Mean CER: {best_model['mean_cer']*100:.2f}%")
        print(f"Perfect Matches: {best_model['perfect_matches_pct']:.1f}%")

if __name__ == "__main__":
    present_performance_summary()
    # Also show best performers
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    find_best_performers(df) 