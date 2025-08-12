"""
OCR Comprehensive Summary Presentation (No Color Version)
การแสดงผลสรุปครบถ้วนของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import numpy as np

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def calculate_comprehensive_metrics(df):
    """Calculate comprehensive metrics for presentation"""
    summary_data = []

    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            # Basic metrics
            total_samples = len(subset)
            mean_cer = subset['cer'].mean()
            std_cer = subset['cer'].std()
            min_cer = subset['cer'].min()
            max_cer = subset['cer'].max()
            median_cer = subset['cer'].median()

            # Accuracy distribution
            perfect_matches = len(subset[subset['cer'] == 0])
            high_accuracy = len(subset[subset['cer'] < 0.1])
            medium_accuracy = len(subset[(subset['cer'] >= 0.1) & (subset['cer'] < 0.5)])
            low_accuracy = len(subset[subset['cer'] >= 0.5])

            # Performance rating
            if mean_cer < 0.1:
                rating = "EXCELLENT"
            elif mean_cer < 0.2:
                rating = "GOOD"
            elif mean_cer < 0.3:
                rating = "FAIR"
            else:
                rating = "POOR"

            summary_data.append({
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
                'low_accuracy_pct': (low_accuracy / total_samples) * 100,
                'rating': rating
            })

    return pd.DataFrame(summary_data)

def present_executive_summary(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present executive summary for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    print("="*120)
    print("OCR EVALUATION - EXECUTIVE SUMMARY")
    print("="*120)

    # Overall statistics
    total_evaluations = len(df)
    overall_mean_cer = df['cer'].mean()
    models_count = df['model_name'].nunique()
    datasets_count = df['dataset_name'].nunique()

    print(f"\nOVERALL EVALUATION STATISTICS:")
    print(f"\tTotal Evaluations: {total_evaluations:,}")
    print(f"\tModels Evaluated: {models_count}")
    print(f"\tDatasets Used: {datasets_count}")
    print(f"\tOverall Mean CER: {overall_mean_cer*100:.2f}%")

    # Best and worst performers
    best_performance = summary_df.loc[summary_df['mean_cer'].idxmin()]
    worst_performance = summary_df.loc[summary_df['mean_cer'].idxmax()]

    print(f"\nBEST PERFORMANCE:")
    print(f"\tModel: {best_performance['model_name']}")
    print(f"\tDataset: {best_performance['dataset_name']}")
    print(f"\tMean CER: {best_performance['mean_cer']*100:.2f}%")
    print(f"\tRating: {best_performance['rating']}")

    print(f"\nWORST PERFORMANCE:")
    print(f"\tModel: {worst_performance['model_name']}")
    print(f"\tDataset: {worst_performance['dataset_name']}")
    print(f"\tMean CER: {worst_performance['mean_cer']*100:.2f}%")
    print(f"\tRating: {worst_performance['rating']}")

def present_detailed_summary(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present detailed summary for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    print(f"\n" + "="*120)
    print("DETAILED PERFORMANCE SUMMARY")
    print("="*120)

    for _, row in summary_df.iterrows():
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
        print(f"\nOverall Performance Rating: {row['rating']}")

def present_recommendations(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present recommendations based on evaluation results (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    print(f"\n" + "="*120)
    print("RECOMMENDATIONS BASED ON EVALUATION RESULTS")
    print("="*120)

    # Find best models for each dataset
    for dataset in summary_df['dataset_name'].unique():
        dataset_summary = summary_df[summary_df['dataset_name'] == dataset]
        best_model = dataset_summary.loc[dataset_summary['mean_cer'].idxmin()]

        print(f"\nDataset: {dataset}")
        print(f"Recommended Model: {best_model['model_name']}")
        print(f"Performance: {best_model['mean_cer']*100:.2f}% CER")
        print(f"Rating: {best_model['rating']}")

    # Overall recommendations
    print(f"\nOVERALL RECOMMENDATIONS:")

    excellent_models = summary_df[summary_df['rating'] == 'EXCELLENT']
    if not excellent_models.empty:
        print(f"\tExcellent performing models found:")
        for _, model in excellent_models.iterrows():
            print(f"\t\t- {model['model_name']} on {model['dataset_name']} ({model['mean_cer']*100:.2f}% CER)")

    poor_models = summary_df[summary_df['rating'] == 'POOR']
    if not poor_models.empty:
        print(f"\tModels needing improvement:")
        for _, model in poor_models.iterrows():
            print(f"\t\t- {model['model_name']} on {model['dataset_name']} ({model['mean_cer']*100:.2f}% CER)")

def present_key_insights(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present key insights from the evaluation (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    print(f"\n" + "="*120)
    print("KEY INSIGHTS FROM OCR EVALUATION")
    print("="*120)

    # Performance distribution
    rating_counts = summary_df['rating'].value_counts()
    print(f"\nPerformance Distribution:")
    for rating, count in rating_counts.items():
        print(f"\t{rating}: {count} model-dataset combinations")

    # Dataset performance comparison
    dataset_performance = summary_df.groupby('dataset_name')['mean_cer'].mean()
    print(f"\nDataset Performance Comparison:")
    for dataset, mean_cer in dataset_performance.items():
        print(f"\t{dataset}: {mean_cer*100:.2f}% average CER")

    # Model performance comparison
    model_performance = summary_df.groupby('model_name')['mean_cer'].mean()
    print(f"\nModel Performance Comparison:")
    for model, mean_cer in model_performance.items():
        print(f"\t{model}: {mean_cer*100:.2f}% average CER")

if __name__ == "__main__":
    present_executive_summary()
    present_detailed_summary()
    present_recommendations()
    present_key_insights() 