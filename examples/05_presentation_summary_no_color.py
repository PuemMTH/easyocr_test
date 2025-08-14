"""
OCR Comprehensive Summary Presentation (No Color Version)
การแสดงผลสรุปครบถ้วนของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(no_color=True)

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

    console.rule("OCR EVALUATION - EXECUTIVE SUMMARY")

    # Overall statistics
    total_evaluations = len(df)
    overall_mean_cer = df['cer'].mean()
    models_count = df['model_name'].nunique()
    datasets_count = df['dataset_name'].nunique()

    overall_table = Table(show_header=False, box=None)
    overall_table.add_column("Field", style="bold")
    overall_table.add_column("Value")
    overall_table.add_row("Total Evaluations", f"{total_evaluations:,}")
    overall_table.add_row("Models Evaluated", str(models_count))
    overall_table.add_row("Datasets Used", str(datasets_count))
    overall_table.add_row("Overall Mean CER", f"{overall_mean_cer*100:.2f}%")
    console.print(Panel(overall_table, title="OVERALL EVALUATION STATISTICS", expand=False))

    # Best and worst performers
    best_performance = summary_df.loc[summary_df['mean_cer'].idxmin()]
    worst_performance = summary_df.loc[summary_df['mean_cer'].idxmax()]

    best_table = Table(show_header=False, box=None)
    best_table.add_column("Field", style="bold")
    best_table.add_column("Value")
    best_table.add_row("Model", best_performance['model_name'])
    best_table.add_row("Dataset", best_performance['dataset_name'])
    best_table.add_row("Mean CER", f"{best_performance['mean_cer']*100:.2f}%")
    best_table.add_row("Rating", best_performance['rating'])
    console.print(Panel(best_table, title="BEST PERFORMANCE", expand=False))

    worst_table = Table(show_header=False, box=None)
    worst_table.add_column("Field", style="bold")
    worst_table.add_column("Value")
    worst_table.add_row("Model", worst_performance['model_name'])
    worst_table.add_row("Dataset", worst_performance['dataset_name'])
    worst_table.add_row("Mean CER", f"{worst_performance['mean_cer']*100:.2f}%")
    worst_table.add_row("Rating", worst_performance['rating'])
    console.print(Panel(worst_table, title="WORST PERFORMANCE", expand=False))

def present_detailed_summary(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present detailed summary for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    console.print("")
    console.rule("DETAILED PERFORMANCE SUMMARY")

    for _, row in summary_df.iterrows():
        console.print("")
        header = Table(box=None)
        header.add_column("Field", style="bold")
        header.add_column("Value")
        header.add_row("Model", row['model_name'])
        header.add_row("Dataset", row['dataset_name'])
        header.add_row("Total Samples", str(row['total_samples']))
        console.print(Panel(header, title="Overview", expand=False))

        stats = Table(box=None)
        stats.add_column("CER Metric", style="bold")
        stats.add_column("Value")
        stats.add_row("Mean", f"{row['mean_cer']*100:.2f}%")
        stats.add_row("Std", f"{row['std_cer']*100:.2f}%")
        stats.add_row("Min", f"{row['min_cer']*100:.2f}%")
        stats.add_row("Max", f"{row['max_cer']*100:.2f}%")
        stats.add_row("Median", f"{row['median_cer']*100:.2f}%")
        console.print(Panel(stats, title="CER Statistics", expand=False))

        acc = Table(box=None)
        acc.add_column("Accuracy Bucket", style="bold")
        acc.add_column("Count")
        acc.add_column("Percent")
        acc.add_row("Perfect (0%)", str(row['perfect_matches']), f"{row['perfect_matches_pct']:.1f}%")
        acc.add_row("High (<10%)", str(row['high_accuracy']), f"{row['high_accuracy_pct']:.1f}%")
        acc.add_row("Medium (10-50%)", str(row['medium_accuracy']), f"{row['medium_accuracy_pct']:.1f}%")
        acc.add_row("Low (>=50%)", str(row['low_accuracy']), f"{row['low_accuracy_pct']:.1f}%")
        console.print(Panel(acc, title="Accuracy Distribution", expand=False))

        console.print(f"\nOverall Performance Rating: {row['rating']}")

def present_recommendations(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present recommendations based on evaluation results (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    console.print("")
    console.rule("RECOMMENDATIONS BASED ON EVALUATION RESULTS")

    # Find best models for each dataset
    for dataset in summary_df['dataset_name'].unique():
        dataset_summary = summary_df[summary_df['dataset_name'] == dataset]
        best_model = dataset_summary.loc[dataset_summary['mean_cer'].idxmin()]

        tbl = Table(box=None)
        tbl.add_column("Field", style="bold")
        tbl.add_column("Value")
        tbl.add_row("Dataset", dataset)
        tbl.add_row("Recommended Model", best_model['model_name'])
        tbl.add_row("Performance", f"{best_model['mean_cer']*100:.2f}% CER")
        tbl.add_row("Rating", best_model['rating'])
        console.print(Panel(tbl, title="Dataset Recommendation", expand=False))

    # Overall recommendations
    console.print("\nOVERALL RECOMMENDATIONS:")

    excellent_models = summary_df[summary_df['rating'] == 'EXCELLENT']
    if not excellent_models.empty:
        console.print("\tExcellent performing models found:")
        for _, model in excellent_models.iterrows():
            console.print(f"\t\t- {model['model_name']} on {model['dataset_name']} ({model['mean_cer']*100:.2f}% CER)")

    poor_models = summary_df[summary_df['rating'] == 'POOR']
    if not poor_models.empty:
        console.print("\tModels needing improvement:")
        for _, model in poor_models.iterrows():
            console.print(f"\t\t- {model['model_name']} on {model['dataset_name']} ({model['mean_cer']*100:.2f}% CER)")

def present_key_insights(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present key insights from the evaluation (no color)"""
    df = load_evaluation_data(csv_path)
    summary_df = calculate_comprehensive_metrics(df)

    console.print("")
    console.rule("KEY INSIGHTS FROM OCR EVALUATION")

    # Performance distribution
    rating_counts = summary_df['rating'].value_counts()
    console.print("\nPerformance Distribution:")
    for rating, count in rating_counts.items():
        console.print(f"\t{rating}: {count} model-dataset combinations")

    # Dataset performance comparison
    dataset_performance = summary_df.groupby('dataset_name')['mean_cer'].mean()
    console.print("\nDataset Performance Comparison:")
    for dataset, mean_cer in dataset_performance.items():
        console.print(f"\t{dataset}: {mean_cer*100:.2f}% average CER")

    # Model performance comparison
    model_performance = summary_df.groupby('model_name')['mean_cer'].mean()
    console.print("\nModel Performance Comparison:")
    for model, mean_cer in model_performance.items():
        console.print(f"\t{model}: {mean_cer*100:.2f}% average CER")

if __name__ == "__main__":
    present_executive_summary()
    present_detailed_summary()
    present_recommendations()
    present_key_insights() 