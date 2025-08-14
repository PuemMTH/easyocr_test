"""
OCR Performance Summary Presentation (With Color)
การแสดงผลสรุปประสิทธิภาพของ OCR สำหรับการนำเสนอ (เวอร์ชันมีสี)
"""

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

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

def get_rating_color(rating):
    """Get color for rating"""
    colors = {
        "EXCELLENT": "bright_green",
        "GOOD": "green", 
        "FAIR": "yellow",
        "POOR": "red"
    }
    return colors.get(rating, "white")

def get_percentage_color(percentage):
    """Get color for percentage values"""
    if percentage >= 80:
        return "bright_green"
    elif percentage >= 60:
        return "green"
    elif percentage >= 40:
        return "yellow"
    elif percentage >= 20:
        return "orange3"
    else:
        return "red"

def present_performance_summary(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present performance summary for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    performance_df = calculate_performance_metrics(df)

    console.rule("OCR PERFORMANCE SUMMARY - PRESENTATION")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Dataset")
    table.add_column("Total", justify="right")
    table.add_column("Mean %", justify="right")
    table.add_column("Std %", justify="right")
    table.add_column("Min %", justify="right")
    table.add_column("Max %", justify="right")
    table.add_column("Median %", justify="right")
    table.add_column("Perfect %", justify="right")
    table.add_column("High %", justify="right")
    table.add_column("Medium %", justify="right")
    table.add_column("Low %", justify="right")
    table.add_column("Rating")

    for _, row in performance_df.iterrows():
        mean_cer = row['mean_cer']
        if mean_cer < 0.1:
            rating = "EXCELLENT"
        elif mean_cer < 0.2:
            rating = "GOOD"
        elif mean_cer < 0.3:
            rating = "FAIR"
        else:
            rating = "POOR"

        perfect_pct = row['perfect_matches_pct']
        high_pct = row['high_accuracy_pct']
        medium_pct = row['medium_accuracy_pct']
        low_pct = row['low_accuracy_pct']

        table.add_row(
            str(row['model_name']),
            str(row['dataset_name']),
            f"{int(row['total_samples'])}",
            f"[{get_rating_color(rating)}]{row['mean_cer']*100:.2f}[/]",
            f"{row['std_cer']*100:.2f}",
            f"{row['min_cer']*100:.2f}",
            f"{row['max_cer']*100:.2f}",
            f"{row['median_cer']*100:.2f}",
            f"[{get_percentage_color(perfect_pct)}]{perfect_pct:.1f}[/]",
            f"[{get_percentage_color(high_pct)}]{high_pct:.1f}[/]",
            f"[{get_percentage_color(medium_pct)}]{medium_pct:.1f}[/]",
            f"[{get_percentage_color(low_pct)}]{low_pct:.1f}[/]",
            f"[{get_rating_color(rating)}]{rating}[/]",
        )

    console.print(table)

def present_accuracy_distribution(csv_path='reports/ocr_evaluation_20250813_002540/data/ocr_evaluation_detailed.csv'):
    """Present accuracy distribution for each model-dataset combination separated by dataset"""
    df = load_evaluation_data(csv_path)
    
    console.rule("ACCURACY DISTRIBUTION BY MODEL AND DATASET")
    
    for dataset in df['dataset_name'].unique():
        console.print("")
        console.print(f"[bold yellow]Dataset: {dataset}[/]")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Perfect (0%)", justify="right")
        table.add_column("High (<10%)", justify="right")
        table.add_column("Medium (10-50%)", justify="right")
        table.add_column("Low (≥50%)", justify="right")
        
        for model in df['model_name'].unique():
            subset = df.query("model_name == @model and dataset_name == @dataset")
            
            if subset.empty:
                continue
                
            total_samples = len(subset)
            
            # Calculate accuracy categories
            perfect_matches = len(subset[subset['cer'] == 0])
            high_accuracy = len(subset[subset['cer'] < 0.1])  # CER < 10%
            medium_accuracy = len(subset[(subset['cer'] >= 0.1) & (subset['cer'] < 0.5)])  # 10% ≤ CER < 50%
            low_accuracy = len(subset[subset['cer'] >= 0.5])  # CER ≥ 50%
            
            # Calculate percentages
            perfect_pct = (perfect_matches / total_samples) * 100
            high_pct = (high_accuracy / total_samples) * 100
            medium_pct = (medium_accuracy / total_samples) * 100
            low_pct = (low_accuracy / total_samples) * 100
            
            table.add_row(
                model,
                str(total_samples),
                f"[bright_green]{perfect_matches} ({perfect_pct:.1f}%)[/]",
                f"[green]{high_accuracy} ({high_pct:.1f}%)[/]",
                f"[yellow]{medium_accuracy} ({medium_pct:.1f}%)[/]",
                f"[red]{low_accuracy} ({low_pct:.1f}%)[/]"
            )
        
        console.print(table)

def find_best_performers(df):
    """Find and present the best performing models"""
    performance_df = calculate_performance_metrics(df)

    console.print("\n")
    console.rule("BEST PERFORMING MODELS BY DATASET")

    for dataset in performance_df['dataset_name'].unique():
        dataset_performance = performance_df[performance_df['dataset_name'] == dataset]
        best_model = dataset_performance.loc[dataset_performance['mean_cer'].idxmin()]

        console.print("")
        console.print(f"[bold]Dataset:[/] [cyan]{dataset}[/]")
        console.print(f"[bold]Best Model:[/] [green]{best_model['model_name']}[/]")
        console.print(f"[bold]Best Mean CER:[/] [bright_green]{best_model['mean_cer']*100:.2f}%[/]")
        console.print(f"[bold]Perfect Matches:[/] [bright_green]{best_model['perfect_matches_pct']:.1f}%[/]")

if __name__ == "__main__":
    present_performance_summary()
    # Also show best performers
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    find_best_performers(df) 