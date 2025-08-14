"""
OCR Comparison Table Presentation
การแสดงตารางเปรียบเทียบ OCR สำหรับการนำเสนอ
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import numpy as np  # noqa: F401 (may be useful later)

console = Console()  # allow rich styling

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def create_comparison_table(df):
    """Create a comprehensive comparison table for all model-dataset combinations"""
    # derive pivot and stats
    
    # Create pivot table for mean CER
    comparison_df = df.pivot_table(
        values='cer', 
        index='model_name', 
        columns='dataset_name', 
        aggfunc='mean'
    )
    
    # Add additional statistics
    stats_df = df.groupby(['model_name', 'dataset_name']).agg({
        'cer': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    
    return comparison_df, stats_df

def present_comparison_table(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present comparison table for presentation"""
    df = load_evaluation_data(csv_path)
    comparison_df, stats_df = create_comparison_table(df)

    console.rule("OCR MODEL COMPARISON TABLE - PRESENTATION")

    table = Table(title="Mean CER Comparison Matrix (%)", show_lines=False)
    table.add_column("Model", style="bold red")
    for dataset in comparison_df.columns:
        table.add_column(str(dataset))

    for model in comparison_df.index:
        cells = [Text(model, style="bold red")]
        for dataset in comparison_df.columns:
            cer = comparison_df.loc[model, dataset]
            if pd.isna(cer):
                cells.append("N/A")
            else:
                if cer < 0.1:
                    style = "green"
                elif cer < 0.2:
                    style = "yellow"
                elif cer < 0.3:
                    style = "cyan"
                else:
                    style = "red"
                cells.append(Text(f"{cer*100:.2f}%", style=style))
        table.add_row(*cells)
    console.print(table)

def present_detailed_statistics(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present detailed statistics for presentation"""
    df = load_evaluation_data(csv_path)
    _, stats_df = create_comparison_table(df)

    console.print("")
    console.rule("DETAILED STATISTICS - PRESENTATION")

    for (model, dataset), stats in stats_df.iterrows():
        mean_cer = stats[('cer', 'mean')]
        std_cer = stats[('cer', 'std')]
        min_cer = stats[('cer', 'min')]
        max_cer = stats[('cer', 'max')]

        info = Table(box=None)
        info.add_column("Field", style="bold")
        info.add_column("Value")
        info.add_row("Model", model)
        info.add_row("Dataset", str(dataset))
        info.add_row("Sample Count", str(stats[('cer', 'count')]))
        info.add_row("Mean CER", f"{mean_cer*100:.2f}%")
        info.add_row("Std CER", f"{std_cer*100:.2f}%")
        info.add_row("Min CER", f"{min_cer*100:.2f}%")
        info.add_row("Max CER", f"{max_cer*100:.2f}%")

        if mean_cer < 0.1:
            rating_style = "green"
            rating_text = "EXCELLENT"
        elif mean_cer < 0.2:
            rating_style = "yellow"
            rating_text = "GOOD"
        elif mean_cer < 0.3:
            rating_style = "cyan"
            rating_text = "FAIR"
        else:
            rating_style = "red"
            rating_text = "POOR"

        console.print(Panel(info, title="Statistics", expand=False))
        console.print(Text(f"Performance Rating: {rating_text}", style=rating_style))

def create_ranking_table(df):
    """Create a ranking table for all model-dataset combinations"""
    rankings = []
    
    for dataset in df['dataset_name'].unique():
        dataset_subset = df[df['dataset_name'] == dataset].copy()
        dataset_subset = dataset_subset.sort_values('cer')
        dataset_subset['rank'] = range(1, len(dataset_subset) + 1)
        
        for _, row in dataset_subset.iterrows():
            rankings.append({
                'dataset': dataset,
                'model': row['model_name'],
                'cer': row['cer'],
                'rank': row['rank']
            })
    
    return pd.DataFrame(rankings)

def present_ranking_table(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present ranking table for presentation"""
    df = load_evaluation_data(csv_path)
    ranking_df = create_ranking_table(df)

    console.print("")
    console.rule("MODEL RANKINGS BY DATASET - PRESENTATION")

    for dataset in ranking_df['dataset'].unique():
        dataset_rankings = ranking_df[ranking_df['dataset'] == dataset].sort_values('rank')

        table = Table(title=f"Dataset: {dataset}")
        table.add_column("Rank", justify="right")
        table.add_column("Model")
        table.add_column("CER", justify="right")
        table.add_column("Performance")

        for _, row in dataset_rankings.iterrows():
            rank = int(row['rank'])
            model = row['model']
            cer = row['cer']

            if cer < 0.1:
                performance_style = "green"
                performance = "EXCELLENT"
            elif cer < 0.2:
                performance_style = "yellow"
                performance = "GOOD"
            elif cer < 0.3:
                performance_style = "cyan"
                performance = "FAIR"
            else:
                performance_style = "red"
                performance = "POOR"

            rank_style = {1: "green", 2: "yellow", 3: "cyan"}.get(rank, "white")
            table.add_row(Text(str(rank), style=rank_style), Text(model, style="bold red"), f"{cer*100:.2f}%", Text(performance, style=performance_style))

        console.print(table)

def find_best_performers(df):
    """Find and present the best performing models"""
    console.print("")
    console.rule("BEST PERFORMING MODELS - PRESENTATION")

    for dataset in df['dataset_name'].unique():
        dataset_subset = df[df['dataset_name'] == dataset]
        best_model = dataset_subset.loc[dataset_subset['cer'].idxmin()]

        mean_cer = best_model['cer']
        if mean_cer < 0.1:
            rating_style = "green"
            rating_text = "EXCELLENT"
        elif mean_cer < 0.2:
            rating_style = "yellow"
            rating_text = "GOOD"
        elif mean_cer < 0.3:
            rating_style = "cyan"
            rating_text = "FAIR"
        else:
            rating_style = "red"
            rating_text = "POOR"

        details = Table(box=None)
        details.add_column("Field", style="bold")
        details.add_column("Value")
        details.add_row("Dataset", dataset)
        details.add_row("Best Model", best_model['model_name'])
        details.add_row("Best CER", f"{best_model['cer']*100:.2f}%")
        details.add_row("Performance Rating", rating_text)

        console.print(Panel(details, title="Best Performer", expand=False, border_style=rating_style))

if __name__ == "__main__":
    present_comparison_table()
    present_detailed_statistics()
    present_ranking_table()
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    find_best_performers(df) 