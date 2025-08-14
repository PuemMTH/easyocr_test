"""
OCR Basic Results Presentation (No Color Version)
การแสดงผลลัพธ์พื้นฐานของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import ast
from rich.console import Console
from rich.table import Table
import os
import glob

# Use Rich for structured, readable output with colors
console = Console(no_color=False)

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def resolve_csv_path(csv_path):
    """Return csv_path if exists; otherwise find latest reports/*/data/ocr_evaluation_detailed.csv"""
    if os.path.exists(csv_path):
        return csv_path
    candidates = sorted(glob.glob("reports/*/data/ocr_evaluation_detailed.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"CSV not found. Given: {csv_path} and no candidates in reports/*/data/")

def get_rating(cer):
    """Get rating based on CER value"""
    if cer >= 1.0:  # CER >= 100%
        return "VERY POOR"
    elif cer >= 0.5:  # CER >= 50%
        return "POOR"
    elif cer >= 0.3:  # CER >= 30%
        return "FAIR"
    elif cer >= 0.2:  # CER >= 20%
        return "GOOD"
    elif cer >= 0.1:  # CER >= 10%
        return "EXCELLENT"
    else:  # CER < 10%
        return "PERFECT"

def get_rating_color(rating):
    """Get color style based on rating"""
    color_map = {
        "PERFECT": "bright_white",
        "EXCELLENT": "bright_green",
        "GOOD": "green", 
        "FAIR": "yellow",
        "POOR": "red",
        "VERY POOR": "bright_red"
    }
    return color_map.get(rating, "white")

def get_type_color(example_type):
    """Get color style based on example type"""
    type_colors = {
        "MIN": "bright_green",
        "MEAN": "blue", 
        "MAX": "bright_red"
    }
    return type_colors.get(example_type, "white")

def present_basic_results(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv', 
                         min_examples=1, max_examples=1, mean_examples=1, export_csv=None):
    """Present basic OCR results in a single table for presentation (with colors)
    
    Args:
        csv_path: Path to the CSV file with evaluation results
        min_examples: Number of MIN CER examples to show per model-dataset combination
        max_examples: Number of MAX CER examples to show per model-dataset combination  
        mean_examples: Number of MEAN CER examples to show per model-dataset combination
        export_csv: Path to export CSV file with results (optional)
    """
    csv_path = resolve_csv_path(csv_path)
    df = load_evaluation_data(csv_path)

    console.rule("OCR EVALUATION RESULTS - BASIC PRESENTATION")

    # Create a single table with all results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="bright_cyan")
    table.add_column("Type", style="white")
    table.add_column("CER %", justify="right", style="white")
    table.add_column("Rating", style="white")
    table.add_column("Reference Text", style="dim white", max_width=50)
    table.add_column("Hypothesis Text", style="dim white", max_width=50)

    # For CSV export
    export_data = []

    for model in sorted(df['model_name'].unique()):
        for dataset in sorted(df['dataset_name'].unique()):
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            # Find mean CER and example rows
            mean_cer = subset['cer'].mean()

            # Get example rows for min, max, mean CER
            min_rows = subset.nsmallest(min_examples, 'cer')
            max_rows = subset.nlargest(max_examples, 'cer')
            
            # For mean examples, find rows closest to the mean CER
            mean_indices = (subset['cer'] - mean_cer).abs().argsort()[:mean_examples]
            mean_rows = subset.iloc[mean_indices]

            # Add rows for min examples
            for idx, (_, row) in enumerate(min_rows.iterrows(), 1):
                reference_words = ' '.join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
                cer = row['cer']
                rating = get_rating(cer)
                source_file = row['source_file']

                # Truncate long text for table display
                ref_display = reference_words[:47] + "..." if len(reference_words) > 50 else reference_words
                hyp_display = hypothesis_words[:47] + "..." if len(hypothesis_words) > 50 else hypothesis_words

                # Get colors for dynamic styling
                type_color = get_type_color("MIN")
                rating_color = get_rating_color(rating)
                
                # Color CER based on value
                if cer == 0.0:
                    cer_color = "bright_green"
                elif cer < 0.1:
                    cer_color = "green"
                elif cer < 0.3:
                    cer_color = "yellow"
                else:
                    cer_color = "red"

                example_type = f"MIN-{idx}" if min_examples > 1 else "MIN"
                table.add_row(
                    model,
                    dataset,
                    f"[{type_color}]{example_type}[/{type_color}]",
                    f"[{cer_color}]{cer*100:.1f}[/{cer_color}]",
                    f"[{rating_color}]{rating}[/{rating_color}]",
                    ref_display,
                    hyp_display
                )

                # Collect data for CSV export
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'type': example_type,
                    'cer_percent': f"{cer*100:.1f}",
                    'rating': rating,
                    'reference_text': reference_words,
                    'hypothesis_text': hypothesis_words,
                    'source_file': source_file,
                    'box_index': row['box_index']
                })

            # Add rows for max examples
            for idx, (_, row) in enumerate(max_rows.iterrows(), 1):
                reference_words = ' '.join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
                cer = row['cer']
                rating = get_rating(cer)
                source_file = row['source_file']

                # Truncate long text for table display
                ref_display = reference_words[:47] + "..." if len(reference_words) > 50 else reference_words
                hyp_display = hypothesis_words[:47] + "..." if len(hypothesis_words) > 50 else hypothesis_words

                # Get colors for dynamic styling
                type_color = get_type_color("MAX")
                rating_color = get_rating_color(rating)
                
                # Color CER based on value
                if cer == 0.0:
                    cer_color = "bright_green"
                elif cer < 0.1:
                    cer_color = "green"
                elif cer < 0.3:
                    cer_color = "yellow"
                else:
                    cer_color = "red"

                example_type = f"MAX-{idx}" if max_examples > 1 else "MAX"
                table.add_row(
                    model,
                    dataset,
                    f"[{type_color}]{example_type}[/{type_color}]",
                    f"[{cer_color}]{cer*100:.1f}[/{cer_color}]",
                    f"[{rating_color}]{rating}[/{rating_color}]",
                    ref_display,
                    hyp_display
                )

                # Collect data for CSV export
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'type': example_type,
                    'cer_percent': f"{cer*100:.1f}",
                    'rating': rating,
                    'reference_text': reference_words,
                    'hypothesis_text': hypothesis_words,
                    'source_file': source_file,
                    'box_index': row['box_index']
                })

            # Add rows for mean examples
            for idx, (_, row) in enumerate(mean_rows.iterrows(), 1):
                reference_words = ' '.join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
                cer = row['cer']
                rating = get_rating(cer)
                source_file = row['source_file']

                # Truncate long text for table display
                ref_display = reference_words[:47] + "..." if len(reference_words) > 50 else reference_words
                hyp_display = hypothesis_words[:47] + "..." if len(hypothesis_words) > 50 else hypothesis_words

                # Get colors for dynamic styling
                type_color = get_type_color("MEAN")
                rating_color = get_rating_color(rating)
                
                # Color CER based on value
                if cer == 0.0:
                    cer_color = "bright_green"
                elif cer < 0.1:
                    cer_color = "green"
                elif cer < 0.3:
                    cer_color = "yellow"
                else:
                    cer_color = "red"

                example_type = f"MEAN-{idx}" if mean_examples > 1 else "MEAN"
                table.add_row(
                    model,
                    dataset,
                    f"[{type_color}]{example_type}[/{type_color}]",
                    f"[{cer_color}]{cer*100:.1f}[/{cer_color}]",
                    f"[{rating_color}]{rating}[/{rating_color}]",
                    ref_display,
                    hyp_display
                )

                # Collect data for CSV export
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'type': example_type,
                    'cer_percent': f"{cer*100:.1f}",
                    'rating': rating,
                    'reference_text': reference_words,
                    'hypothesis_text': hypothesis_words,
                    'source_file': source_file,
                    'box_index': row['box_index']
                })

    console.print(table)

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Results exported to: {export_csv}")
        console.print(f"🎯 Total examples exported: {len(export_data)}")

def present_basic_results_with_names(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv', min_examples=2, max_examples=2, mean_examples=2, export_csv=None):
    """Present basic OCR results with source file names displayed in table"""
    csv_path = resolve_csv_path(csv_path)
    df = load_evaluation_data(csv_path)

    # Title
    console.rule("OCR BASIC RESULTS WITH FILENAMES - PRESENTATION")

    # Create Rich table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("CER %", justify="right")
    table.add_column("Rating")
    table.add_column("Source File", style="dim")
    table.add_column("Reference Text", max_width=30)
    table.add_column("Hypothesis Text", max_width=30)

    export_data = []
    
    for model_name in df['model_name'].unique():
        for dataset_name in df['dataset_name'].unique():
            subset = df.query("model_name == @model_name and dataset_name == @dataset_name")
            
            if subset.empty:
                continue

            # Get examples for each type
            examples = []
            
            # MIN examples
            min_subset = subset.nsmallest(min_examples, 'cer')
            for _, row in min_subset.iterrows():
                examples.append((row, "MIN"))
            
            # MAX examples  
            max_subset = subset.nlargest(max_examples, 'cer')
            for _, row in max_subset.iterrows():
                examples.append((row, "MAX"))
            
            # MEAN examples (closest to mean)
            mean_cer = subset['cer'].mean()
            mean_subset = subset.iloc[(subset['cer'] - mean_cer).abs().argsort()[:mean_examples]]
            for _, row in mean_subset.iterrows():
                examples.append((row, "MEAN"))

            # Add rows to table
            for row, example_type in examples:
                cer = row['cer']
                rating = get_rating(cer)
                
                # Parse reference and hypothesis
                reference_words = ", ".join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ", ".join(ast.literal_eval(row['hypothesis_words']))
                
                # Extract filename from source_file path
                source_file = os.path.basename(row['source_file'])
                
                table.add_row(
                    str(row['model_name']),
                    str(row['dataset_name']),
                    f"[{get_type_color(example_type)}]{example_type}[/]",
                    f"[{get_rating_color(rating)}]{cer*100:.1f}[/]",
                    f"[{get_rating_color(rating)}]{rating}[/]",
                    source_file,
                    reference_words,
                    hypothesis_words
                )

                export_data.append({
                    'model_name': row['model_name'],
                    'dataset_name': row['dataset_name'],
                    'type': example_type,
                    'cer_percent': f"{cer*100:.1f}",
                    'rating': rating,
                    'reference_text': reference_words,
                    'hypothesis_text': hypothesis_words,
                    'source_file': source_file,
                    'box_index': row['box_index']
                })

    console.print(table)

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Results exported to: {export_csv}")
        console.print(f"🎯 Total examples exported: {len(export_data)}")

def present_results_by_file(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv', min_examples=2, max_examples=2, mean_examples=2, export_csv=None):
    """Present OCR results grouped by file and separated by dataset, showing how each model processes the same files"""
    csv_path = resolve_csv_path(csv_path)
    df = load_evaluation_data(csv_path)

    # Title
    console.rule("OCR RESULTS BY FILE - SEPARATED BY DATASET")

    # Find files that appear in multiple model-dataset combinations
    file_counts = df.groupby('source_file').agg({
        'model_name': 'nunique',
        'dataset_name': 'nunique'
    }).reset_index()
    
    # Get files with multiple model-dataset combinations
    multi_model_files = file_counts[
        (file_counts['model_name'] > 1) | (file_counts['dataset_name'] > 1)
    ]['source_file'].tolist()

    if not multi_model_files:
        console.print("[red]No files found that appear across multiple model-dataset combinations[/red]")
        return

    # Select example files based on CER patterns
    example_files = []
    
    # Get examples for different CER scenarios
    for file_path in multi_model_files[:min_examples + max_examples + mean_examples]:
        file_data = df[df['source_file'] == file_path]
        if len(file_data) > 1:  # Has multiple model-dataset results
            example_files.append(file_path)
    
    # Limit to requested number of examples
    total_examples = min_examples + max_examples + mean_examples
    example_files = example_files[:total_examples]

    export_data = []
    
    # Group by dataset and create separate tables
    for dataset_name in sorted(df['dataset_name'].unique()):
        dataset_files = []
        
        # Check which files have data for this dataset
        for file_path in example_files:
            file_data = df[(df['source_file'] == file_path) & (df['dataset_name'] == dataset_name)]
            if not file_data.empty:
                dataset_files.append(file_path)
        
        if not dataset_files:
            continue
            
        console.print(f"\n[bold magenta]📊 Dataset: {dataset_name}[/bold magenta]")
        
        # Create table for this dataset
        table = Table(show_header=True, header_style="bold", title=f"Results for Dataset: {dataset_name}")
        table.add_column("Sample", style="bright_blue")
        table.add_column("Source File", style="dim", max_width=25)
        table.add_column("Model", style="cyan")
        table.add_column("CER %", justify="right")
        table.add_column("Rating")
        table.add_column("Reference Text", max_width=30)
        table.add_column("Hypothesis Text", max_width=30)

        sample_number = 1

        for file_path in dataset_files:
            file_data = df[(df['source_file'] == file_path) & (df['dataset_name'] == dataset_name)].sort_values(['model_name'])
            filename = os.path.basename(file_path)
            
            if file_data.empty:
                continue
                
            # Calculate stats for this file in this dataset
            best_result = file_data.loc[file_data['cer'].idxmin()]
            worst_result = file_data.loc[file_data['cer'].idxmax()]
            cer_range = worst_result['cer'] - best_result['cer'] if len(file_data) > 1 else 0
            
            first_row = True
            for _, row in file_data.iterrows():
                cer = row['cer']
                rating = get_rating(cer)
                
                # Parse reference and hypothesis
                reference_words = ", ".join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ", ".join(ast.literal_eval(row['hypothesis_words']))
                
                # Show sample number only for first row of each file
                sample_display = f"#{sample_number}" if first_row else ""
                filename_display = filename if first_row else ""
                
                table.add_row(
                    sample_display,
                    filename_display,
                    str(row['model_name']),
                    f"[{get_rating_color(rating)}]{cer*100:.1f}[/]",
                    f"[{get_rating_color(rating)}]{rating}[/]",
                    reference_words,
                    hypothesis_words
                )

                export_data.append({
                    'dataset_name': dataset_name,
                    'sample_number': sample_number,
                    'source_file': filename,
                    'model_name': row['model_name'],
                    'cer_percent': f"{cer*100:.1f}",
                    'rating': rating,
                    'reference_text': reference_words,
                    'hypothesis_text': hypothesis_words,
                    'box_index': row['box_index'],
                    'best_cer': f"{best_result['cer']*100:.1f}",
                    'worst_cer': f"{worst_result['cer']*100:.1f}",
                    'cer_range': f"{cer_range*100:.1f}"
                })
                
                first_row = False
                
            # Add a separator row (empty row) between samples
            if sample_number < len(dataset_files):
                table.add_row("", "", "", "", "", "", "")
            
            sample_number += 1

        console.print(table)
        
        # Print summary statistics for this dataset
        console.print(f"\n📊 [bold]Summary for {dataset_name}:[/bold]")
        for i, file_path in enumerate(dataset_files, 1):
            file_data = df[(df['source_file'] == file_path) & (df['dataset_name'] == dataset_name)]
            if file_data.empty:
                continue
                
            best_result = file_data.loc[file_data['cer'].idxmin()]
            worst_result = file_data.loc[file_data['cer'].idxmax()]
            cer_range = worst_result['cer'] - best_result['cer'] if len(file_data) > 1 else 0
            
            console.print(f"  #{i} [cyan]{os.path.basename(file_path)}[/cyan]:")
            console.print(f"    🏆 [green]Best:[/green] {best_result['model_name']} = {best_result['cer']*100:.1f}% CER")
            if len(file_data) > 1:
                console.print(f"    💔 [red]Worst:[/red] {worst_result['model_name']} = {worst_result['cer']*100:.1f}% CER")
                console.print(f"    📊 [yellow]Range:[/yellow] {cer_range*100:.1f}% difference")

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Results exported to: {export_csv}")
        console.print(f"🎯 Total file comparisons exported: {len(export_data)}")

if __name__ == "__main__":
    present_basic_results(min_examples=1, max_examples=1, mean_examples=1) 