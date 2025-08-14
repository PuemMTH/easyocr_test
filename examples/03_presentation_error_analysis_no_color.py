"""
OCR Error Analysis Presentation (No Color Version)
การวิเคราะห์ข้อผิดพลาดของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import ast
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console(no_color=False)

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def analyze_character_errors(reference_words, hypothesis_words):
    """Analyze character-level errors between reference and hypothesis"""
    ref_chars = list(''.join(reference_words))
    hyp_chars = list(''.join(hypothesis_words))

    errors = []
    min_len = min(len(ref_chars), len(hyp_chars))

    for i in range(min_len):
        if ref_chars[i] != hyp_chars[i]:
            errors.append({
                'position': i,
                'reference': ref_chars[i],
                'hypothesis': hyp_chars[i]
            })

    return errors

def analyze_word_errors(reference_words, hypothesis_words):
    """Analyze word-level errors"""
    ref_set = set(reference_words)
    hyp_set = set(hypothesis_words)

    deletions = ref_set - hyp_set
    insertions = hyp_set - ref_set

    return {
        'deletions': list(deletions),
        'insertions': list(insertions)
    }

def present_error_analysis(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv', export_csv=None):
    """Present error analysis in a single colored table with CSV export"""
    df = load_evaluation_data(csv_path)

    console.rule("OCR ERROR ANALYSIS - WORST CASES")

    # Create a single table with all worst cases
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="bright_cyan")
    table.add_column("Rank", justify="center", style="yellow")
    table.add_column("CER %", justify="right", style="white")
    table.add_column("Reference Text", style="dim white", max_width=40)
    table.add_column("Hypothesis Text", style="dim white", max_width=40)

    # For CSV export
    export_data = []

    # Analyze worst cases for each model-dataset combination
    for model in sorted(df['model_name'].unique()):
        for dataset in sorted(df['dataset_name'].unique()):
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            # Show worst cases
            worst_cases = subset.nlargest(3, 'cer')

            for rank, (idx, row) in enumerate(worst_cases.iterrows(), 1):
                reference_words = ' '.join(ast.literal_eval(row['reference_words']))
                hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
                cer = row['cer']

                # Truncate long text for table display
                ref_display = reference_words[:37] + "..." if len(reference_words) > 40 else reference_words
                hyp_display = hypothesis_words[:37] + "..." if len(hypothesis_words) > 40 else hypothesis_words

                # Color CER based on severity
                if cer >= 0.8:
                    cer_color = "bright_red"
                elif cer >= 0.5:
                    cer_color = "red"
                elif cer >= 0.3:
                    cer_color = "yellow"
                else:
                    cer_color = "green"

                # Color rank
                rank_color = "bright_red" if rank == 1 else "red" if rank == 2 else "yellow"

                table.add_row(
                    model,
                    dataset,
                    f"[{rank_color}]#{rank}[/{rank_color}]",
                    f"[{cer_color}]{cer*100:.1f}[/{cer_color}]",
                    ref_display,
                    hyp_display
                )

                # Collect data for CSV export
                if export_csv:
                    export_data.append({
                        'model': model,
                        'dataset': dataset,
                        'rank': rank,
                        'cer_percent': f"{cer*100:.1f}",
                        'reference_text': reference_words,
                        'hypothesis_text': hypothesis_words,
                        'source_file': row['source_file'],
                        'box_index': row['box_index']
                    })

    console.print(table)

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Worst cases exported to: {export_csv}")
        console.print(f"🎯 Total examples exported: {len(export_data)}")

def find_common_error_patterns(df, top_n=5):
    """Find common error patterns for each model-dataset combination"""
    results = {}
    
    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            subset = df.query("model_name == @model and dataset_name == @dataset")
            
            if subset.empty:
                continue
                
            all_char_errors = []
            all_word_deletions = []
            all_word_insertions = []

            for _, row in subset.iterrows():
                reference_words = ast.literal_eval(row['reference_words'])
                hypothesis_words = ast.literal_eval(row['hypothesis_words'])

                # Collect character errors
                char_errors = analyze_character_errors(reference_words, hypothesis_words)
                for error in char_errors:
                    all_char_errors.append(f"{error['reference']}->{error['hypothesis']}")

                # Collect word errors
                word_errors = analyze_word_errors(reference_words, hypothesis_words)
                all_word_deletions.extend(word_errors['deletions'])
                all_word_insertions.extend(word_errors['insertions'])

            # Count patterns
            char_error_counts = Counter(all_char_errors)
            deletion_counts = Counter(all_word_deletions)
            insertion_counts = Counter(all_word_insertions)

            results[f"{model}_{dataset}"] = {
                'model': model,
                'dataset': dataset,
                'char_errors': char_error_counts.most_common(top_n),
                'deletions': deletion_counts.most_common(top_n),
                'insertions': insertion_counts.most_common(top_n)
            }
    
    return results

def present_error_patterns(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv', export_csv=None):
    """Present common error patterns by model-dataset in a single colored table with CSV export"""
    df = load_evaluation_data(csv_path)
    patterns_by_combo = find_common_error_patterns(df)

    console.rule("COMMON ERROR PATTERNS BY MODEL-DATASET")

    # Create a single consolidated table with all error patterns
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="bright_cyan")
    table.add_column("Error Type", style="yellow", no_wrap=True)
    table.add_column("Pattern/Word", style="white")
    table.add_column("Count", justify="right", style="white")
    table.add_column("Frequency", justify="right", style="white")

    # For CSV export
    export_data = []

    for combo_key, patterns in patterns_by_combo.items():
        model = patterns['model']
        dataset = patterns['dataset']
        
        # Calculate totals for frequency percentages
        total_char_errors = sum(count for _, count in patterns['char_errors'])
        total_deletions = sum(count for _, count in patterns['deletions'])
        total_insertions = sum(count for _, count in patterns['insertions'])

        # Add character errors
        for pattern, count in patterns['char_errors']:
            frequency = (count / total_char_errors * 100) if total_char_errors > 0 else 0
            
            # Color based on frequency
            if frequency >= 30:
                freq_color = "bright_red"
            elif frequency >= 20:
                freq_color = "red"
            elif frequency >= 10:
                freq_color = "yellow"
            else:
                freq_color = "green"

            table.add_row(
                model,
                dataset,
                "[red]Character Error[/red]",
                f"[white]{pattern}[/white]",
                f"{count:,}",
                f"[{freq_color}]{frequency:.1f}%[/{freq_color}]"
            )

            if export_csv:
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'error_type': 'Character Error',
                    'pattern_word': pattern,
                    'count': count,
                    'frequency_percent': f"{frequency:.1f}",
                    'total_in_category': total_char_errors
                })

        # Add word deletions
        for word, count in patterns['deletions']:
            frequency = (count / total_deletions * 100) if total_deletions > 0 else 0
            
            # Color based on frequency
            if frequency >= 30:
                freq_color = "bright_red"
            elif frequency >= 20:
                freq_color = "red"
            elif frequency >= 10:
                freq_color = "yellow"
            else:
                freq_color = "green"

            table.add_row(
                model,
                dataset,
                "[yellow]Word Deletion[/yellow]",
                f"[white]{word}[/white]",
                f"{count:,}",
                f"[{freq_color}]{frequency:.1f}%[/{freq_color}]"
            )

            if export_csv:
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'error_type': 'Word Deletion',
                    'pattern_word': word,
                    'count': count,
                    'frequency_percent': f"{frequency:.1f}",
                    'total_in_category': total_deletions
                })

        # Add word insertions
        for word, count in patterns['insertions']:
            frequency = (count / total_insertions * 100) if total_insertions > 0 else 0
            
            # Color based on frequency
            if frequency >= 30:
                freq_color = "bright_red"
            elif frequency >= 20:
                freq_color = "red"
            elif frequency >= 10:
                freq_color = "yellow"
            else:
                freq_color = "green"

            table.add_row(
                model,
                dataset,
                "[blue]Word Insertion[/blue]",
                f"[white]{word}[/white]",
                f"{count:,}",
                f"[{freq_color}]{frequency:.1f}%[/{freq_color}]"
            )

            if export_csv:
                export_data.append({
                    'model': model,
                    'dataset': dataset,
                    'error_type': 'Word Insertion',
                    'pattern_word': word,
                    'count': count,
                    'frequency_percent': f"{frequency:.1f}",
                    'total_in_category': total_insertions
                })

    console.print(table)

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Error patterns exported to: {export_csv}")
        console.print(f"🎯 Total patterns exported: {len(export_data)}")

def get_range_color(cer_range):
    """Get color style based on CER range"""
    if "Excellent" in cer_range:
        return "bright_green"
    elif "Good" in cer_range:
        return "green"
    elif "Fair" in cer_range:
        return "yellow"
    elif "Poor" in cer_range and "Very" not in cer_range:
        return "red"
    else:  # Very Poor
        return "bright_red"

def get_percentage_color(percentage):
    """Get color style based on percentage"""
    if percentage >= 50:
        return "bright_red"
    elif percentage >= 30:
        return "red"
    elif percentage >= 15:
        return "yellow"
    elif percentage >= 5:
        return "green"
    else:
        return "bright_green"

def analyze_error_distribution(df, export_csv=None):
    """Analyze error distribution across different CER ranges in a single colored table"""
    console.rule("ERROR DISTRIBUTION ANALYSIS")

    # Define CER ranges
    ranges = [
        (0, 0.1, "Excellent (0-10%)"),
        (0.1, 0.2, "Good (10-20%)"),
        (0.2, 0.3, "Fair (20-30%)"),
        (0.3, 0.5, "Poor (30-50%)"),
        (0.5, 1.0, "Very Poor (50-100%)")
    ]

    # Create a single table with all results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="bright_cyan")
    table.add_column("CER Range", style="white")
    table.add_column("Count", justify="right", style="white")
    table.add_column("Percentage", justify="right", style="white")
    table.add_column("Total Samples", justify="right", style="dim white")

    # For CSV export
    export_data = []

    for model in sorted(df['model_name'].unique()):
        for dataset in sorted(df['dataset_name'].unique()):
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            total_samples = len(subset)

            for min_cer, max_cer, label in ranges:
                range_subset = subset[(subset['cer'] >= min_cer) & (subset['cer'] < max_cer)]
                count = len(range_subset)
                percentage = (count / total_samples) * 100

                if count > 0:  # Only show ranges that have data
                    # Get colors for dynamic styling
                    range_color = get_range_color(label)
                    percentage_color = get_percentage_color(percentage)

                    table.add_row(
                        model,
                        dataset,
                        f"[{range_color}]{label}[/{range_color}]",
                        f"{count:,}",
                        f"[{percentage_color}]{percentage:.1f}%[/{percentage_color}]",
                        f"{total_samples:,}"
                    )

                # Collect data for CSV export
                if export_csv and count > 0:
                    # Add sample examples from this range
                    for _, row in range_subset.head(5).iterrows():  # Top 5 examples per range
                        reference_words = ' '.join(ast.literal_eval(row['reference_words']))
                        hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
                        
                        export_data.append({
                            'model': model,
                            'dataset': dataset,
                            'cer_range': label,
                            'cer_min': min_cer,
                            'cer_max': max_cer,
                            'cer_percent': f"{row['cer']*100:.1f}",
                            'range_count': count,
                            'range_percentage': f"{percentage:.1f}",
                            'reference_text': reference_words,
                            'hypothesis_text': hypothesis_words,
                            'source_file': row['source_file'],
                            'box_index': row['box_index']
                        })

    console.print(table)

    # Export to CSV if requested
    if export_csv and export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
        console.print(f"\n📄 Error distribution exported to: {export_csv}")
        console.print(f"🎯 Total examples exported: {len(export_data)}")

if __name__ == "__main__":
    present_error_analysis()
    present_error_patterns()
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    analyze_error_distribution(df) 