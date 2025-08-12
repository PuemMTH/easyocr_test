"""
OCR Error Analysis Presentation (No Color Version)
การวิเคราะห์ข้อผิดพลาดของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import ast
from collections import Counter

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

def present_error_analysis(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present error analysis for presentation (no color)"""
    df = load_evaluation_data(csv_path)

    print("="*100)
    print("OCR ERROR ANALYSIS - PRESENTATION")
    print("="*100)

    # Analyze worst cases for each model-dataset combination
    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            print(f"\nModel: {model}")
            print(f"Dataset: {dataset}")

            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                print("\tNo data for this combination.")
                continue

            # Show worst cases
            worst_cases = subset.nlargest(3, 'cer')
            print(f"\nTop 3 Worst Cases:")

            for idx, row in worst_cases.iterrows():
                reference_words = ast.literal_eval(row['reference_words'])
                hypothesis_words = ast.literal_eval(row['hypothesis_words'])
                cer = row['cer']

                print(f"\n" + "="*60)
                print(f"Case #{idx} - CER: {cer*100:.2f}%")
                print("="*60)

                # Print reference and hypothesis
                print(f"Reference: {' '.join(reference_words)}")
                print(f"Hypothesis: {' '.join(hypothesis_words)}")

                # Character-level error analysis
                char_errors = analyze_character_errors(reference_words, hypothesis_words)
                if char_errors:
                    print(f"\nCharacter-level Errors:")
                    for error in char_errors[:10]:  # Show first 10 errors
                        print(f"\tPosition {error['position']}: '{error['reference']}' -> '{error['hypothesis']}'")
                    if len(char_errors) > 10:
                        print(f"\t... and {len(char_errors) - 10} more errors")

                # Word-level error analysis
                word_errors = analyze_word_errors(reference_words, hypothesis_words)
                if word_errors['deletions'] or word_errors['insertions']:
                    print(f"\nWord-level Errors:")
                    if word_errors['deletions']:
                        print(f"\tDeletions: {', '.join(word_errors['deletions'])}")
                    if word_errors['insertions']:
                        print(f"\tInsertions: {', '.join(word_errors['insertions'])}")

def find_common_error_patterns(df, top_n=5):
    """Find common error patterns across the dataset"""
    all_char_errors = []
    all_word_deletions = []
    all_word_insertions = []

    for _, row in df.iterrows():
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

    return {
        'char_errors': char_error_counts.most_common(top_n),
        'deletions': deletion_counts.most_common(top_n),
        'insertions': insertion_counts.most_common(top_n)
    }

def present_error_patterns(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present common error patterns for presentation (no color)"""
    df = load_evaluation_data(csv_path)
    patterns = find_common_error_patterns(df)

    print(f"\n" + "="*100)
    print("COMMON ERROR PATTERNS ANALYSIS")
    print("="*100)

    print(f"\nMost Common Character Errors:")
    for error, count in patterns['char_errors']:
        print(f"\t{error}: {count} times")

    print(f"\nMost Common Word Deletions:")
    for word, count in patterns['deletions']:
        print(f"\t'{word}': {count} times")

    print(f"\nMost Common Word Insertions:")
    for word, count in patterns['insertions']:
        print(f"\t'{word}': {count} times")

def analyze_error_distribution(df):
    """Analyze error distribution across different CER ranges"""
    print(f"\n" + "="*80)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("="*80)

    # Define CER ranges
    ranges = [
        (0, 0.1, "Excellent (0-10%)"),
        (0.1, 0.2, "Good (10-20%)"),
        (0.2, 0.3, "Fair (20-30%)"),
        (0.3, 0.5, "Poor (30-50%)"),
        (0.5, 1.0, "Very Poor (50-100%)")
    ]

    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                continue

            print(f"\nModel: {model}")
            print(f"Dataset: {dataset}")
            print(f"Total Samples: {len(subset)}")

            for min_cer, max_cer, label in ranges:
                count = len(subset[(subset['cer'] >= min_cer) & (subset['cer'] < max_cer)])
                percentage = (count / len(subset)) * 100

                if percentage > 0:
                    print(f"\t{label}: {count} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    present_error_analysis()
    present_error_patterns()
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    analyze_error_distribution(df) 