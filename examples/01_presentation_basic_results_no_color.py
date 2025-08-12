"""
OCR Basic Results Presentation (No Color Version)
การแสดงผลลัพธ์พื้นฐานของ OCR สำหรับการนำเสนอ (เวอร์ชันไม่มีสี)
"""

import pandas as pd
import ast

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def print_example(row, label):
    """Print a single OCR example without color coding"""
    reference_words = ' '.join(ast.literal_eval(row['reference_words']))
    hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
    cer = row['cer']

    print(f"\t{label}:")
    print(f"\t\tREF: {reference_words}")
    print(f"\t\tHYP: {hypothesis_words}")

    if cer == 1.0:
        print(f"\t\tCER: {cer*100:.1f}% (VERY POOR)")
    elif cer < 0.1:
        print(f"\t\tCER: {cer*100:.1f}% (EXCELLENT)")
    elif cer < 0.2:
        print(f"\t\tCER: {cer*100:.1f}% (GOOD)")
    elif cer < 0.3:
        print(f"\t\tCER: {cer*100:.1f}% (FAIR)")
    else:
        print(f"\t\tCER: {cer*100:.1f}% (POOR)")

def present_basic_results(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present basic OCR results for presentation (no color)"""
    df = load_evaluation_data(csv_path)

    print("="*80)
    print("OCR EVALUATION RESULTS - BASIC PRESENTATION")
    print("="*80)

    for model in df['model_name'].unique():
        for dataset in df['dataset_name'].unique():
            print(f"\nModel: {model} vs Dataset: {dataset}")
            subset = df.query("model_name == @model and dataset_name == @dataset")

            if subset.empty:
                print("\tNo data for this combination.")
                continue

            # Find min, max, mean CER
            min_cer = subset['cer'].min()
            max_cer = subset['cer'].max()
            mean_cer = subset['cer'].mean()

            # Get example rows for min, max, mean CER
            min_row = subset.loc[subset['cer'].idxmin()]
            max_row = subset.loc[subset['cer'].idxmax()]
            mean_row = subset.iloc[(subset['cer'] - mean_cer).abs().argsort()[:1]].iloc[0]

            print_example(min_row, "MIN CER")
            print_example(max_row, "MAX CER")
            print_example(mean_row, "MEAN CER")

if __name__ == "__main__":
    present_basic_results() 