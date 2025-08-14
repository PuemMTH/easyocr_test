import pandas as pd
from colorama import Fore, Style
import ast

df = pd.read_csv('reports/ocr_evaluation_20250813_002540/data/ocr_evaluation_detailed.csv')

for model in df['model_name'].unique():
    for dataset in df['dataset_name'].unique():
        print(f"{Fore.RED}{model}{Style.RESET_ALL} vs {Fore.BLUE}{dataset}{Style.RESET_ALL}")
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

        def print_example(row, label):
            reference_words = ' '.join(ast.literal_eval(row['reference_words']))
            hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
            cer = row['cer']
            print(f"\t{label}:")
            print(f"\t\t{Fore.GREEN}REF: {Style.RESET_ALL}{Fore.GREEN}{reference_words}{Style.RESET_ALL}")
            print(f"\t\t{Fore.CYAN}HYP: {Style.RESET_ALL}{Fore.CYAN}{hypothesis_words}{Style.RESET_ALL}")
            if cer == 1.0:
                print(f"\t\t{Fore.RED}CER: {cer*100:.1f}%{Style.RESET_ALL}")
            elif cer < 0.5:
                print(f"\t\t{Fore.GREEN}CER: {cer*100:.1f}%{Style.RESET_ALL}")
            else:
                print(f"\t\tCER: {cer*100:.1f}%")

        print_example(min_row, f"{Fore.YELLOW}MIN CER{Style.RESET_ALL}")
        print_example(max_row, f"{Fore.MAGENTA}MAX CER{Style.RESET_ALL}")
        print_example(mean_row, f"{Fore.BLUE}MEAN CER{Style.RESET_ALL}")