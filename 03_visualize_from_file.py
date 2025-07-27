import pandas as pd
from colorama import Fore, Style
import ast

df = pd.read_csv('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')

print(Fore.GREEN + "Model Name" + Style.RESET_ALL)
# print(df['model_name'].unique())
for model in df['model_name'].unique():
    print('\t' + Fore.RED + model + Style.RESET_ALL)

print(Fore.GREEN + "Dataset Name" + Style.RESET_ALL)
# print(df['dataset_name'].unique())
for dataset in df['dataset_name'].unique():
    print('\t' + Fore.RED + dataset + Style.RESET_ALL)

for model in df['model_name'].unique():
    for dataset in df['dataset_name'].unique():
        print(f"{Fore.RED}{model}{Style.RESET_ALL} vs {Fore.BLUE}{dataset}{Style.RESET_ALL}")
        subset = df.query("model_name == @model and dataset_name == @dataset").sample(2)
        # print(f"\t{Fore.YELLOW}First 10 reference_words vs hypothesis_words:{Style.RESET_ALL}")
        for _, row in subset.iterrows():
            reference_words = ' '.join(ast.literal_eval(row['reference_words']))
            hypothesis_words = ' '.join(ast.literal_eval(row['hypothesis_words']))
            cer = row['cer']
            print(f"\t{Fore.GREEN}REF: {Style.RESET_ALL}{Fore.GREEN}{reference_words}{Style.RESET_ALL}")
            print(f"\t{Fore.CYAN}HYP: {Style.RESET_ALL}{Fore.CYAN}{hypothesis_words}{Style.RESET_ALL}")
            if cer == 1.0:
                print(f"\t{Fore.RED}CER: {cer*100:.1f}%{Style.RESET_ALL}")
            elif cer < 0.5:
                print(f"\t{Fore.GREEN}CER: {cer*100:.1f}%{Style.RESET_ALL}")