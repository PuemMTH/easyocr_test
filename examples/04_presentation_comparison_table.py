"""
OCR Comparison Table Presentation
การแสดงตารางเปรียบเทียบ OCR สำหรับการนำเสนอ
"""

import pandas as pd
from colorama import Fore, Style
import numpy as np

def load_evaluation_data(csv_path):
    """Load OCR evaluation data from CSV file"""
    return pd.read_csv(csv_path)

def create_comparison_table(df):
    """Create a comprehensive comparison table for all model-dataset combinations"""
    models = df['model_name'].unique()
    datasets = df['dataset_name'].unique()
    
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
    
    print(f"{Fore.CYAN}{'='*120}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}OCR MODEL COMPARISON TABLE - PRESENTATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*120}{Style.RESET_ALL}")
    
    # Print comparison matrix
    print(f"\n{Fore.YELLOW}Mean CER Comparison Matrix (%):{Style.RESET_ALL}")
    print("-" * 120)
    
    # Print header
    header = f"{'Model':<40}"
    for dataset in comparison_df.columns:
        header += f"{dataset:<20}"
    print(header)
    print("-" * 120)
    
    # Print each model row
    for model in comparison_df.index:
        row = f"{Fore.RED}{model:<40}{Style.RESET_ALL}"
        for dataset in comparison_df.columns:
            cer = comparison_df.loc[model, dataset]
            if pd.isna(cer):
                row += f"{'N/A':<20}"
            else:
                if cer < 0.1:
                    color = Fore.GREEN
                elif cer < 0.2:
                    color = Fore.YELLOW
                elif cer < 0.3:
                    color = Fore.CYAN
                else:
                    color = Fore.RED
                row += f"{color}{cer*100:.2f}%{Style.RESET_ALL:<20}"
        print(row)

def present_detailed_statistics(csv_path='reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv'):
    """Present detailed statistics for presentation"""
    df = load_evaluation_data(csv_path)
    _, stats_df = create_comparison_table(df)
    
    print(f"\n{Fore.CYAN}{'='*140}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}DETAILED STATISTICS - PRESENTATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*140}{Style.RESET_ALL}")
    
    for (model, dataset), stats in stats_df.iterrows():
        print(f"\n{Fore.RED}Model: {model}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Dataset: {dataset}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Sample Count: {stats[('cer', 'count')]}{Style.RESET_ALL}")
        
        mean_cer = stats[('cer', 'mean')]
        std_cer = stats[('cer', 'std')]
        min_cer = stats[('cer', 'min')]
        max_cer = stats[('cer', 'max')]
        
        print(f"{Fore.GREEN}Mean CER: {mean_cer*100:.2f}%{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Std CER: {std_cer*100:.2f}%{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Min CER: {min_cer*100:.2f}%{Style.RESET_ALL}")
        print(f"{Fore.RED}Max CER: {max_cer*100:.2f}%{Style.RESET_ALL}")
        
        # Performance rating
        if mean_cer < 0.1:
            rating = f"{Fore.GREEN}EXCELLENT{Style.RESET_ALL}"
        elif mean_cer < 0.2:
            rating = f"{Fore.YELLOW}GOOD{Style.RESET_ALL}"
        elif mean_cer < 0.3:
            rating = f"{Fore.CYAN}FAIR{Style.RESET_ALL}"
        else:
            rating = f"{Fore.RED}POOR{Style.RESET_ALL}"
        
        print(f"{Fore.MAGENTA}Performance Rating: {rating}{Style.RESET_ALL}")

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
    
    print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}MODEL RANKINGS BY DATASET - PRESENTATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    
    for dataset in ranking_df['dataset'].unique():
        dataset_rankings = ranking_df[ranking_df['dataset'] == dataset].sort_values('rank')
        
        print(f"\n{Fore.BLUE}Dataset: {dataset}{Style.RESET_ALL}")
        print(f"{'Rank':<8} {'Model':<50} {'CER':<12} {'Performance':<15}")
        print("-" * 85)
        
        for _, row in dataset_rankings.iterrows():
            rank = row['rank']
            model = row['model']
            cer = row['cer']
            
            # Performance rating
            if cer < 0.1:
                performance = f"{Fore.GREEN}EXCELLENT{Style.RESET_ALL}"
            elif cer < 0.2:
                performance = f"{Fore.YELLOW}GOOD{Style.RESET_ALL}"
            elif cer < 0.3:
                performance = f"{Fore.CYAN}FAIR{Style.RESET_ALL}"
            else:
                performance = f"{Fore.RED}POOR{Style.RESET_ALL}"
            
            if rank == 1:
                rank_color = Fore.GREEN
            elif rank == 2:
                rank_color = Fore.YELLOW
            elif rank == 3:
                rank_color = Fore.CYAN
            else:
                rank_color = Fore.WHITE
            
            print(f"{rank_color}{rank:<8}{Style.RESET_ALL} {Fore.RED}{model:<50}{Style.RESET_ALL} {cer*100:.2f}%{'':<8} {performance}")

def find_best_performers(df):
    """Find and present the best performing models"""
    print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}BEST PERFORMING MODELS - PRESENTATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    
    for dataset in df['dataset_name'].unique():
        dataset_subset = df[df['dataset_name'] == dataset]
        best_model = dataset_subset.loc[dataset_subset['cer'].idxmin()]
        
        print(f"\n{Fore.BLUE}Dataset: {dataset}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Best Model: {best_model['model_name']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Best CER: {best_model['cer']*100:.2f}%{Style.RESET_ALL}")
        
        # Performance rating
        cer = best_model['cer']
        if cer < 0.1:
            rating = f"{Fore.GREEN}EXCELLENT{Style.RESET_ALL}"
        elif cer < 0.2:
            rating = f"{Fore.YELLOW}GOOD{Style.RESET_ALL}"
        elif cer < 0.3:
            rating = f"{Fore.CYAN}FAIR{Style.RESET_ALL}"
        else:
            rating = f"{Fore.RED}POOR{Style.RESET_ALL}"
        
        print(f"{Fore.MAGENTA}Performance Rating: {rating}{Style.RESET_ALL}")

if __name__ == "__main__":
    present_comparison_table()
    present_detailed_statistics()
    present_ranking_table()
    df = load_evaluation_data('reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv')
    find_best_performers(df) 