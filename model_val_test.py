import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_loss_plot(df, save_dir='plots'):
    """Create and save loss comparison plot"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['epoch'], df['train_loss'], 
             color='#e74c3c', linewidth=3, 
             marker='o', markersize=8, label='Training Loss', alpha=0.8)
    plt.plot(df['epoch'], df['valid_loss'], 
             color='#3498db', linewidth=3, 
             marker='s', markersize=8, label='Validation Loss', alpha=0.8)
    
    plt.title('📈 Training vs Validation Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = Path(save_dir) / 'loss_plot.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Loss plot saved: {loss_path}")
    return loss_path

def create_accuracy_plot(df, save_dir='plots'):
    """Create and save accuracy plot"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['epoch'], df['accuracy'], 
             color='#27ae60', linewidth=3, 
             marker='^', markersize=8, label='Accuracy (%)', alpha=0.8)
    
    plt.title('🎯 Model Accuracy Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    acc_path = Path(save_dir) / 'accuracy_plot.png'
    plt.savefig(acc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Accuracy plot saved: {acc_path}")
    return acc_path

def create_norm_ed_plot(df, save_dir='plots'):
    """Create and save Norm ED plot"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['epoch'], df['norm_ED'], 
             color='#f39c12', linewidth=3, 
             marker='D', markersize=8, label='Norm ED', alpha=0.8)
    
    plt.title('📊 Norm ED Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Norm ED', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    norm_path = Path(save_dir) / 'norm_ed_plot.png'
    plt.savefig(norm_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Norm ED plot saved: {norm_path}")
    return norm_path

def create_combined_metrics_plot(df, save_dir='plots'):
    """Create accuracy and norm ED combined plot"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Accuracy on left axis
    color1 = '#27ae60'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', color=color1, fontsize=14)
    line1 = ax1.plot(df['epoch'], df['accuracy'], 
                     color=color1, linewidth=3, 
                     marker='^', markersize=8, label='Accuracy (%)', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Norm ED on right axis
    ax2 = ax1.twinx()
    color2 = '#f39c12'
    ax2.set_ylabel('Norm ED', color=color2, fontsize=14)
    line2 = ax2.plot(df['epoch'], df['norm_ED'], 
                     color=color2, linewidth=3, 
                     marker='D', markersize=8, label='Norm ED', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    plt.title('🎯 Accuracy & Norm ED Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    combined_path = Path(save_dir) / 'accuracy_norm_ed_plot.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Combined metrics plot saved: {combined_path}")
    return combined_path

def create_overview_plot(df, save_dir='plots'):
    """Create overview plot with all metrics"""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Loss on left axis
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14, color='#2c3e50')
    ax1.plot(df['epoch'], df['train_loss'], 
             color='#e74c3c', linewidth=2.5, 
             marker='o', markersize=6, label='Training Loss', alpha=0.9)
    ax1.plot(df['epoch'], df['valid_loss'], 
             color='#3498db', linewidth=2.5, 
             marker='s', markersize=6, label='Validation Loss', alpha=0.9)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy and Norm ED on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%) / Norm ED (scaled)', fontsize=14, color='#27ae60')
    ax2.plot(df['epoch'], df['accuracy'], 
             color='#27ae60', linewidth=2.5, 
             marker='^', markersize=6, label='Accuracy (%)', alpha=0.9)
    ax2.plot(df['epoch'], df['norm_ED'] * 100, 
             color='#f39c12', linewidth=2.5, 
             marker='D', markersize=6, label='Norm ED (scaled)', alpha=0.9)
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    
    plt.title('⚡ Complete Training Overview', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12)
    
    plt.tight_layout()
    
    overview_path = Path(save_dir) / 'training_overview.png'
    plt.savefig(overview_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Overview plot saved: {overview_path}")
    return overview_path

def plot_all_metrics(csv_file, save_dir='plots'):
    """
    Read CSV and create all individual plots
    
    Args:
        csv_file (str): Path to CSV file
        save_dir (str): Directory to save plots
    """
    
    # Create save directory if not exists
    Path(save_dir).mkdir(exist_ok=True)
    
    # Read CSV file
    print(f"📊 Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"📈 Creating individual plots...")
    
    # Create all plots
    plots_created = []
    plots_created.append(create_loss_plot(df, save_dir))
    plots_created.append(create_accuracy_plot(df, save_dir))
    plots_created.append(create_norm_ed_plot(df, save_dir))
    plots_created.append(create_combined_metrics_plot(df, save_dir))
    plots_created.append(create_overview_plot(df, save_dir))
    
    # Print summary
    print(f"\n🎉 All plots created successfully!")
    print(f"📁 Saved to directory: {save_dir}")
    print(f"📊 Total plots: {len(plots_created)}")
    
    # Print statistics
    best_accuracy = df['accuracy'].max()
    best_norm_ed = df['norm_ED'].max()
    final_train_loss = df['train_loss'].iloc[-1]
    final_valid_loss = df['valid_loss'].iloc[-1]
    
    print(f"\n📈 Quick Stats:")
    print(f"   🎯 Best Accuracy: {best_accuracy:.3f}%")
    print(f"   📊 Best Norm ED: {best_norm_ed:.4f}")
    print(f"   📉 Final Train Loss: {final_train_loss:.5f}")
    print(f"   📉 Final Valid Loss: {final_valid_loss:.5f}")
    
    return plots_created

# Example usage
if __name__ == "__main__":
    # Change this to your CSV file name
    csv_filename = "training_log.csv"
    
    # Create all plots
    plot_all_metrics(csv_filename)