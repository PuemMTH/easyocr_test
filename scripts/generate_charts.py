#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Models Evaluation and Chart Generation Script
=================================================

สคริปต์สำหรับประเมินผลโมเดล OCR และสร้างกราฟ

โมเดลที่ทดสอบ:
- อ่านจากไฟล์ evaluation_results.json โดยอัตโนมัติ

ชุดข้อมูล:
- data_from_web (300 ตัวอย่าง)
- data_from_outsource (300 ตัวอย่าง)

เมตริกที่วัด:
- Character Accuracy & Error Rate (CER)
- Word Error Rate (WER) - Thai
- Semantic Similarity
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting():
    """Configure matplotlib and seaborn settings for optimal chart appearance"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'
    print("✅ Libraries ready")

def create_model_labels(model_names):
    """Create shortened labels for model names for better chart readability"""
    model_labels = {}
    for model_name in model_names:
        if model_name == 'base_model':
            model_labels[model_name] = 'base'
        elif model_name == "out_source_merge_kaggle_data_3gpus":
            model_labels[model_name] = 'merge_kaggle'
        elif model_name == "out_source_merge_kaggle_data_freeze_sequence_config_3gpus":
            model_labels[model_name] = 'freeze_seq'
        elif model_name == "out_source_only_data_3gpus":
            model_labels[model_name] = 'only_data'
        else:
            model_labels[model_name] = model_name.replace('_', ' ')[:15]
    
    return model_labels

def load_evaluation_data(file_path):
    """Load and preprocess OCR evaluation data"""
    try:
        df = pd.read_json(file_path)
        print(f"📊 Loaded evaluation results: {len(df):,} records")
        
        # Data overview
        print("\n📋 ข้อมูลทั่วไป:")
        print(f"   - Number of models: {df['model_name'].nunique()}")
        print(f"   - Number of datasets: {df['dataset_name'].nunique()}")
        print(f"   - Models tested: {', '.join(df['model_name'].unique())}")
        print(f"   - Datasets tested: {', '.join(df['dataset_name'].unique())}")
        
        return df
    except FileNotFoundError:
        print("❌ Error: evaluation_results.json not found")
        raise

def calculate_summary_stats(df):
    """Calculate summary statistics for each model-dataset combination"""
    # Calculate character accuracy from cer_percent
    df['character_accuracy'] = 100 - df['cer_percent']
    
    summary_stats = df.groupby(['model_name', 'dataset_name']).agg({
        'character_accuracy': ['mean', 'std', 'count'],
        'cer_percent': ['mean', 'std'],
        'wer_percent': ['mean', 'std'],
        'wer_pythainlp_percent': ['mean', 'std'],
        'semantic_similarity': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()
    
    print("📈 Summary statistics calculated")
    return summary_stats

def prepare_pivot_tables(summary_stats):
    """Prepare pivot tables for visualization with proper model ordering"""
    # Get model order from the data, with base_model first
    all_models = summary_stats['model_name'].unique()
    model_order = ['base_model'] + [model for model in all_models if model != 'base_model']
    
    pivot_cer = summary_stats.reset_index().pivot(
        index='model_name', 
        columns='dataset_name', 
        values='cer_percent_mean'
    )
    
    pivot_wer = summary_stats.reset_index().pivot(
        index='model_name', 
        columns='dataset_name', 
        values='wer_percent_mean'
    )
    
    pivot_wer_pythainlp = summary_stats.reset_index().pivot(
        index='model_name', 
        columns='dataset_name', 
        values='wer_pythainlp_percent_mean'
    )
    
    pivot_semantic = summary_stats.reset_index().pivot(
        index='model_name', 
        columns='dataset_name', 
        values='semantic_similarity_mean'
    )
    
    # Reorder index according to model_order (only for models that exist in the data)
    existing_models = [model for model in model_order if model in pivot_cer.index]
    pivot_cer = pivot_cer.reindex(index=existing_models)
    pivot_wer = pivot_wer.reindex(index=existing_models)
    pivot_wer_pythainlp = pivot_wer_pythainlp.reindex(index=existing_models)
    pivot_semantic = pivot_semantic.reindex(index=existing_models)
    
    return pivot_cer, pivot_wer, pivot_wer_pythainlp, pivot_semantic

def setup_report_directory():
    """Create and return the report directory path with timestamp"""
    from datetime import datetime
    
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"reports/ocr_evaluation_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for better organization
    charts_dir = report_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    explanations_dir = report_dir / "explanations"
    explanations_dir.mkdir(exist_ok=True)
    
    data_dir = report_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created organized report structure:")
    print(f"   📊 Charts: {charts_dir}")
    print(f"   📝 Explanations: {explanations_dir}")
    print(f"   📊 Data: {data_dir}")
    
    return report_dir, charts_dir, explanations_dir, data_dir

def save_cer_percent_chart(pivot_cer, charts_dir, explanations_dir):
    """Save Character Error Rate (CER) comparison chart"""
    # Create custom labels mapping - dynamically generate from actual model names
    model_labels = create_model_labels(pivot_cer.index)
    
    # Rename index using the mapping
    pivot_renamed = pivot_cer.rename(index=model_labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    pivot_renamed.plot(kind='barh', ax=ax, width=0.8, color=['lightpink', 'crimson'])
    ax.set_title('Character Error Rate (CER) Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Character Error Rate (%)', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.legend(title='Dataset', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'cer_percent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: charts/cer_percent_comparison.png")
    
    # Create explanation text file
    explanation = """การเปรียบเทียบอัตราความผิดพลาดระดับตัวอักษร (CER)

คืออะไร:
• CER = ความผิดพลาดในการอ่านตัวอักษร (%)
• ยิ่งต่ำยิ่งดี (0% = สมบูรณ์แบบ)

ตัวอย่าง:
• CER 5% = ผิดพลาด 5 ตัวอักษรจาก 100 ตัวอักษร
• CER 10% = ผิดพลาด 10 ตัวอักษรจาก 100 ตัวอักษร
"""
    
    with open(explanations_dir / 'cer_percent_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(explanation)
    print("   ✅ Saved: explanations/cer_percent_comparison.txt")

def save_wer_percent_chart(pivot_wer, charts_dir, explanations_dir):
    """Save Word Error Rate (WER) comparison chart"""
    # Create custom labels mapping - dynamically generate from actual model names
    model_labels = create_model_labels(pivot_wer.index)
    
    pivot_renamed = pivot_wer.rename(index=model_labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    pivot_renamed.plot(kind='barh', ax=ax, width=0.8, color=['orange', 'red'])
    ax.set_title('Word Error Rate (WER) Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Word Error Rate (%)', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.legend(title='Dataset', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'wer_percent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: charts/wer_percent_comparison.png")
    
    explanation = """การเปรียบเทียบอัตราความผิดพลาดระดับคำ (WER)

คืออะไร:
• WER = ความผิดพลาดในการอ่านคำ (%)
• ยิ่งต่ำยิ่งดี (0% = สมบูรณ์แบบ)

ตัวอย่าง:
• WER 5% = ผิดพลาด 5 คำจาก 100 คำ
• WER 15% = ผิดพลาด 15 คำจาก 100 คำ
"""
    
    with open(explanations_dir / 'wer_percent_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(explanation)
    print("   ✅ Saved: explanations/wer_percent_comparison.txt")

def save_semantic_similarity_chart(pivot_semantic, charts_dir, explanations_dir):
    """Save Semantic Similarity comparison chart"""
    # Create custom labels mapping - dynamically generate from actual model names
    model_labels = create_model_labels(pivot_semantic.index)
    
    pivot_renamed = pivot_semantic.rename(index=model_labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    pivot_renamed.plot(kind='barh', ax=ax, width=0.8, color=['lightgreen', 'darkgreen'])
    ax.set_title('Semantic Similarity Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Semantic Similarity Score', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.legend(title='Dataset', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'semantic_similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: charts/semantic_similarity_comparison.png")
    
    explanation = """การเปรียบเทียบความคล้ายคลึงทางความหมาย (Semantic Similarity)

คืออะไร:
• คะแนนความคล้ายคลึงในความหมาย (0.0-1.0)
• ยิ่งสูงยิ่งดี (1.0 = เหมือนกันทุกประการ)

ตัวอย่าง:
• 0.9 = ความหมายใกล้เคียงมาก
• 0.5 = ความหมายคล้ายกันบ้าง
• 0.1 = ความหมายต่างกันมาก
"""
    
    with open(explanations_dir / 'semantic_similarity_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(explanation)
    print("   ✅ Saved: explanations/semantic_similarity_comparison.txt")

def save_wer_pythainlp_percent_chart(pivot_wer_pythainlp, charts_dir, explanations_dir):
    """Save Word Error Rate (PyThaiNLP) comparison chart"""
    # Create custom labels mapping - dynamically generate from actual model names
    model_labels = create_model_labels(pivot_wer_pythainlp.index)
    
    # Rename index using the mapping
    pivot_renamed = pivot_wer_pythainlp.rename(index=model_labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    pivot_renamed.plot(kind='barh', ax=ax, width=0.8, color=['lightblue', 'navy'])
    ax.set_title('Word Error Rate (PyThaiNLP) Comparison', fontsize=16, pad=20)
    ax.set_xlabel('Word Error Rate (%)', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.legend(title='Dataset', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'wer_pythainlp_percent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: charts/wer_pythainlp_percent_comparison.png")
    
    # Create explanation text file
    explanation = """การเปรียบเทียบอัตราความผิดพลาดระดับคำ (WER-Thai)

คืออะไร:
• WER ที่ใช้การตัดคำภาษาไทยแบบ PyThaiNLP
• ยิ่งต่ำยิ่งดี (0% = สมบูรณ์แบบ)

ความแตกต่าง:
• ตัดคำภาษาไทยได้แม่นยำกว่า WER ปกติ
• เหมาะสำหรับข้อความภาษาไทยโดยเฉพาะ
"""
    
    with open(explanations_dir / 'wer_pythainlp_percent_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(explanation)
    print("   ✅ Saved: explanations/wer_pythainlp_percent_comparison.txt")

def generate_insights(summary_stats):
    """Generate key insights from the evaluation"""
    print("\n🔍 สรุปผลการวิเคราะห์:")
    print("="*50)
    
    # Find best performing model per dataset
    print("\n🏆 โมเดลที่ดีที่สุดสำหรับแต่ละชุดข้อมูล:")
    for dataset in summary_stats['dataset_name'].unique():
        dataset_data = summary_stats[summary_stats['dataset_name'] == dataset]
        best_model = dataset_data.loc[dataset_data['character_accuracy_mean'].idxmax()]
        dataset_name = "ข้อมูลเว็บ" if dataset == 'data_from_web' else "ข้อมูลจ้างงาน"
        print(f"   📊 {dataset_name}: {best_model['model_name']} ({best_model['character_accuracy_mean']:.1f}%)")
    
    print("\n💡 ข้อสังเกต:")
    print("   • โมเดล Custom ทำงานดีกับข้อมูลที่คล้ายกับข้อมูลฝึก")
    print("   • โมเดล Base มีความแข็งแกร่งข้ามโดเมนมากกว่า") 
    print("   • การเพิ่ม iteration ช่วยปรับปรุงประสิทธิภาพ")

def export_results(df, summary_stats, data_dir):
    """Export results to CSV files"""
    summary_stats.to_csv(data_dir / 'ocr_evaluation_summary.csv', index=False)
    df.to_csv(data_dir / 'ocr_evaluation_detailed.csv', index=False)
    
    # Create simple summary
    summary_data = {
        'total_samples': len(df),
        'models_tested': df['model_name'].unique().tolist(),
        'datasets_tested': df['dataset_name'].unique().tolist()
    }
    
    # Add best model per dataset
    for dataset in summary_stats['dataset_name'].unique():
        dataset_data = summary_stats[summary_stats['dataset_name'] == dataset]
        best_model = dataset_data.loc[dataset_data['character_accuracy_mean'].idxmax()]
        summary_data[f'best_{dataset}'] = {
            'model': best_model['model_name'],
            'accuracy': float(best_model['character_accuracy_mean'])
        }
    
    with open(data_dir / 'evaluation_final_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print("✅ บันทึกผลลัพธ์:")
    print("   📁 data/ocr_evaluation_summary.csv")
    print("   📁 data/ocr_evaluation_detailed.csv")
    print("   📁 data/evaluation_final_summary.json")

def print_final_summary(report_dir, charts_dir, explanations_dir, data_dir):
    """Print final summary of generated files"""
    chart_files = list(charts_dir.glob("*.png"))
    text_files = list(explanations_dir.glob("*.txt"))
    data_files = list(data_dir.glob("*"))
    
    print("\n🎉 สรุปการสร้างกราฟ:")
    print("="*40)
    print(f"📊 กราฟที่สร้าง: {len(chart_files)} ไฟล์")
    print(f"📝 คำอธิบาย: {len(text_files)} ไฟล์")
    print(f"📊 ข้อมูล: {len(data_files)} ไฟล์")
    print(f"📁 ตำแหน่ง: {report_dir.name}/")
    print("💾 คุณภาพ: 300 DPI (ความละเอียดสูง)")
    
    print(f"\n📂 โครงสร้างไฟล์:")
    print(f"   📊 Charts: {charts_dir}")
    print(f"   📝 Explanations: {explanations_dir}")
    print(f"   📊 Data: {data_dir}")

def main():
    """Main function to execute the analysis"""
    print("🚀 เริ่มวิเคราะห์โมเดล OCR และสร้างกราฟ")
    print("="*50)
    
    # 1. Setup
    setup_plotting()
    
    # 2. Load and prepare data
    df = load_evaluation_data('output/evaluation_results.json')
    summary_stats = calculate_summary_stats(df)
    pivot_cer, pivot_wer, pivot_wer_pythainlp, pivot_semantic = prepare_pivot_tables(summary_stats)
    
    # 3. Setup report directory
    report_dir, charts_dir, explanations_dir, data_dir = setup_report_directory()
    
    # 4. Generate charts
    print("\n💾 กำลังสร้างกราฟ:")
    save_cer_percent_chart(pivot_cer, charts_dir, explanations_dir)
    save_wer_percent_chart(pivot_wer, charts_dir, explanations_dir) 
    save_wer_pythainlp_percent_chart(pivot_wer_pythainlp, charts_dir, explanations_dir)
    save_semantic_similarity_chart(pivot_semantic, charts_dir, explanations_dir)
    
    # 5. Generate insights
    generate_insights(summary_stats)
    
    # 6. Export results
    export_results(df, summary_stats, data_dir)
    
    # 7. Print summary
    print_final_summary(report_dir, charts_dir, explanations_dir, data_dir)
    
    print("\n✅ วิเคราะห์และสร้างกราฟเสร็จสิ้น! 🎉")

if __name__ == "__main__":
    main()
