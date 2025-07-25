#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def analyze_negative_accuracy():
    """วิเคราะห์กรณีที่ character_accuracy ติดลบ"""
    
    print("🔍 กำลังวิเคราะห์ข้อมูล...")
    
    # อ่านข้อมูลจาก CSV
    try:
        df = pd.read_csv('ocr_evaluation_detailed.csv')
        print(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")
    except FileNotFoundError:
        print("❌ ไม่พบไฟล์ ocr_evaluation_detailed.csv")
        return
    
    # แสดงข้อมูลทั่วไป
    print(f"\n📊 ข้อมูลทั่วไป:")
    print(f"   - จำนวนโมเดล: {df['model_name'].nunique()}")
    print(f"   - จำนวนชุดข้อมูล: {df['dataset_name'].nunique()}")
    print(f"   - คอลัมน์ทั้งหมด: {list(df.columns)}")
    
    # ตรวจสอบ CER values
    print(f"\n📈 สถิติ CER percent:")
    print(f"   - Min CER: {df['cer_percent'].min():.4f}%")
    print(f"   - Max CER: {df['cer_percent'].max():.4f}%")
    print(f"   - Mean CER: {df['cer_percent'].mean():.4f}%")
    print(f"   - Std CER: {df['cer_percent'].std():.4f}%")
    
    # คำนวณ character_accuracy
    df['character_accuracy_calculated'] = 100 - df['cer_percent']
    
    print(f"\n🧮 สถิติ Character Accuracy (100 - CER):")
    print(f"   - Min Accuracy: {df['character_accuracy_calculated'].min():.4f}%")
    print(f"   - Max Accuracy: {df['character_accuracy_calculated'].max():.4f}%")
    print(f"   - Mean Accuracy: {df['character_accuracy_calculated'].mean():.4f}%")
    
    # ตรวจหากรณีที่ติดลบ
    negative_cases = df[df['character_accuracy_calculated'] < 0]
    
    if len(negative_cases) > 0:
        print(f"\n⚠️  พบ Character Accuracy ติดลบ: {len(negative_cases):,} กรณี")
        print(f"   - คิดเป็น {(len(negative_cases)/len(df)*100):.2f}% ของข้อมูลทั้งหมด")
        
        # แสดงตัวอย่างกรณีที่แย่ที่สุด
        worst_cases = negative_cases.nsmallest(5, 'character_accuracy_calculated')
        
        print(f"\n🔴 5 กรณีที่แย่ที่สุด:")
        for i, (idx, row) in enumerate(worst_cases.iterrows(), 1):
            print(f"   {i}. Model: {row['model_name']}")
            print(f"      Dataset: {row['dataset_name']}")
            print(f"      CER: {row['cer_percent']:.2f}%")
            print(f"      Character Accuracy: {row['character_accuracy_calculated']:.2f}%")
            
            # ใช้คอลัมน์ที่มีจริง
            if 'ground_truth' in df.columns and 'ocr_text' in df.columns:
                print(f"      Ground Truth: {str(row['ground_truth'])[:80]}...")
                print(f"      OCR Result: {str(row['ocr_text'])[:80]}...")
            elif 'reference_normalized' in df.columns and 'hypothesis_normalized' in df.columns:
                print(f"      Reference: {str(row['reference_normalized'])[:80]}...")
                print(f"      Hypothesis: {str(row['hypothesis_normalized'])[:80]}...")
            print("")
    else:
        print(f"\n✅ ไม่พบ Character Accuracy ติดลบ")
    
    # วิเคราะห์ตาม model และ dataset
    print(f"\n📊 สถิติตาม Model:")
    model_stats = df.groupby('model_name')['character_accuracy_calculated'].agg(['min', 'max', 'mean', 'count']).round(2)
    print(model_stats)
    
    print(f"\n📊 สถิติตาม Dataset:")
    dataset_stats = df.groupby('dataset_name')['character_accuracy_calculated'].agg(['min', 'max', 'mean', 'count']).round(2)
    print(dataset_stats)
    
    # วิเคราะห์ตาม Model + Dataset
    print(f"\n📊 สถิติตาม Model x Dataset:")
    combined_stats = df.groupby(['model_name', 'dataset_name'])['character_accuracy_calculated'].agg(['min', 'max', 'mean', 'count']).round(2)
    print(combined_stats)
    
    # ตรวจสอบกรณีที่ CER > 100%
    high_cer = df[df['cer_percent'] > 100]
    
    if len(high_cer) > 0:
        print(f"\n🚨 พบ CER > 100%: {len(high_cer):,} กรณี")
        print(f"   - CER สูงสุด: {high_cer['cer_percent'].max():.2f}%")
        
        # แสดงตัวอย่าง
        extreme_cases = high_cer.nlargest(3, 'cer_percent')
        print(f"\n🔥 3 กรณีที่ CER สูงที่สุด:")
        for i, (idx, row) in enumerate(extreme_cases.iterrows(), 1):
            print(f"   {i}. CER: {row['cer_percent']:.2f}% | Accuracy: {row['character_accuracy_calculated']:.2f}%")
            print(f"      Model: {row['model_name']} | Dataset: {row['dataset_name']}")
            if 'ground_truth' in df.columns and 'ocr_text' in df.columns:
                print(f"      Truth: '{str(row['ground_truth'])[:50]}...'")
                print(f"      OCR: '{str(row['ocr_text'])[:50]}...'")
            print("")
    else:
        print(f"\n✅ ไม่พบ CER > 100%")
    
    # วิเคราะห์การกระจายตัว
    print(f"\n📈 การกระจายตัวของ CER:")
    bins = [0, 10, 20, 30, 50, 100, 200, float('inf')]
    labels = ['0-10%', '10-20%', '20-30%', '30-50%', '50-100%', '100-200%', '>200%']
    df['cer_range'] = pd.cut(df['cer_percent'], bins=bins, labels=labels, right=False)
    cer_distribution = df['cer_range'].value_counts().sort_index()
    
    for range_label, count in cer_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"   {range_label}: {count:,} กรณี ({percentage:.1f}%)")
    
    return df

if __name__ == "__main__":
    df_analyzed = analyze_negative_accuracy()
