# OCR Presentation Examples

ตัวอย่างการแสดงผลผลลัพธ์การประเมิน OCR สำหรับการนำเสนอ

## 📁 ไฟล์ตัวอย่างสำหรับการนำเสนอ (Presentation Files)

### 1. `01_presentation_basic_results.py`
**การแสดงผลลัพธ์พื้นฐาน (Basic Results)**
- แสดง MIN, MAX, MEAN CER สำหรับแต่ละโมเดลและชุดข้อมูล
- ใช้สีเพื่อแยกแยะระหว่าง Reference และ Hypothesis
- เหมาะสำหรับการนำเสนอผลลัพธ์เบื้องต้น

```bash
python examples/01_presentation_basic_results.py
```

### 2. `02_presentation_performance_summary.py`
**สรุปประสิทธิภาพ (Performance Summary)**
- คำนวณสถิติเชิงลึก (Mean, Std, Min, Max, Median CER)
- แสดงการกระจายของความแม่นยำ (Perfect, High, Medium, Low Accuracy)
- ให้คะแนนประสิทธิภาพ (EXCELLENT, GOOD, FAIR, POOR)
- แสดงโมเดลที่ดีที่สุดสำหรับแต่ละชุดข้อมูล

```bash
python examples/02_presentation_performance_summary.py
```

### 3. `03_presentation_error_analysis.py`
**วิเคราะห์ข้อผิดพลาด (Error Analysis)**
- วิเคราะห์ข้อผิดพลาดระดับตัวอักษรและคำ
- แสดงกรณีที่แย่ที่สุด (Top 3 Worst Cases)
- วิเคราะห์รูปแบบข้อผิดพลาดที่พบบ่อย
- แสดงการกระจายของข้อผิดพลาดตามช่วง CER

```bash
python examples/03_presentation_error_analysis.py
```

### 4. `04_presentation_comparison_table.py`
**ตารางเปรียบเทียบ (Comparison Table)**
- สร้างตารางเปรียบเทียบระหว่างโมเดลและชุดข้อมูล
- แสดงอันดับ (Ranking) ของโมเดลสำหรับแต่ละชุดข้อมูล
- ระบุโมเดลที่ดีที่สุดสำหรับแต่ละชุดข้อมูล
- แสดงสถิติรายละเอียดพร้อมคะแนนประสิทธิภาพ

```bash
python examples/04_presentation_comparison_table.py
```

### 5. `05_presentation_summary.py`
**สรุปครบถ้วน (Comprehensive Summary)**
- Executive Summary สำหรับผู้บริหาร
- สรุปรายละเอียดประสิทธิภาพของทุกโมเดล
- ให้คำแนะนำ (Recommendations) ตามผลการประเมิน
- แสดง Key Insights จากผลการประเมิน

```bash
python examples/05_presentation_summary.py
```

## 🎯 การใช้งานสำหรับการนำเสนอ (Usage for Presentations)

### ข้อกำหนด (Requirements)
```bash
pip install pandas colorama numpy
```

### ข้อมูลที่ใช้ (Data Source)
ไฟล์ CSV ที่มีคอลัมน์ต่อไปนี้:
- `model_name`: ชื่อโมเดล
- `dataset_name`: ชื่อชุดข้อมูล
- `cer`: Character Error Rate
- `reference_words`: คำอ้างอิง (ในรูปแบบ list string)
- `hypothesis_words`: คำที่โมเดลทำนาย (ในรูปแบบ list string)

### ตัวอย่างการเรียกใช้ (Example Usage)

```python
# สำหรับการนำเสนอผลลัพธ์พื้นฐาน
from examples import presentation_basic_results
presentation_basic_results.present_basic_results()

# สำหรับการนำเสนอสรุปประสิทธิภาพ
from examples import presentation_performance_summary
presentation_performance_summary.present_performance_summary()
```

## 📊 ฟีเจอร์หลักสำหรับการนำเสนอ (Key Presentation Features)

### 🎨 การแสดงผลด้วยสี (Color Coding)
- 🟢 **เขียว**: Reference text, Perfect matches, EXCELLENT rating
- 🔵 **น้ำเงิน**: Hypothesis text, Dataset names, FAIR rating
- 🔴 **แดง**: Model names, High error rates, POOR rating
- 🟡 **เหลือง**: Warning levels, Statistics, GOOD rating
- 🟣 **ม่วง**: Error analysis, Headers
- 🔵 **ฟ้า**: Headers and separators

### 📈 สถิติที่แสดง (Displayed Statistics)
- **CER Metrics**: Mean, Standard Deviation, Min, Max, Median
- **Accuracy Distribution**: Perfect, High, Medium, Low accuracy percentages
- **Performance Ratings**: EXCELLENT, GOOD, FAIR, POOR
- **Rankings**: Model performance rankings by dataset
- **Error Analysis**: Character and word-level error patterns

### 🎯 ประเภทการนำเสนอ (Presentation Types)

#### 1. **Executive Summary** (`05_presentation_summary.py`)
- เหมาะสำหรับผู้บริหาร
- แสดงภาพรวมและผลลัพธ์สำคัญ
- ให้คำแนะนำและ Key Insights

#### 2. **Technical Deep Dive** (`03_presentation_error_analysis.py`)
- เหมาะสำหรับทีมเทคนิค
- วิเคราะห์ข้อผิดพลาดเชิงลึก
- แสดงรูปแบบข้อผิดพลาดที่พบบ่อย

#### 3. **Performance Comparison** (`04_presentation_comparison_table.py`)
- เหมาะสำหรับการเปรียบเทียบโมเดล
- แสดงตารางเปรียบเทียบและอันดับ
- ระบุโมเดลที่ดีที่สุด

#### 4. **Basic Results** (`01_presentation_basic_results.py`)
- เหมาะสำหรับการนำเสนอเบื้องต้น
- แสดงตัวอย่างผลลัพธ์ที่สำคัญ
- ง่ายต่อการเข้าใจ

#### 5. **Performance Summary** (`02_presentation_performance_summary.py`)
- เหมาะสำหรับการสรุปประสิทธิภาพ
- แสดงสถิติครบถ้วน
- ให้คะแนนประสิทธิภาพ

## 🔧 การปรับแต่งสำหรับการนำเสนอ (Customization for Presentations)

### เปลี่ยนไฟล์ข้อมูล (Change Data File)
```python
# ในแต่ละไฟล์ เปลี่ยนค่า default ของ csv_path
present_basic_results('your_custom_path.csv')
```

### ปรับแต่งเกณฑ์การให้คะแนน (Customize Rating Criteria)
```python
# ปรับเกณฑ์การให้คะแนนในฟังก์ชัน calculate_performance_metrics()
if mean_cer < 0.05:  # เปลี่ยนจาก 0.1 เป็น 0.05
    rating = "EXCELLENT"
```

### เพิ่มสถิติใหม่ (Add New Statistics)
```python
# เพิ่มการคำนวณสถิติใหม่
'new_metric': subset['cer'].quantile(0.95)  # 95th percentile
```

## 📋 โครงสร้างไฟล์ (File Structure)

```
examples/
├── 01_presentation_basic_results.py      # ผลลัพธ์พื้นฐาน
├── 02_presentation_performance_summary.py # สรุปประสิทธิภาพ
├── 03_presentation_error_analysis.py     # วิเคราะห์ข้อผิดพลาด
├── 04_presentation_comparison_table.py   # ตารางเปรียบเทียบ
├── 05_presentation_summary.py            # สรุปครบถ้วน
└── README_PRESENTATION.md                # คู่มือการใช้งาน
```

## 🎯 ข้อแนะนำสำหรับการนำเสนอ (Presentation Tips)

### 1. **เลือกไฟล์ตามผู้ฟัง**
- **ผู้บริหาร**: ใช้ `05_presentation_summary.py`
- **ทีมเทคนิค**: ใช้ `03_presentation_error_analysis.py`
- **การเปรียบเทียบ**: ใช้ `04_presentation_comparison_table.py`

### 2. **ลำดับการนำเสนอ**
1. เริ่มด้วย Executive Summary
2. แสดงผลลัพธ์พื้นฐาน
3. วิเคราะห์ข้อผิดพลาด
4. สรุปและให้คำแนะนำ

### 3. **การปรับแต่งสี**
- ใช้สีเพื่อเน้นจุดสำคัญ
- ใช้สีแดงสำหรับข้อผิดพลาด
- ใช้สีเขียวสำหรับผลลัพธ์ที่ดี

## 📝 หมายเหตุ (Notes)

- ไฟล์ทั้งหมดใช้ข้อมูลจริงจาก `reports/ocr_evaluation_20250727_235535/data/ocr_evaluation_detailed.csv`
- สามารถปรับแต่งเกณฑ์การให้คะแนนได้ตามความเหมาะสม
- ฟังก์ชันทั้งหมดสามารถเรียกใช้แยกกันได้
- เหมาะสำหรับการนำเสนอในรูปแบบ Terminal หรือ Console 