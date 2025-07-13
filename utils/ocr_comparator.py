import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw
import os
import json
from .metrics import OCRMetrics

class OcrComparator:
    def __init__(self, reader_custom, reader_base):
        self.reader_custom = reader_custom
        self.reader_base = reader_base
        self.metrics = OCRMetrics()
        self.setup_thai_font()

    def setup_thai_font(self):
        """Setup Thai font for matplotlib if available"""
        try:
            thai_fonts = ['Tahoma', 'Arial Unicode MS', 'Noto Sans Thai', 'TH Sarabun New']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in thai_fonts:
                if font in available_fonts:
                    plt.rcParams['font.family'] = [font]
                    print(f"✅ Using font: {font}")
                    return font
            
            print("⚠️  Using default font (may not display Thai correctly)")
            return None
        except Exception as e:
            print(f"Font setup warning: {e}")
            return None

    def draw_boxes_enhanced(self, image_path, results, model_name, color='red'):
        """Enhanced version of draw_boxes function"""
        try:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
            
            for i, result in enumerate(results_sorted):
                box, text, confidence = result
                
                if confidence < 0.2:
                    continue
                    
                flattened_box = [tuple(point) for point in box]
                draw.polygon(flattened_box, outline=color, width=3)
                
                label = f"{i+1}. {text} ({confidence:.2f})"
                
                try:
                    bbox = draw.textbbox((box[0][0], box[0][1] - 25), label)
                    draw.rectangle(bbox, fill=color, outline=color)
                    draw.text((box[0][0], box[0][1] - 25), label, fill='white')
                except Exception:
                    draw.text((box[0][0], box[0][1] - 20), label, fill=color)
                    
            return image
        except Exception as e:
            print(f"Error drawing boxes for {model_name}: {e}")
            return None

    def compare_models_on_image(self, image_path, filename):
        """Compare both models on a single image with side-by-side display"""
        print(f"\n{'='*80}")
        print(f"🔍 Analyzing: {filename}")
        print(f"{'='*80}")
        
        try:
            print("⏳ Processing with both models...")
            results_custom = self.reader_custom.readtext(image_path, detail=1, paragraph=False)
            results_base = self.reader_base.readtext(image_path, detail=1, paragraph=False)
            print("✅ Processing complete!")
            
            img_custom = self.draw_boxes_enhanced(image_path, results_custom, "Custom", color='blue')
            img_base = self.draw_boxes_enhanced(image_path, results_base, "Base", color='red')
            
            if img_custom is None or img_base is None:
                print("❌ Error creating annotated images")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
            
            ax1.imshow(img_custom)
            ax1.set_title(f"🔵 Custom Model - {len(results_custom)} detections\n{filename}", 
                          fontsize=16, fontweight='bold', color='blue')
            ax1.axis('off')
            
            ax2.imshow(img_base)
            ax2.set_title(f"🔴 Base Model - {len(results_base)} detections\n{filename}", 
                          fontsize=16, fontweight='bold', color='red')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            self.print_comparison_summary(results_custom, results_base)
            self.print_detailed_comparison(results_custom, results_base)
            
        except Exception as e:
            print(f"❌ Error in comparison: {e}")

    def print_comparison_summary(self, results_custom, results_base):
        """Print summary statistics"""
        total_custom = len(results_custom)
        total_base = len(results_base)
        
        high_conf_custom = len([r for r in results_custom if r[2] > 0.7])
        high_conf_base = len([r for r in results_base if r[2] > 0.7])
        
        med_conf_custom = len([r for r in results_custom if 0.4 <= r[2] <= 0.7])
        med_conf_base = len([r for r in results_base if 0.4 <= r[2] <= 0.7])
        
        low_conf_custom = len([r for r in results_custom if 0.2 <= r[2] < 0.4])
        low_conf_base = len([r for r in results_base if 0.2 <= r[2] < 0.4])
        
        print("\n📊 Detection Summary:")
        print(f"{'Metric':<25} | {'Custom':<8} | {'Base':<8} | {'Diff':<8}")
        print(f"{'-'*55}")
        print(f"{'Total Detections':<25} | {total_custom:<8} | {total_base:<8} | {total_custom - total_base:+d}")
        print(f"{'High Conf (>0.7)':<25} | {high_conf_custom:<8} | {high_conf_base:<8} | {high_conf_custom - high_conf_base:+d}")
        print(f"{'Med Conf (0.4-0.7)':<25} | {med_conf_custom:<8} | {med_conf_base:<8} | {med_conf_custom - med_conf_base:+d}")
        print(f"{'Low Conf (0.2-0.4)':<25} | {low_conf_custom:<8} | {low_conf_base:<8} | {low_conf_custom - low_conf_base:+d}")

    def print_detailed_comparison(self, results_custom, results_base):
        """Print detailed text comparison"""
        good_custom = sorted([r for r in results_custom if r[2] > 0.3], key=lambda x: x[2], reverse=True)
        good_base = sorted([r for r in results_base if r[2] > 0.3], key=lambda x: x[2], reverse=True)
        
        print("\n📝 Text Detection Results (Confidence > 0.3):")
        print(f"{'='*85}")
        print(f"{'🔵 Custom Model':<42} | {'🔴 Base Model':<42}")
        print(f"{'='*85}")
        
        max_results = max(len(good_custom), len(good_base))
        
        for i in range(min(max_results, 10)):
            custom_str = f"{i+1}. '{good_custom[i][1]}' ({good_custom[i][2]:.3f})" if i < len(good_custom) else "-"
            base_str = f"{i+1}. '{good_base[i][1]}' ({good_base[i][2]:.3f})" if i < len(good_base) else "-"
            
            custom_display = custom_str[:40] + ".." if len(custom_str) > 42 else custom_str
            base_display = base_str[:40] + ".." if len(base_str) > 42 else base_str
            
            print(f"{custom_display:<42} | {base_display:<42}")

    def run_comparison_poc(self, folder_images, pd_files, num_samples=3):
        """Run comprehensive comparison POC"""
        print("🚀 Starting OCR Model Comparison POC")
        print("=" * 80)

        if pd_files.empty:
            print("❌ No files to process - CSV is empty")
            return

        sample_files = pd_files.sample(min(num_samples, len(pd_files)), random_state=42)
        print(f"📁 Processing {len(sample_files)} sample images...")

        for _, row in sample_files.iterrows():
            image_path = os.path.join(folder_images, row['new_image_name'])
            
            if os.path.exists(image_path):
                self.compare_models_on_image(image_path, row['new_image_name'])
            else:
                print(f"❌ File not found: {image_path}")
            
            print(f"\n{'-'*80}")

        print("✅ POC Analysis Complete!")
    
    def calculate_metrics_with_ground_truth(self, results_custom, results_base, ground_truth_text):
        """Calculate metrics comparing OCR results with ground truth"""
        custom_text = ' '.join([result[1] for result in results_custom if result[2] > 0.3])
        base_text = ' '.join([result[1] for result in results_base if result[2] > 0.3])
        
        custom_metrics = self.metrics.calculate_all_metrics(ground_truth_text, custom_text)
        base_metrics = self.metrics.calculate_all_metrics(ground_truth_text, base_text)
        
        return {
            'custom_model': {
                'extracted_text': custom_text,
                'confidence_scores': [result[2] for result in results_custom],
                'detection_count': len(results_custom),
                'high_confidence_count': len([r for r in results_custom if r[2] > 0.7]),
                'metrics': custom_metrics
            },
            'base_model': {
                'extracted_text': base_text,
                'confidence_scores': [result[2] for result in results_base],
                'detection_count': len(results_base),
                'high_confidence_count': len([r for r in results_base if r[2] > 0.7]),
                'metrics': base_metrics
            },
            'ground_truth': ground_truth_text
        }
    
    def save_metrics_to_json(self, metrics_data, filename_prefix):
        """Save metrics data to JSON file"""
        output_file = f"{filename_prefix}_metrics.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Metrics saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
            return None
