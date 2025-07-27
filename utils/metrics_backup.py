import jiwer
from pythainlp import word_tokenize
from sentence_transformers import SentenceTransformer, util
import unicodedata
import re
import os
from typing import Dict, Any


class OCRMetrics:
    def __init__(self, semantic_model: str = 'distiluse-base-multilingual-cased', local_model_path: str = None):
        """
        Initialize OCR metrics calculator
        
        Args:
            semantic_model: SentenceTransformer model name for semantic similarity (None for offline mode)
            local_model_path: Local path to model folder (for offline mode)
        """
        self.semantic_model = None
        
        # Offline mode
        if semantic_model is None:
            print("⚠️  Semantic model disabled (offline mode)")
            return
        
        # Try local model path first if provided
        if local_model_path and os.path.exists(local_model_path):
            try:
                self.semantic_model = SentenceTransformer(local_model_path)
                print(f"✅ Loaded semantic model from local path: {local_model_path}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load from local path {local_model_path}: {e}")
        
        # Try online model
        try:
            self.semantic_model = SentenceTransformer(semantic_model)
            print(f"✅ Loaded semantic model: {semantic_model}")
        except Exception as e:
            print(f"⚠️  Could not load semantic model: {e}")
            print("💡 To use offline mode, download model first:")
            print(f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{semantic_model}', cache_folder='./models')\"")
    
    def download_model_to_local(self, model_name: str = 'distiluse-base-multilingual-cased', local_path: str = './models'):
        """
        Download model to local folder for offline use
        
        Args:
            model_name: Model name to download
            local_path: Local folder to save model
        """
        try:
            os.makedirs(local_path, exist_ok=True)
            print(f"🔄 Downloading {model_name} to {local_path}...")
            model = SentenceTransformer(model_name, cache_folder=local_path)
            print(f"✅ Model downloaded successfully to {local_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False

    def _normalize_text(self, text: str) -> str:
        """
        Normalize Unicode text and whitespace for consistent OCR evaluation
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with consistent Unicode and whitespace
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Whitespace normalization
        # Convert newlines, tabs, and carriage returns to spaces
        text = re.sub(r'[\n\t\r]+', ' ', text)
        
        # Merge multiple consecutive spaces into single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate using jiwer
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            CER value between 0 and 1
        """
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        if not ref:
            return 1.0 if hyp else 0.0
        
        return jiwer.cer(ref, hyp)
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate using jiwer (space-separated words)
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            WER value between 0 and 1
        """
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        if not ref:
            return 1.0 if hyp else 0.0
        
        return jiwer.wer(ref, hyp)
    
    def calculate_wer_pythainlp(self, reference: str, hypothesis: str, 
                               engine: str = 'newmm') -> float:
        """
        Calculate WER using pythainlp tokenization for both reference and hypothesis
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            engine: pythainlp tokenization engine ('newmm', 'longest', 'icu')
            
        Returns:
            WER value between 0 and 1
        """
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        if not ref:
            return 1.0 if hyp else 0.0
        
        try:
            # Tokenize both reference and hypothesis using pythainlp
            ref_tokens = word_tokenize(ref, engine=engine, keep_whitespace=False)
            hyp_tokens = word_tokenize(hyp, engine=engine, keep_whitespace=False)
            
            # Join tokens back to space-separated strings for jiwer
            ref_tokenized = ' '.join(ref_tokens)
            hyp_tokenized = ' '.join(hyp_tokens)
            
            return jiwer.wer(ref_tokenized, hyp_tokenized)
        except Exception as e:
            print(f"Tokenization error: {e}")
            return 1.0
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity using SentenceTransformer and cosine similarity
        
        Args:
            reference: Ground truth text  
            hypothesis: OCR output text
            
        Returns:
            Cosine similarity between 0 and 1
        """
        if not self.semantic_model:
            print("Semantic model not loaded")
            return 0.0
            
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        if not ref or not hyp:
            return 0.0
        
        try:
            # Encode texts to embeddings
            emb1 = self.semantic_model.encode(ref, convert_to_tensor=True)
            emb2 = self.semantic_model.encode(hyp, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(emb1, emb2)
            
            return float(similarity.item())
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 0.0
    
    def calculate_edit_operations(self, reference: str, hypothesis: str) -> Dict[str, int]:
        """
        Get detailed edit operations using jiwer
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            Dictionary with insertions, deletions, substitutions, hits
        """
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        if not ref and not hyp:
            return {'insertions': 0, 'deletions': 0, 'substitutions': 0, 'hits': 0}
        
        try:
            # Use jiwer to get detailed word-level operations
            result = jiwer.process_words(ref, hyp)
            
            return {
                'insertions': result.insertions,
                'deletions': result.deletions,
                'substitutions': result.substitutions,
                'hits': result.hits
            }
        except Exception as e:
            print(f"Edit operations error: {e}")
            return {'insertions': 0, 'deletions': 0, 'substitutions': 0, 'hits': 0}
    
    def evaluate(self, reference: str, hypothesis: str, 
                 pythainlp_engine: str = 'newmm') -> Dict[str, Any]:
        """
        Calculate all metrics for a reference-hypothesis pair
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            pythainlp_engine: Engine for pythainlp tokenization
            
        Returns:
            Dictionary containing all calculated metrics
        """
        results = {}
        
        # 1. Character Error Rate
        results['cer'] = self.calculate_cer(reference, hypothesis)
        results['cer_percent'] = results['cer'] * 100
        
        # 2. Word Error Rate (standard)
        results['wer'] = self.calculate_wer(reference, hypothesis)
        results['wer_percent'] = results['wer'] * 100
        
        # 3. Word Error Rate with pythainlp tokenization
        results['wer_pythainlp'] = self.calculate_wer_pythainlp(
            reference, hypothesis, pythainlp_engine)
        results['wer_pythainlp_percent'] = results['wer_pythainlp'] * 100
        
        # 4. Semantic similarity
        results['semantic_similarity'] = self.calculate_semantic_similarity(
            reference, hypothesis)
        
        # 5. Edit operations
        results['edit_operations'] = self.calculate_edit_operations(
            reference, hypothesis)
        
        # Add summary info
        results['reference_length'] = len(self._normalize_text(reference))
        results['hypothesis_length'] = len(self._normalize_text(hypothesis))
        results['pythainlp_engine'] = pythainlp_engine

        # text normalization
        results['reference_normalized'] = self._normalize_text(reference)
        results['hypothesis_normalized'] = self._normalize_text(hypothesis)
        results['reference_words'] = results['reference_normalized'].split()
        results['hypothesis_words'] = results['hypothesis_normalized'].split()
        results['reference_word_count'] = len(results['reference_words'])
        results['hypothesis_word_count'] = len(results['hypothesis_words'])
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print formatted results with detailed explanations
        
        Args:
            results: Results dictionary from evaluate() method
        """
        print("=== OCR Evaluation Results ===")
        print(f"Reference length: {results['reference_length']} characters")
        print(f"Hypothesis length: {results['hypothesis_length']} characters")
        print()
        
        print("Error Rates:")
        print(f"  CER: {results['cer_percent']:.2f}%")
        print(f"  WER (standard): {results['wer_percent']:.2f}%", end="")
        if results['wer_percent'] > 100:
            print(" ⚠️  (>100% is possible when insertions > reference words)")
        else:
            print()
        print(f"  WER (pythainlp-{results['pythainlp_engine']}): {results['wer_pythainlp_percent']:.2f}%")
        print()
        
        print(f"Semantic Similarity: {results['semantic_similarity']:.4f}")
        print()
        
        ops = results['edit_operations']
        print("Edit Operations (Word-level):")
        print(f"  Hits: {ops['hits']}")
        print(f"  Substitutions: {ops['substitutions']}")
        print(f"  Insertions: {ops['insertions']}")
        print(f"  Deletions: {ops['deletions']}")
        
        total_ops = ops['substitutions'] + ops['insertions'] + ops['deletions']
        print(f"  Total errors: {total_ops}")
        
        # Add explanation for high WER
        if results['wer_percent'] > 100:
            print()
            print("📝 WER Explanation:")
            print("   Formula: (S+I+D) / Total_Reference_Words * 100")
            print(f"   = ({ops['substitutions']}+{ops['insertions']}+{ops['deletions']}) / ? * 100")
            print(f"   = {total_ops} / ? * 100 = {results['wer_percent']:.1f}%")
            print("   High WER often indicates severe tokenization differences")
    
    def debug_tokenization(self, reference: str, hypothesis: str, 
                          engine: str = 'newmm') -> None:
        """
        Debug tokenization differences to understand WER calculations
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            engine: pythainlp tokenization engine
        """
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        print("=== Tokenization Debug ===")
        print(f"Reference: '{ref}'")
        print(f"Hypothesis: '{hyp}'")
        print()
        
        # Standard word splitting (by spaces)
        ref_words = ref.split()
        hyp_words = hyp.split()
        print("Standard Word Splitting:")
        print(f"  Reference words: {ref_words} (count: {len(ref_words)})")
        print(f"  Hypothesis words: {hyp_words} (count: {len(hyp_words)})")
        print(f"  Standard WER: {self.calculate_wer(reference, hypothesis)*100:.1f}%")
        print()
        
        # pythainlp tokenization
        try:
            ref_tokens = word_tokenize(ref, engine=engine, keep_whitespace=False)
            hyp_tokens = word_tokenize(hyp, engine=engine, keep_whitespace=False)
            print(f"pythainlp ({engine}) Tokenization:")
            print(f"  Reference tokens: {ref_tokens} (count: {len(ref_tokens)})")
            print(f"  Hypothesis tokens: {hyp_tokens} (count: {len(hyp_tokens)})")
            print(f"  pythainlp WER: {self.calculate_wer_pythainlp(reference, hypothesis, engine)*100:.1f}%")
        except Exception as e:
            print(f"pythainlp tokenization error: {e}")
        
        print()
        print("💡 Tips:")
        print("  - WER can exceed 100% when (insertions + substitutions + deletions) > reference_word_count")
        print("  - Different tokenizers can give very different WER scores")
        print("  - For mixed language text, consider using CER as primary metric")


# Example usage
if __name__ == "__main__":
    # Initialize metrics calculator
    metrics = OCRMetrics()
    
    # Example 1: Text with whitespace issues
    print("=== Example 1: Whitespace Normalization ===")
    ref_whitespace = "สวัสดี ครับ ผม ชื่อ สมชาย"
    hyp_whitespace = "สวัสดี\nครับ\tผม   ชื่อ  สมชาย"  # same content, different whitespace
    
    results_whitespace = metrics.evaluate(ref_whitespace, hyp_whitespace)
    metrics.print_results(results_whitespace)
    print()
    
    # Example 2: Thai text
    print("=== Example 2: Thai Text ===")
    ref_thai = "สวัสดีครับ ผมชื่อสมชาย"
    hyp_thai = "สวัสดีครับ ผมชื่อสมชัย"
    
    results_thai = metrics.evaluate(ref_thai, hyp_thai)
    metrics.print_results(results_thai)
    print()
    
    # Example 2: Mixed Thai-English (High WER case)
    print("=== Example 2: Mixed Thai-English (High WER case) ===")
    ref_mixed = "tnnnewsข่าวค่ำ"
    hyp_mixed = "tnn news ข่าว"
    
    results_mixed = metrics.evaluate(ref_mixed, hyp_mixed)
    metrics.print_results(results_mixed)
    
    # Debug the high WER
    print()
    metrics.debug_tokenization(ref_mixed, hyp_mixed)
    print()
    
    # Example 3: English text
    print("=== Example 3: English Text ===")
    ref_eng = "Hello World this is a test"
    hyp_eng = "Hello Word this is test"
    
    results_eng = metrics.evaluate(ref_eng, hyp_eng)
    metrics.print_results(results_eng)