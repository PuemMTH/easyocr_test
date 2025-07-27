"""
GPU configuration settings for optimal OCR performance
"""
import os
import torch
from typing import Dict, Any


def setup_gpu_environment():
    """Setup optimal GPU environment variables and settings"""
    
    # Set environment variables for better GPU performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable CUDA cache
    os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'  # Set cache path
    
    # Set PyTorch optimizations
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for fixed input sizes
        torch.backends.cudnn.benchmark = True
        
        # Disable deterministic mode for better performance
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("✅ GPU environment optimized")
    else:
        print("⚠️  No GPU available, using CPU optimizations")


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
        "compute_capability": torch.cuda.get_device_capability(),
        "driver_version": torch.version.cuda
    }
    
    return gpu_info


def optimize_batch_size_for_memory(memory_gb: float) -> int:
    """Calculate optimal batch size based on GPU memory"""
    if memory_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
        return 64
    elif memory_gb >= 16:  # Mid-high GPU (RTX 4080, 3090, etc.)
        return 48
    elif memory_gb >= 12:  # Mid-range GPU (RTX 3080, 4070, etc.)
        return 32
    elif memory_gb >= 8:  # Lower-end GPU (RTX 3060, 4060, etc.)
        return 16
    elif memory_gb >= 6:  # Budget GPU (GTX 1660, etc.)
        return 8
    else:  # Very low memory GPU
        return 4


def get_optimal_workers_for_gpu(memory_gb: float) -> int:
    """Calculate optimal number of workers based on GPU memory"""
    if memory_gb >= 24:
        return 8
    elif memory_gb >= 16:
        return 6
    elif memory_gb >= 12:
        return 6
    elif memory_gb >= 8:
        return 4
    elif memory_gb >= 6:
        return 2
    else:
        return 1


def print_gpu_status():
    """Print current GPU status and recommendations"""
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info["available"]:
        print("❌ No GPU detected")
        print("💡 Recommendations:")
        print("  - Use CPU mode with smaller batch sizes")
        print("  - Consider using cloud GPU instances")
        return
    
    print("🎮 GPU Status:")
    print(f"  📊 Device: {gpu_info['device_name']}")
    print(f"  💾 Total Memory: {gpu_info['memory_total']:.1f}GB")
    print(f"  📦 Allocated: {gpu_info['memory_allocated']:.1f}GB")
    print(f"  🔧 Reserved: {gpu_info['memory_reserved']:.1f}GB")
    print(f"  ⚡ Compute Capability: {gpu_info['compute_capability']}")
    
    # Calculate recommendations
    optimal_batch = optimize_batch_size_for_memory(gpu_info['memory_total'])
    optimal_workers = get_optimal_workers_for_gpu(gpu_info['memory_total'])
    
    print(f"\n💡 Recommendations:")
    print(f"  📦 Optimal Batch Size: {optimal_batch}")
    print(f"  🔧 Optimal Workers: {optimal_workers}")
    
    # Memory usage percentage
    usage_percent = (gpu_info['memory_allocated'] / gpu_info['memory_total']) * 100
    if usage_percent > 80:
        print(f"  ⚠️  High memory usage: {usage_percent:.1f}%")
        print("     Consider reducing batch size or clearing cache")
    elif usage_percent > 50:
        print(f"  ✅ Moderate memory usage: {usage_percent:.1f}%")
    else:
        print(f"  ✅ Low memory usage: {usage_percent:.1f}%")
        print("     Consider increasing batch size for better performance")


def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cleared")


def set_mixed_precision(enabled: bool = True):
    """Enable or disable mixed precision training"""
    if enabled and torch.cuda.is_available():
        # Enable automatic mixed precision
        os.environ['TORCH_AMP_ENABLED'] = '1'
        print("⚡ Mixed precision enabled")
    else:
        os.environ['TORCH_AMP_ENABLED'] = '0'
        print("🔧 Mixed precision disabled")


def configure_for_inference():
    """Configure GPU for optimal inference performance"""
    if torch.cuda.is_available():
        # Set inference optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction for inference
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("🚀 GPU configured for optimal inference")
    else:
        print("💻 CPU inference mode") 