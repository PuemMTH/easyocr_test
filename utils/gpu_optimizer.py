"""
GPU optimization utilities for OCR processing
"""
import torch
import gc
import psutil
import os
from typing import Dict, Any, Optional
import numpy as np


class GPUOptimizer:
    """GPU optimization manager for OCR processing"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_info = self._get_gpu_info()
        self.optimal_settings = self._calculate_optimal_settings()
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        if not self.gpu_available:
            return {"available": False}
        
        gpu_info = {
            "available": True,
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
            "compute_capability": torch.cuda.get_device_capability(0),
            "driver_version": torch.version.cuda
        }
        
        return gpu_info
    
    def _calculate_optimal_settings(self) -> Dict[str, Any]:
        """Calculate optimal settings based on GPU capabilities"""
        if not self.gpu_available:
            return {
                "batch_size": 8,
                "max_workers": 2,
                "mixed_precision": False,
                "memory_fraction": 0.8
            }
        
        memory_gb = self.gpu_info["memory_total"]
        
        # Batch size based on GPU memory
        if memory_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
            batch_size = 64
            max_workers = 8
            mixed_precision = True
        elif memory_gb >= 12:  # Mid-range GPU (RTX 3080, 4070, etc.)
            batch_size = 32
            max_workers = 6
            mixed_precision = True
        elif memory_gb >= 8:  # Lower-end GPU (RTX 3060, 4060, etc.)
            batch_size = 16
            max_workers = 4
            mixed_precision = False
        else:  # Budget GPU
            batch_size = 8
            max_workers = 2
            mixed_precision = False
        
        return {
            "batch_size": batch_size,
            "max_workers": max_workers,
            "mixed_precision": mixed_precision,
            "memory_fraction": 0.8
        }
    
    def setup_gpu_environment(self):
        """Setup optimal GPU environment"""
        if not self.gpu_available:
            print("💻 CPU mode - no GPU optimizations applied")
            return
        
        # Set PyTorch optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.optimal_settings["memory_fraction"])
        
        print(f"🎮 GPU Environment Setup:")
        print(f"  📊 GPU: {self.gpu_info['name']}")
        print(f"  💾 Memory: {self.gpu_info['memory_total']:.1f}GB")
        print(f"  📦 Batch Size: {self.optimal_settings['batch_size']}")
        print(f"  🔧 Workers: {self.optimal_settings['max_workers']}")
        print(f"  ⚡ Mixed Precision: {self.optimal_settings['mixed_precision']}")
        print(f"  🎯 Memory Fraction: {self.optimal_settings['memory_fraction']}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU and system memory usage"""
        memory_info = {
            "system_ram_gb": psutil.virtual_memory().total / 1024**3,
            "system_ram_used_gb": psutil.virtual_memory().used / 1024**3,
            "system_ram_percent": psutil.virtual_memory().percent
        }
        
        if self.gpu_available:
            memory_info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_total_gb": self.gpu_info["memory_total"]
            })
        
        return memory_info
    
    def clear_gpu_cache(self):
        """Clear GPU cache and garbage collect"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def monitor_memory(self, stage: str = "Unknown"):
        """Monitor and log memory usage"""
        memory = self.get_memory_usage()
        
        print(f"📊 Memory Usage ({stage}):")
        print(f"  💻 System RAM: {memory['system_ram_used_gb']:.1f}/{memory['system_ram_gb']:.1f}GB ({memory['system_ram_percent']:.1f}%)")
        
        if self.gpu_available:
            gpu_used = memory['gpu_memory_allocated_gb']
            gpu_total = memory['gpu_memory_total_gb']
            gpu_percent = (gpu_used / gpu_total) * 100
            print(f"  🎮 GPU Memory: {gpu_used:.1f}/{gpu_total:.1f}GB ({gpu_percent:.1f}%)")
    
    def optimize_batch_size(self, current_batch_size: int, memory_threshold: float = 0.9) -> int:
        """Dynamically adjust batch size based on memory usage"""
        if not self.gpu_available:
            return current_batch_size
        
        memory = self.get_memory_usage()
        gpu_usage = memory['gpu_memory_allocated_gb'] / memory['gpu_memory_total_gb']
        
        if gpu_usage > memory_threshold:
            # Reduce batch size if memory usage is high
            new_batch_size = max(1, current_batch_size // 2)
            print(f"⚠️  High GPU memory usage ({gpu_usage:.1%}), reducing batch size: {current_batch_size} → {new_batch_size}")
            return new_batch_size
        elif gpu_usage < 0.5 and current_batch_size < self.optimal_settings["batch_size"]:
            # Increase batch size if memory usage is low
            new_batch_size = min(self.optimal_settings["batch_size"], current_batch_size * 2)
            print(f"✅ Low GPU memory usage ({gpu_usage:.1%}), increasing batch size: {current_batch_size} → {new_batch_size}")
            return new_batch_size
        
        return current_batch_size
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for current GPU"""
        return self.optimal_settings["batch_size"]
    
    def get_optimal_workers(self) -> int:
        """Get the optimal number of workers for current GPU"""
        return self.optimal_settings["max_workers"]


def create_gpu_optimizer() -> GPUOptimizer:
    """Create and configure GPU optimizer"""
    optimizer = GPUOptimizer()
    optimizer.setup_gpu_environment()
    return optimizer


def batch_process_with_memory_monitoring(optimizer: GPUOptimizer, process_func, data, 
                                        initial_batch_size: Optional[int] = None) -> list:
    """
    Process data in batches with memory monitoring and dynamic batch size adjustment
    
    Args:
        optimizer: GPUOptimizer instance
        process_func: Function to process each batch
        data: Data to process
        initial_batch_size: Initial batch size (will be auto-adjusted)
    
    Returns:
        List of processed results
    """
    if initial_batch_size is None:
        batch_size = optimizer.get_optimal_batch_size()
    else:
        batch_size = initial_batch_size
    
    results = []
    total_items = len(data)
    
    print(f"🚀 Starting batch processing with batch size: {batch_size}")
    
    for i in range(0, total_items, batch_size):
        batch = data[i:i + batch_size]
        
        # Monitor memory before processing
        optimizer.monitor_memory(f"Batch {i//batch_size + 1}")
        
        # Process batch
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        # Clear cache after each batch
        optimizer.clear_gpu_cache()
        
        # Adjust batch size based on memory usage
        new_batch_size = optimizer.optimize_batch_size(batch_size)
        if new_batch_size != batch_size:
            batch_size = new_batch_size
        
        print(f"✅ Processed batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}")
    
    return results 