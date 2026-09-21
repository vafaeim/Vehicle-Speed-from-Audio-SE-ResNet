import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import time
from typing import List, Dict, Any
from src.models_torch import build_se_resnet
from src.config import Config

def measure_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int = 10, rep: int = 100) -> List[float]:
    """Measures inference latency for a given input tensor, returning a list of times in milliseconds."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
        
    times = []
    with torch.no_grad():
        for _ in range(rep):
            start = time.perf_counter()
            _ = model(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
            
    return times


def run_benchmark(device_str: str, batch_sizes: List[int], use_amp: bool = True) -> List[Dict[str, Any]]:
    device = torch.device(device_str)
    print(f"\n{'='*50}")
    print(f"Starting Benchmark on Device: {device.type.upper()} (AMP: {use_amp})")
    print(f"{'='*50}")
    
    # Initialize a single model
    single_model = build_se_resnet(
        input_shape=(1, Config.N_MELS, 313),
        se_ratio=Config.SE_RATIO,
        dropout=Config.DROPOUT_RATE,
        stages=3
    ).to(device)
    single_model.eval()

    # Initialize ensemble (10 models)
    ensemble = [
        build_se_resnet(
            input_shape=(1, Config.N_MELS, 313),
            se_ratio=Config.SE_RATIO,
            dropout=Config.DROPOUT_RATE,
            stages=3
        ).to(device).eval()
        for _ in range(10)
    ]
    
    results = []
    
    for b in batch_sizes:
        print(f"\n--- Benchmarking Batch Size: {b} ---")
        # Dummy data matching the Mel-Spectrogram shape
        x = torch.randn(b, 1, Config.N_MELS, 313, device=device)
        
        # 1. Single Model Benchmark
        print("  -> Single Model...")
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            times_single = measure_latency(single_model, x, warmup=20, rep=100)
            
        mean_latency = np.mean(times_single)
        p95_latency = np.percentile(times_single, 95)
        p99_latency = np.percentile(times_single, 99)
        throughput = (b * 1000.0) / mean_latency  # samples per second
        
        res_single = {
            "Device": device.type.upper(),
            "Mode": "Single Model",
            "Batch Size": b,
            "AMP": use_amp,
            "Mean Latency (ms)": round(mean_latency, 2),
            "P95 Latency (ms)": round(p95_latency, 2),
            "P99 Latency (ms)": round(p99_latency, 2),
            "Throughput (samples/s)": round(throughput, 2)
        }
        results.append(res_single)
        print(f"     Mean: {mean_latency:.2f}ms | P95: {p95_latency:.2f}ms | Throughput: {throughput:.2f} samples/s")

        # 2. Ensemble Benchmark
        print("  -> 10-Fold Ensemble...")
        # Custom ensemble forward wrapper
        class EnsembleWrapper(torch.nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = torch.nn.ModuleList(models)
            def forward(self, x):
                return torch.mean(torch.stack([m(x) for m in self.models]), dim=0)
                
        ens_wrapper = EnsembleWrapper(ensemble).to(device).eval()
        
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            times_ens = measure_latency(ens_wrapper, x, warmup=20, rep=100)
            
        mean_latency_ens = np.mean(times_ens)
        p95_latency_ens = np.percentile(times_ens, 95)
        p99_latency_ens = np.percentile(times_ens, 99)
        throughput_ens = (b * 1000.0) / mean_latency_ens
        
        res_ens = {
            "Device": device.type.upper(),
            "Mode": "10-Fold Ensemble",
            "Batch Size": b,
            "AMP": use_amp,
            "Mean Latency (ms)": round(mean_latency_ens, 2),
            "P95 Latency (ms)": round(p95_latency_ens, 2),
            "P99 Latency (ms)": round(p99_latency_ens, 2),
            "Throughput (samples/s)": round(throughput_ens, 2)
        }
        results.append(res_ens)
        print(f"     Mean: {mean_latency_ens:.2f}ms | P95: {p95_latency_ens:.2f}ms | Throughput: {throughput_ens:.2f} samples/s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    all_results = []
    
    # 1. CPU Benchmarks (Batch sizes 1 and 32)
    # CPU doesn't typically benefit from AMP for this model scale, so we run FP32
    all_results.extend(run_benchmark("cpu", batch_sizes=[1, 32], use_amp=False))
    
    # 2. CUDA Benchmarks (if available)
    if torch.cuda.is_available():
        all_results.extend(run_benchmark("cuda", batch_sizes=[1, 32], use_amp=True))
    else:
        print("\n[WARN] CUDA not available. Skipping GPU benchmarks.")
        
    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Print Markdown Table
    print("\n\n" + "="*80)
    print(" FINAL DEPLOYMENT BENCHMARK SUMMARY")
    print("="*80)
    print("| Device | Mode             | Batch | AMP | Mean (ms) | P95 (ms) | Throughput (S/s) |")
    print("|--------|------------------|-------|-----|-----------|----------|------------------|")
    for r in all_results:
        print(f"| {r['Device']:<6} | {r['Mode']:<16} | {r['Batch Size']:<5} | {str(r['AMP']):<3} | {r['Mean Latency (ms)']:>9.2f} | {r['P95 Latency (ms)']:>8.2f} | {r['Throughput (samples/s)']:>16.2f} |")
    print("="*80 + "\n")
    print(f"Full benchmark data saved to {args.output}")

if __name__ == "__main__":
    from src.utils import set_seed
    set_seed(42)
    main()
