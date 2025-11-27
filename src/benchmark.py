"""
Hardware latency profiler for the SE-ResNet inference pipeline.

Benchmarks single-model and 10-fold ensemble throughput across CPU (FP32)
and CUDA (AMP) at batch sizes 1 and 32. When PyTorch >= 2.0 and CUDA are
available, also profiles torch.compile (AOT kernel fusion via Triton).

Usage:
    python -m src.benchmark [--output benchmark_results.json]
"""

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
    """Measures per-inference latency over `rep` repetitions, returning times in milliseconds."""
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

    single_model = build_se_resnet(
        input_shape=(1, Config.N_MELS, 313),
        se_ratio=Config.SE_RATIO,
        dropout=Config.DROPOUT_RATE,
        stages=3
    ).to(device)
    single_model.eval()

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
        x = torch.randn(b, 1, Config.N_MELS, 313, device=device)

        print("  -> Single Model...")
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            times_single = measure_latency(single_model, x, warmup=20, rep=100)

        mean_latency = np.mean(times_single)
        p95_latency = np.percentile(times_single, 95)
        p99_latency = np.percentile(times_single, 99)
        throughput = (b * 1000.0) / mean_latency

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

        print("  -> 10-Fold Ensemble...")

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

        if device.type == "cuda" and hasattr(torch, "compile"):
            print("  -> Torch.Compiled Single Model (Kernel Fusion & Triton)...")
            try:
                compiled_model = torch.compile(single_model)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    # Extended warmup required for AOT trace compilation
                    times_comp = measure_latency(compiled_model, x, warmup=30, rep=100)
                mean_comp = np.mean(times_comp)
                p95_comp = np.percentile(times_comp, 95)
                p99_comp = np.percentile(times_comp, 99)
                throughput_comp = (b * 1000.0) / mean_comp

                res_comp = {
                    "Device": "CUDA-COMPILED",
                    "Mode": "Single Model",
                    "Batch Size": b,
                    "AMP": use_amp,
                    "Mean Latency (ms)": round(mean_comp, 2),
                    "P95 Latency (ms)": round(p95_comp, 2),
                    "P99 Latency (ms)": round(p99_comp, 2),
                    "Throughput (samples/s)": round(throughput_comp, 2)
                }
                results.append(res_comp)
                print(f"     Mean: {mean_comp:.2f}ms | P95: {p95_comp:.2f}ms | Throughput: {throughput_comp:.2f} samples/s")
            except Exception as e:
                print(f"     [WARN] Torch Compile failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    all_results = []

    all_results.extend(run_benchmark("cpu", batch_sizes=[1, 32], use_amp=False))

    if torch.cuda.is_available():
        all_results.extend(run_benchmark("cuda", batch_sizes=[1, 32], use_amp=True))
    else:
        print("\n[WARN] CUDA not available. Skipping GPU benchmarks.")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)

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
