"""
Unit and integration tests for Ablation Studies Runner (src/ablation_runner.py).
Tests fast dummy verification mode, model builds, config parsing, and CSV logging.
"""

import os
import sys
import tempfile
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ablation_runner import (
    get_ablation_configs,
    run_dummy_variant,
    save_results_to_csv,
    apply_augmentations,
    AblationConfig,
    CSV_FIELDNAMES,
)


def test_ablation_configs_resolution():
    """Verify CLI group resolvers return correct number of variants."""
    se_configs = get_ablation_configs("se")
    assert len(se_configs) == 4

    depth_configs = get_ablation_configs("depth")
    assert len(depth_configs) == 3

    aug_configs = get_ablation_configs("aug")
    assert len(aug_configs) == 4

    all_configs = get_ablation_configs("all")
    assert len(all_configs) == 11


def test_apply_augmentations_shape():
    """Verify stochastic augmentations preserve 1D audio length."""
    import numpy as np

    dummy_audio = np.random.randn(160000).astype(np.float32)
    aug_audio = apply_augmentations(dummy_audio, use_gain=True, use_noise=True, augment_prob=1.0)

    assert aug_audio.shape == dummy_audio.shape
    assert not np.isnan(aug_audio).any()


def test_dummy_variant_execution_cpu():
    """Verify run_dummy_variant runs on CPU, returns metrics, and has correct shapes."""
    cfg = AblationConfig(
        experiment_name="test_dummy_se_r16",
        group="se",
        variant_type="SE Ratio r=16",
        variant_name="Test SE-ResNet",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_gain=True,
        use_noise=True,
    )

    device = torch.device("cpu")
    res = run_dummy_variant(cfg, device=device, batch_size=2)

    assert res.experiment_name == "test_dummy_se_r16"
    assert res.parameter_count > 1000000
    assert res.val_rmse > 0.0
    assert res.val_mae > 0.0
    assert res.latency_ms > 0.0


def test_csv_output_serialization():
    """Verify save_results_to_csv writes all required fields to CSV."""
    cfg = AblationConfig(
        experiment_name="test_csv_variant",
        group="depth",
        variant_type="2 Stages",
        variant_name="Shallow Test",
        use_se=True,
        se_ratio=16,
        stages=2,
    )

    device = torch.device("cpu")
    res = run_dummy_variant(cfg, device=device, batch_size=2)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_results_to_csv([res], tmp_path)
        assert os.path.exists(tmp_path)

        with open(tmp_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            assert header == CSV_FIELDNAMES
            data_line = f.readline().strip().split(",")
            assert len(data_line) == len(CSV_FIELDNAMES)
            assert data_line[0] == "test_csv_variant"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
