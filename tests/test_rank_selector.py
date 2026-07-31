"""
Rank selector tests — pure logic, no external deps.

Tests the boundary conditions of the dataset-size → LoRA config mapping.
These boundaries matter because oversizing rank wastes VRAM (could OOM on
a 24GB pod), while undersizing loses fine-tuning quality.
"""

from worker.rank_selector import (
    select_lora_config,
    ATTENTION_ONLY,
    ATTENTION_FULL,
    ATTENTION_AND_MLP,
)


class TestRankSelector:
    """Boundary tests for select_lora_config()."""

    # ── Tier 1: <100 examples → r=8, attention only ──

    def test_small_dataset(self):
        cfg = select_lora_config(50)
        assert cfg.r == 8
        assert cfg.alpha == 16
        assert cfg.target_modules == ATTENTION_ONLY

    def test_boundary_99(self):
        """99 is the last value in the <100 tier."""
        cfg = select_lora_config(99)
        assert cfg.r == 8

    # ── Tier 2: 100–499 → r=16, full attention ──

    def test_boundary_100(self):
        """100 is the first value in the 100–499 tier."""
        cfg = select_lora_config(100)
        assert cfg.r == 16
        assert cfg.alpha == 32
        assert cfg.target_modules == ATTENTION_FULL

    def test_boundary_499(self):
        cfg = select_lora_config(499)
        assert cfg.r == 16

    # ── Tier 3: 500–1999 → r=32, full attention ──

    def test_boundary_500(self):
        cfg = select_lora_config(500)
        assert cfg.r == 32
        assert cfg.alpha == 64
        assert cfg.target_modules == ATTENTION_FULL

    def test_boundary_1999(self):
        cfg = select_lora_config(1999)
        assert cfg.r == 32

    # ── Tier 4: ≥2000 → r=64, attention + MLP ──

    def test_boundary_2000(self):
        cfg = select_lora_config(2000)
        assert cfg.r == 64
        assert cfg.alpha == 128
        assert cfg.target_modules == ATTENTION_AND_MLP

    def test_large_dataset(self):
        cfg = select_lora_config(10000)
        assert cfg.r == 64

    # ── Invariant: alpha = r × 2 across all tiers ──

    def test_alpha_is_double_rank(self):
        """Design invariant: alpha = r × 2 for all tiers."""
        for n in [20, 99, 100, 499, 500, 1999, 2000, 5000]:
            cfg = select_lora_config(n)
            assert cfg.alpha == cfg.r * 2, f"alpha != r*2 for num_examples={n}"
