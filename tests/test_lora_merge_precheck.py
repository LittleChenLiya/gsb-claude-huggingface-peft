# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the LoRA merge error pre-check (`precheck_merge_error` / `merge_error_tolerance`).

All tests run on CPU with small matrices, covering fp32 and low-precision (fp16/bf16) base weights.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig, MergePrecheckError, get_peft_model


IN_FEATURES = 8
OUT_FEATURES = 4
R = 2
LORA_ALPHA = 4  # scaling = alpha / r = 2


class TinyModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).to(dtype)

    def forward(self, x):
        return self.lin(x)


def _build_lora_layer(dtype, seed=0):
    """Build a tiny LoRA-wrapped Linear layer with deterministic, non-trivial adapter weights."""
    torch.manual_seed(seed)
    model = TinyModel(dtype)
    config = LoraConfig(target_modules=["lin"], r=R, lora_alpha=LORA_ALPHA, lora_dropout=0.0)
    model = get_peft_model(model, config)
    layer = model.base_model.model.lin
    # make the adapter update non-trivial and deterministic (lora_B is zero-initialized by default)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.randn(R, IN_FEATURES))
        layer.lora_B["default"].weight.copy_(torch.randn(OUT_FEATURES, R))
    return layer


def _expected_merge_error(layer, dtype):
    """Manually computed max abs error of the dtype round-trip of the merged weights."""
    base = layer.get_base_layer().weight.detach().to(torch.float32)
    delta = layer.get_delta_weight("default").to(torch.float32)
    merged_fp32 = base + delta
    return (merged_fp32.to(dtype).to(torch.float32) - merged_fp32).abs().max().item()


class TestPrecheckMergeError:
    def test_fp32_error_is_zero_and_weights_untouched(self):
        layer = _build_lora_layer(torch.float32)
        base_before = layer.get_base_layer().weight.detach().clone()
        lora_A_before = layer.lora_A["default"].weight.detach().clone()
        lora_B_before = layer.lora_B["default"].weight.detach().clone()

        errors = layer.precheck_merge_error()

        # fp32 round-trip is exact
        assert errors == {"default": 0.0}
        # the pre-check must not modify anything
        assert torch.equal(layer.get_base_layer().weight, base_before)
        assert torch.equal(layer.lora_A["default"].weight, lora_A_before)
        assert torch.equal(layer.lora_B["default"].weight, lora_B_before)
        assert not layer.merged
        assert layer.merged_adapters == []

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_error_matches_manual_estimate(self, dtype):
        layer = _build_lora_layer(dtype)
        base_before = layer.get_base_layer().weight.detach().clone()

        errors = layer.precheck_merge_error()

        expected = _expected_merge_error(layer, dtype)
        assert expected > 0.0  # sanity: rounding really occurs for these weights
        assert errors["default"] == pytest.approx(expected)
        # the pre-check must not modify the base weight
        assert torch.equal(layer.get_base_layer().weight, base_before)
        assert not layer.merged

    def test_target_dtype_override_on_fp32_base(self):
        layer = _build_lora_layer(torch.float32)
        errors = layer.precheck_merge_error(target_dtype=torch.float16)
        expected = _expected_merge_error(layer, torch.float16)
        assert expected > 0.0
        assert errors["default"] == pytest.approx(expected)


class TestMergeErrorTolerance:
    def test_fp32_merge_accepted_with_zero_tolerance(self):
        layer = _build_lora_layer(torch.float32)
        base_before = layer.get_base_layer().weight.detach().clone()
        expected_merged = base_before + layer.get_delta_weight("default").to(torch.float32)

        # fp32 round-trip error is exactly 0, so even a zero tolerance accepts the merge
        layer.merge(safe_merge=True, merge_error_tolerance=0.0)

        assert layer.merged
        assert torch.allclose(layer.get_base_layer().weight, expected_merged, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_merge_rejected_over_tolerance_and_stays_reversible(self, dtype):
        layer = _build_lora_layer(dtype)
        base_before = layer.get_base_layer().weight.detach().clone()
        estimated_error = layer.precheck_merge_error()["default"]
        assert estimated_error > 0.0

        # tolerance below the estimated error -> merge must be rejected
        with pytest.raises(MergePrecheckError, match="merge error"):
            layer.merge(merge_error_tolerance=estimated_error / 2)

        # the rejection happens before any modification: weights and merge state are untouched
        assert torch.equal(layer.get_base_layer().weight, base_before)
        assert not layer.merged
        assert layer.merged_adapters == []

        # the layer is still in a reversible state: merging without a tolerance works ...
        layer.merge(safe_merge=True)
        assert layer.merged
        expected_merged = base_before + layer.get_delta_weight("default").to(dtype)
        assert torch.equal(layer.get_base_layer().weight, expected_merged)

        # ... and unmerging restores the original base weights up to low-precision rounding
        layer.unmerge()
        assert not layer.merged
        atol = 8 * torch.finfo(dtype).eps * base_before.abs().max().item()
        assert torch.allclose(layer.get_base_layer().weight, base_before, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_merge_accepted_within_tolerance(self, dtype):
        layer = _build_lora_layer(dtype)
        base_before = layer.get_base_layer().weight.detach().clone()
        estimated_error = layer.precheck_merge_error()["default"]

        # tolerance above the estimated error -> merge goes through
        layer.merge(safe_merge=True, merge_error_tolerance=estimated_error * 2)

        assert layer.merged
        expected_merged = base_before + layer.get_delta_weight("default").to(dtype)
        assert torch.equal(layer.get_base_layer().weight, expected_merged)

    def test_tolerance_checked_for_all_adapters_before_merging(self):
        # with two adapters, a failing pre-check for any adapter must leave *all* base weights untouched
        layer = _build_lora_layer(torch.float16)
        base_before = layer.get_base_layer().weight.detach().clone()

        torch.manual_seed(123)
        layer.update_layer(
            "other",
            r=R,
            lora_alpha=LORA_ALPHA,
            config=LoraConfig(target_modules=["lin"], r=R, lora_alpha=LORA_ALPHA),
        )
        with torch.no_grad():
            layer.lora_A["other"].weight.copy_(torch.randn(R, IN_FEATURES))
            layer.lora_B["other"].weight.copy_(torch.randn(OUT_FEATURES, R))

        errors = layer.precheck_merge_error(adapter_names=["default", "other"])
        assert set(errors) == {"default", "other"}
        min_error = min(errors.values())

        with pytest.raises(MergePrecheckError):
            layer.merge(
                adapter_names=["default", "other"],
                merge_error_tolerance=min_error / 2,
            )

        assert torch.equal(layer.get_base_layer().weight, base_before)
        assert not layer.merged
        assert layer.merged_adapters == []
