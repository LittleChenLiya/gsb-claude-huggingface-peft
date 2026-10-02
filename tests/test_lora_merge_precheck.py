# Copyright 2025-present the HuggingFace Inc. team.
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

"""
Tests for the LoRA merge error pre-check (`LoraLayer.merge_error_precheck` and the `max_merge_error` argument of
`merge`).

The pre-check estimates the maximum absolute difference that merging the LoRA weights would introduce due to
low-precision (e.g. float16/bfloat16) or quantized storage *before* the merge is carried out. It is purely
diagnostic and read-only: it never modifies the base weights and never changes the merged/unmerged state of the
layer. If a threshold is given and exceeded, the merge is rejected without modifying any weights, keeping the layer
unmerged and thus reversible.

All tests run on CPU with small matrices.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear


def make_lora_linear(base_dtype, lora_dtype, r=2, alpha=4, seed=0):
    """Create a small LoRA-wrapped Linear layer on CPU with non-zero adapter weights."""
    torch.manual_seed(seed)
    base = nn.Linear(8, 6, bias=True).to(base_dtype)
    with torch.no_grad():
        base.weight.copy_(torch.randn(6, 8, dtype=torch.float32).to(base_dtype) * 0.5)
        base.bias.copy_(torch.randn(6, dtype=torch.float32).to(base_dtype) * 0.1)

    config = LoraConfig(r=r, lora_alpha=alpha, init_lora_weights=True)
    layer = LoraLinear(base, "default", config, r=r, lora_alpha=alpha)
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.randn(r, 8) * 0.3)
        layer.lora_B["default"].weight.copy_(torch.randn(6, r) * 0.3)

    if lora_dtype != torch.float32:
        layer.lora_A["default"].weight.data = layer.lora_A["default"].weight.data.to(lora_dtype)
        layer.lora_B["default"].weight.data = layer.lora_B["default"].weight.data.to(lora_dtype)
    return layer


class TestMergeErrorPrecheck:
    def test_fp32_error_is_zero(self):
        # fp32 base + fp32 adapter: the exact and the rounded merge are identical -> zero error
        layer = make_lora_linear(torch.float32, torch.float32)
        result = layer.merge_error_precheck()
        assert set(result.keys()) == {"default"}
        info = result["default"]
        assert info["max_abs_diff"] == 0.0
        assert info["mean_abs_diff"] == 0.0
        assert info["base_dtype"] == torch.float32
        assert info["delta_dtype"] == torch.float32
        assert info["threshold"] is None
        assert info["exceeded"] is False

    @pytest.mark.parametrize("base_dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_error_positive(self, base_dtype):
        # low-precision base + adapter (auto-cast to base dtype): casting the delta to the base dtype introduces
        # rounding
        layer = make_lora_linear(base_dtype, base_dtype)
        info = layer.merge_error_precheck()["default"]
        assert info["max_abs_diff"] > 0.0
        assert info["mean_abs_diff"] > 0.0
        assert info["base_dtype"] == base_dtype
        assert info["delta_dtype"] == base_dtype

    def test_bf16_error_larger_than_fp16(self):
        # bfloat16 has a coarser mantissa than float16, so the rounding error is typically larger
        fp16_info = make_lora_linear(torch.float16, torch.float16).merge_error_precheck()["default"]
        bf16_info = make_lora_linear(torch.bfloat16, torch.bfloat16).merge_error_precheck()["default"]
        assert bf16_info["max_abs_diff"] >= fp16_info["max_abs_diff"]

    def test_precheck_does_not_modify_weights(self):
        layer = make_lora_linear(torch.float16, torch.float32)
        weight_before = layer.get_base_layer().weight.data.clone()
        bias_before = layer.get_base_layer().bias.data.clone()
        delta_before = layer.get_delta_weight("default").clone()

        layer.merge_error_precheck()

        # base weights, bias and adapter weights must be untouched
        assert torch.equal(weight_before, layer.get_base_layer().weight.data)
        assert torch.equal(bias_before, layer.get_base_layer().bias.data)
        assert torch.equal(delta_before, layer.get_delta_weight("default"))
        # the layer must not be merged (reversible state)
        assert not layer.merged
        assert layer.merged_adapters == []

    def test_precheck_with_threshold_returns_threshold(self):
        layer = make_lora_linear(torch.float32, torch.float32)
        result = layer.merge_error_precheck(max_merge_error=1e-3)["default"]
        assert result["threshold"] == 1e-3
        assert result["exceeded"] is False

    def test_merge_rejected_over_threshold_keeps_reversible_state(self):
        layer = make_lora_linear(torch.float16, torch.float32)
        weight_before = layer.get_base_layer().weight.data.clone()
        bias_before = layer.get_base_layer().bias.data.clone()
        delta_before = layer.get_delta_weight("default").clone()
        error = layer.merge_error_precheck()["default"]["max_abs_diff"]

        # a threshold smaller than the estimated error must reject the merge
        with pytest.raises(ValueError, match="merge error pre-check failed"):
            layer.merge(safe_merge=True, max_merge_error=error / 10)

        # base weights and adapter weights are completely untouched
        assert torch.equal(weight_before, layer.get_base_layer().weight.data)
        assert torch.equal(bias_before, layer.get_base_layer().bias.data)
        assert torch.equal(delta_before, layer.get_delta_weight("default"))
        # the layer stays unmerged -> the adapter is still applied and can be merged later (reversible)
        assert not layer.merged
        assert layer.merged_adapters == []
        assert "default" in layer.lora_A and "default" in layer.lora_B

        # unmerge is a no-op on an unmerged layer and must not change anything
        layer.unmerge()
        assert not layer.merged
        assert torch.equal(weight_before, layer.get_base_layer().weight.data)

        # ... and merging with a generous threshold afterwards still works (reversibility)
        layer.merge(safe_merge=True, max_merge_error=error * 100)
        assert layer.merged
        assert layer.merged_adapters == ["default"]

    def test_merge_accepted_under_threshold_fp32(self):
        layer = make_lora_linear(torch.float32, torch.float32)
        weight_before = layer.get_base_layer().weight.data.clone()
        delta = layer.get_delta_weight("default")

        layer.merge(safe_merge=True, max_merge_error=1e-2)

        assert layer.merged
        assert layer.merged_adapters == ["default"]
        expected = weight_before + delta
        assert torch.equal(expected, layer.get_base_layer().weight.data)

        # unmerge restores the base weights (within floating-point round-trip tolerance)
        layer.unmerge()
        assert not layer.merged
        assert torch.allclose(weight_before, layer.get_base_layer().weight.data, atol=1e-6)

    def test_merge_accepted_under_threshold_fp16(self):
        layer = make_lora_linear(torch.float16, torch.float32)
        weight_before = layer.get_base_layer().weight.data.clone()
        delta = layer.get_delta_weight("default")
        error = layer.merge_error_precheck()["default"]["max_abs_diff"]

        # a threshold larger than the estimated error accepts the merge
        layer.merge(safe_merge=True, max_merge_error=error * 100)

        assert layer.merged
        assert layer.merged_adapters == ["default"]
        # safe_merge casts the delta to the base dtype before adding, emulating the real merge
        expected = weight_before + delta.to(torch.float16)
        assert torch.equal(expected, layer.get_base_layer().weight.data)

        # unmerge restores the base weights (within fp16 round-trip tolerance)
        layer.unmerge()
        assert not layer.merged
        assert torch.allclose(weight_before, layer.get_base_layer().weight.data, atol=1e-3)

    def test_threshold_boundary_is_not_exceeded(self):
        # threshold exactly at the estimated error must not reject (strictly greater comparison)
        layer = make_lora_linear(torch.float16, torch.float32)
        error = layer.merge_error_precheck()["default"]["max_abs_diff"]
        layer.merge(safe_merge=True, max_merge_error=error)
        assert layer.merged

    def test_merge_without_max_merge_error_is_unchanged(self):
        # not passing max_merge_error must preserve the original merge behavior
        layer = make_lora_linear(torch.float16, torch.float32)
        weight_before = layer.get_base_layer().weight.data.clone()
        delta = layer.get_delta_weight("default")
        layer.merge(safe_merge=True)
        assert layer.merged
        expected = weight_before + delta.to(torch.float16)
        assert torch.equal(expected, layer.get_base_layer().weight.data)

    def test_precheck_multiple_adapters(self):
        # two adapters: the pre-check reports both and rejects if either exceeds the threshold
        torch.manual_seed(1)
        base = nn.Linear(8, 6, bias=False).to(torch.float16)
        with torch.no_grad():
            base.weight.copy_(torch.randn(6, 8, dtype=torch.float32).to(torch.float16) * 0.5)
        config = LoraConfig(r=2, lora_alpha=4, init_lora_weights=True)
        layer = LoraLinear(base, "default", config, r=2, lora_alpha=4)
        layer.update_layer("second", r=2, lora_alpha=4, config=config)
        with torch.no_grad():
            layer.lora_A["default"].weight.copy_(torch.randn(2, 8) * 0.3)
            layer.lora_B["default"].weight.copy_(torch.randn(6, 2) * 0.3)
            layer.lora_A["second"].weight.copy_(torch.randn(2, 8) * 0.3)
            layer.lora_B["second"].weight.copy_(torch.randn(6, 2) * 0.3)

        result = layer.merge_error_precheck(adapter_names=["default", "second"])
        assert set(result.keys()) == {"default", "second"}
        assert all(info["max_abs_diff"] > 0.0 for info in result.values())

        # a tiny threshold rejects the merge of both adapters without modifying anything
        weight_before = layer.get_base_layer().weight.data.clone()
        with pytest.raises(ValueError, match="merge error pre-check failed"):
            layer.merge(safe_merge=True, adapter_names=["default", "second"], max_merge_error=1e-7)
        assert torch.equal(weight_before, layer.get_base_layer().weight.data)
        assert not layer.merged
