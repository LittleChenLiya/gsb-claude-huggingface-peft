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

"""Tests for the transactional multi-adapter activation entry point `PeftModel.set_adapters`.

Runs on CPU with a tiny linear layer and two LoRA adapters.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)


def _build_model():
    torch.manual_seed(0)
    base_model = TinyModel()
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["lin"], init_lora_weights=False)
    model = get_peft_model(base_model, config, adapter_name="adapter_a")
    model.add_adapter("adapter_b", config)
    return model


def _requires_grad_snapshot(model):
    return {name: param.requires_grad for name, param in model.named_parameters()}


def _lora_requires_grad(model, adapter_name):
    return {
        name: param.requires_grad
        for name, param in model.named_parameters()
        if "lora_" in name and f".{adapter_name}." in name
    }


class TestSetAdaptersTransaction:
    def test_successful_switch_single_adapter(self):
        model = _build_model()
        model.set_adapters("adapter_b")
        assert model.active_adapters == ["adapter_b"]
        # weights of the active adapter are trainable again
        assert all(_lora_requires_grad(model, "adapter_b").values())

    def test_successful_switch_multiple_adapters(self):
        model = _build_model()
        model.set_adapters(["adapter_a", "adapter_b"])
        assert model.active_adapters == ["adapter_a", "adapter_b"]
        assert all(_lora_requires_grad(model, "adapter_a").values())
        assert all(_lora_requires_grad(model, "adapter_b").values())
        # forward pass works with both adapters active
        out = model(torch.randn(2, 8))
        assert out.shape == (2, 8)

    def test_inference_mode_freezes_activated_adapters(self):
        model = _build_model()
        model.set_adapters(["adapter_a", "adapter_b"], inference_mode=True)
        assert model.active_adapters == ["adapter_a", "adapter_b"]
        assert not any(_lora_requires_grad(model, "adapter_a").values())
        assert not any(_lora_requires_grad(model, "adapter_b").values())

    def test_invalid_name_keeps_previous_active_set(self):
        model = _build_model()
        model.set_adapter("adapter_b")
        snapshot_before = _requires_grad_snapshot(model)

        with pytest.raises(ValueError, match="does_not_exist"):
            model.set_adapters(["adapter_a", "does_not_exist"])

        # the active adapter set is unchanged
        assert model.active_adapters == ["adapter_b"]
        # requires_grad flags are not partially switched
        assert _requires_grad_snapshot(model) == snapshot_before

    def test_invalid_single_name_keeps_previous_state(self):
        model = _build_model()
        snapshot_before = _requires_grad_snapshot(model)

        with pytest.raises(ValueError, match="does_not_exist"):
            model.set_adapters("does_not_exist")

        assert model.active_adapters == ["adapter_a"]
        assert _requires_grad_snapshot(model) == snapshot_before

    def test_all_invalid_names_keeps_previous_state(self):
        model = _build_model()
        snapshot_before = _requires_grad_snapshot(model)

        with pytest.raises(ValueError, match="nope"):
            model.set_adapters(["nope", "also_nope"])

        assert model.active_adapters == ["adapter_a"]
        assert _requires_grad_snapshot(model) == snapshot_before

    def test_legacy_set_adapter_interface_unchanged(self):
        model = _build_model()
        # old single-adapter API still works and raises on unknown names
        model.set_adapter("adapter_b")
        assert model.active_adapter == "adapter_b"
        assert model.active_adapters == ["adapter_b"]
        with pytest.raises(ValueError, match="does_not_exist"):
            model.set_adapter("does_not_exist")
        assert model.active_adapters == ["adapter_b"]

    def test_switching_changes_output_and_back(self):
        model = _build_model().eval()
        x = torch.randn(2, 8)
        model.set_adapter("adapter_a")
        out_a = model(x)
        model.set_adapters(["adapter_b"])
        out_b = model(x)
        assert not torch.allclose(out_a, out_b)
        # switching back restores the original output
        model.set_adapters(["adapter_a"])
        assert torch.allclose(model(x), out_a)
