# Copyright 2023-present the HuggingFace Inc. team.
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
"""Tests for the adapter state dict load report and strict-mode loading.

These tests use small, locally-defined models (no network access) to exercise both the happy path and the failure
paths: missing keys, unexpected keys, and shape mismatches. The strict mode must fail without writing partial
weights, while the non-strict mode preserves the existing tolerant behavior.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from peft.utils import (
    get_adapter_state_dict_load_report,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class MLP(nn.Module):
    """A small feed-forward model with two linear layers."""

    def __init__(self, in_features=8, hidden_features=8, out_features=8):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden_features)
        self.lin1 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


def _make_peft_model(seed=0, target_modules=None, modules_to_save=None):
    torch.manual_seed(seed)
    model = MLP()
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=target_modules or ["lin0", "lin1"],
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, config)
    return peft_model


def _make_injected_model(seed=0, target_modules=None):
    torch.manual_seed(seed)
    model = MLP()
    config = LoraConfig(r=2, lora_alpha=4, target_modules=target_modules or ["lin0", "lin1"])
    injected = inject_adapter_in_model(config, model)
    return injected


class TestAdapterStateDictLoadReport:
    """Tests for get_adapter_state_dict_load_report."""

    def test_clean_state_dict_report_is_empty(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

    def test_missing_key_is_reported(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict = {k: v for k, v in state_dict.items() if "lin0.lora_B" not in k}
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert len(report.missing_keys) == 1
        assert "lin0.lora_B" in report.missing_keys[0]
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

    def test_unexpected_key_is_reported(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict["base_model.model.lin0.lora_C.weight"] = torch.randn(2, 8)
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert report.missing_keys == []
        assert report.unexpected_keys == ["base_model.model.lin0.lora_C.weight"]
        assert report.mismatched_keys == []

    def test_shape_mismatch_is_reported(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict["base_model.model.lin0.lora_A.weight"] = torch.randn(3, 8)
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert len(report.mismatched_keys) == 1
        key, checkpoint_shape, model_shape = report.mismatched_keys[0]
        assert key == "base_model.model.lin0.lora_A.default.weight"
        assert checkpoint_shape == (3, 8)
        assert model_shape == (2, 8)

    def test_injected_model_report(self):
        model = _make_injected_model()
        state_dict = get_peft_model_state_dict(model)
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []
        state_dict_missing = {k: v for k, v in state_dict.items() if "lin0.lora_B" not in k}
        report = get_adapter_state_dict_load_report(model, state_dict_missing)
        assert len(report.missing_keys) == 1
        assert report.unexpected_keys == []

    def test_modules_to_save_report_is_clean(self):
        model = _make_peft_model(modules_to_save=["lin1"])
        state_dict = get_peft_model_state_dict(model)
        report = get_adapter_state_dict_load_report(model, state_dict)
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

    def test_multiple_adapters_report_filters_by_adapter_name(self):
        model = _make_peft_model(target_modules=["lin0"])
        model.add_adapter("other", LoraConfig(r=2, lora_alpha=4, target_modules=["lin1"]))
        state_dict = get_peft_model_state_dict(model, adapter_name="default")
        report = get_adapter_state_dict_load_report(model, state_dict, adapter_name="default")
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []


class TestStrictMode:
    """Tests for set_peft_model_state_dict(strict=True)."""

    def test_strict_clean_state_dict_succeeds(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        load_result = set_peft_model_state_dict(model, state_dict, strict=True)
        assert load_result.unexpected_keys == []
        assert load_result.mismatched_keys == []

    def test_strict_missing_key_raises_without_writing(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict = {k: v for k, v in state_dict.items() if "lin0.lora_B" not in k}
        fresh = _make_peft_model()
        lora_B_before = fresh.base_model.model.lin0.lora_B.default.weight.detach().clone()
        with pytest.raises(RuntimeError, match="Missing key"):
            set_peft_model_state_dict(fresh, state_dict, strict=True)
        lora_B_after = fresh.base_model.model.lin0.lora_B.default.weight
        assert torch.allclose(lora_B_before, lora_B_after), "partial weights were written on strict failure"

    def test_strict_unexpected_key_raises_without_writing(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict["base_model.model.lin0.lora_C.weight"] = torch.randn(2, 8)
        fresh = _make_peft_model()
        with pytest.raises(RuntimeError, match="Unexpected key"):
            set_peft_model_state_dict(fresh, state_dict, strict=True)

    def test_strict_shape_mismatch_raises_without_writing(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict["base_model.model.lin0.lora_A.weight"] = torch.randn(3, 8)
        fresh = _make_peft_model()
        lora_A_before = fresh.base_model.model.lin0.lora_A.default.weight.detach().clone()
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            set_peft_model_state_dict(fresh, state_dict, strict=True)
        lora_A_after = fresh.base_model.model.lin0.lora_A.default.weight
        assert torch.allclose(lora_A_before, lora_A_after), "partial weights were written on strict failure"


class TestNonStrictMode:
    """Tests for set_peft_model_state_dict(strict=False) -- the default, tolerant behavior."""

    def test_non_strict_clean_state_dict_succeeds(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        load_result = set_peft_model_state_dict(model, state_dict, strict=False)
        assert load_result.unexpected_keys == []
        assert load_result.mismatched_keys == []

    def test_non_strict_shape_mismatch_raises_by_default(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict["base_model.model.lin0.lora_A.weight"] = torch.randn(3, 8)
        fresh = _make_peft_model()
        with pytest.raises(RuntimeError):
            set_peft_model_state_dict(fresh, state_dict)

    def test_non_strict_ignore_mismatched_sizes_skips_mismatched(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        state_dict = {k: torch.randn_like(v) for k, v in state_dict.items()}
        state_dict["base_model.model.lin0.lora_A.weight"] = torch.randn(3, 8)
        fresh = _make_peft_model()
        lora_A_before = fresh.base_model.model.lin0.lora_A.default.weight.detach().clone()
        lora_B_before = fresh.base_model.model.lin0.lora_B.default.weight.detach().clone()
        with pytest.warns(UserWarning, match="ignore_mismatched_sizes"):
            load_result = set_peft_model_state_dict(fresh, state_dict, ignore_mismatched_sizes=True)
        lora_A_after = fresh.base_model.model.lin0.lora_A.default.weight
        assert torch.allclose(lora_A_before, lora_A_after), "mismatched weight was written"
        lora_B_after = fresh.base_model.model.lin0.lora_B.default.weight
        assert not torch.allclose(lora_B_before, lora_B_after), "non-mismatched weight was not loaded"
        assert load_result.mismatched_keys
        assert any("lin0.lora_A" in k for k, _, _ in load_result.mismatched_keys)

    def test_non_strict_returns_three_fields(self):
        model = _make_peft_model()
        state_dict = get_peft_model_state_dict(model)
        load_result = set_peft_model_state_dict(model, state_dict)
        assert hasattr(load_result, "missing_keys")
        assert hasattr(load_result, "unexpected_keys")
        assert hasattr(load_result, "mismatched_keys")
        assert isinstance(load_result.missing_keys, list)
        assert isinstance(load_result.unexpected_keys, list)
        assert isinstance(load_result.mismatched_keys, list)
