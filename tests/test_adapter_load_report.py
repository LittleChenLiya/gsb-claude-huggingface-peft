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
"""Tests for the adapter weight preflight report (`check_peft_model_state_dict`) and the strict/non-strict loading
paths of `set_peft_model_state_dict`."""

import pytest
import torch
from torch import nn

from peft import LoraConfig, check_peft_model_state_dict, get_peft_model, get_peft_model_state_dict
from peft.utils import set_peft_model_state_dict


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.lin2(self.lin1(x))


def make_peft_model():
    torch.manual_seed(0)
    base_model = TinyMLP()
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["lin1", "lin2"])
    return get_peft_model(base_model, config)


def make_checkpoint(model):
    """A valid adapter checkpoint (saved key format) with non-trivial values."""
    state_dict = get_peft_model_state_dict(model)
    return {key: torch.randn_like(value) for key, value in state_dict.items()}


def snapshot_weights(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_weights_equal(model, snapshot):
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key]), f"weight {key} was modified"


class TestAdapterLoadReport:
    def test_clean_checkpoint_report_is_empty(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)

        report = check_peft_model_state_dict(model, checkpoint)

        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

    def test_report_does_not_modify_model(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        before = snapshot_weights(model)

        check_peft_model_state_dict(model, checkpoint)

        assert_weights_equal(model, before)

    def test_missing_key_is_reported(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        del checkpoint["base_model.model.lin2.lora_B.weight"]

        report = check_peft_model_state_dict(model, checkpoint)

        assert report.missing_keys == ["base_model.model.lin2.lora_B.default.weight"]
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

    def test_unexpected_key_is_reported(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["bogus.weight"] = torch.randn(4, 4)

        report = check_peft_model_state_dict(model, checkpoint)

        assert report.missing_keys == []
        assert report.unexpected_keys == ["bogus.weight"]
        assert report.mismatched_keys == []

    def test_mismatched_shape_is_reported(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["base_model.model.lin1.lora_A.weight"] = torch.randn(2, 8)  # model expects (4, 8)

        report = check_peft_model_state_dict(model, checkpoint)

        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert len(report.mismatched_keys) == 1
        key, checkpoint_shape, model_shape = report.mismatched_keys[0]
        assert key == "base_model.model.lin1.lora_A.default.weight"
        assert tuple(checkpoint_shape) == (2, 8)
        assert tuple(model_shape) == (4, 8)


class TestStrictLoading:
    def test_strict_load_of_clean_checkpoint_writes_weights(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)

        set_peft_model_state_dict(model, checkpoint, strict=True)

        for saved_key, value in checkpoint.items():
            model_key = saved_key.replace("lora_A.", "lora_A.default.").replace("lora_B.", "lora_B.default.")
            assert torch.equal(model.state_dict()[model_key], value)

    def test_strict_missing_key_raises_without_partial_write(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        del checkpoint["base_model.model.lin2.lora_B.weight"]
        before = snapshot_weights(model)

        with pytest.raises(RuntimeError, match="Missing key"):
            set_peft_model_state_dict(model, checkpoint, strict=True)

        assert_weights_equal(model, before)

    def test_strict_unexpected_key_raises_without_partial_write(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["bogus.weight"] = torch.randn(3)
        before = snapshot_weights(model)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            set_peft_model_state_dict(model, checkpoint, strict=True)

        assert_weights_equal(model, before)

    def test_strict_mismatched_shape_raises_without_partial_write(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["base_model.model.lin1.lora_A.weight"] = torch.randn(2, 8)
        before = snapshot_weights(model)

        with pytest.raises(RuntimeError, match="Size mismatch"):
            set_peft_model_state_dict(model, checkpoint, strict=True)

        # in particular, the other (valid) checkpoint weights must not have been written
        assert_weights_equal(model, before)

    def test_strict_mismatched_shape_tolerated_with_ignore_mismatched_sizes(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["base_model.model.lin1.lora_A.weight"] = torch.randn(2, 8)
        old_lora_a = model.state_dict()["base_model.model.lin1.lora_A.default.weight"].clone()

        with pytest.warns(UserWarning, match="ignore_mismatched_sizes"):
            set_peft_model_state_dict(model, checkpoint, strict=True, ignore_mismatched_sizes=True)

        state_dict = model.state_dict()
        # mismatched weight is skipped, the rest is loaded
        assert torch.equal(state_dict["base_model.model.lin1.lora_A.default.weight"], old_lora_a)
        assert torch.equal(
            state_dict["base_model.model.lin2.lora_A.default.weight"],
            checkpoint["base_model.model.lin2.lora_A.weight"],
        )


class TestNonStrictLoadingKeepsTolerantBehavior:
    def test_non_strict_missing_key_is_tolerated(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        del checkpoint["base_model.model.lin2.lora_B.weight"]
        old_lora_b = model.state_dict()["base_model.model.lin2.lora_B.default.weight"].clone()

        load_result = set_peft_model_state_dict(model, checkpoint)

        state_dict = model.state_dict()
        # the missing weight keeps its old value, the other weights are loaded
        assert torch.equal(state_dict["base_model.model.lin2.lora_B.default.weight"], old_lora_b)
        assert torch.equal(
            state_dict["base_model.model.lin1.lora_A.default.weight"],
            checkpoint["base_model.model.lin1.lora_A.weight"],
        )
        assert "base_model.model.lin2.lora_B.default.weight" in load_result.missing_keys

    def test_non_strict_unexpected_key_is_tolerated(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["bogus.weight"] = torch.randn(3)

        load_result = set_peft_model_state_dict(model, checkpoint)

        assert "bogus.weight" in load_result.unexpected_keys
        assert torch.equal(
            model.state_dict()["base_model.model.lin1.lora_A.default.weight"],
            checkpoint["base_model.model.lin1.lora_A.weight"],
        )

    def test_non_strict_mismatched_shape_with_ignore_mismatched_sizes_warns(self):
        model = make_peft_model()
        checkpoint = make_checkpoint(model)
        checkpoint["base_model.model.lin1.lora_A.weight"] = torch.randn(2, 8)
        old_lora_a = model.state_dict()["base_model.model.lin1.lora_A.default.weight"].clone()

        with pytest.warns(UserWarning, match="ignore_mismatched_sizes"):
            set_peft_model_state_dict(model, checkpoint, ignore_mismatched_sizes=True)

        state_dict = model.state_dict()
        assert torch.equal(state_dict["base_model.model.lin1.lora_A.default.weight"], old_lora_a)
        assert torch.equal(
            state_dict["base_model.model.lin2.lora_A.default.weight"],
            checkpoint["base_model.model.lin2.lora_A.weight"],
        )


class AttrDict(dict):
    """Minimal dict-like model config, as expected by prompt-learning setup."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def to_dict(self):
        return dict(self)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AttrDict(
            num_hidden_layers=1, num_attention_heads=1, hidden_size=8, model_type="tiny", vocab_size=50
        )
        self.emb = nn.Embedding(50, 8)
        self.lin = nn.Linear(8, 4)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, **kwargs):
        return self.lin(self.emb(input_ids))


class TestPromptLearningAndModulesToSave:
    def test_prompt_learning_clean_report_and_strict_load(self):
        from peft import PeftModel, PromptTuningConfig

        torch.manual_seed(0)
        config = PromptTuningConfig(num_virtual_tokens=4, token_dim=8, num_transformer_submodules=1)
        model = PeftModel(TinyDecoder(), config)
        checkpoint = get_peft_model_state_dict(model)

        report = check_peft_model_state_dict(model, checkpoint)
        assert report.missing_keys == []
        assert report.unexpected_keys == []
        assert report.mismatched_keys == []

        set_peft_model_state_dict(model, checkpoint, strict=True)

        checkpoint["bogus.weight"] = torch.randn(2)
        report = check_peft_model_state_dict(model, checkpoint)
        assert report.unexpected_keys == ["bogus.weight"]
        with pytest.raises(RuntimeError, match="Unexpected key"):
            set_peft_model_state_dict(model, checkpoint, strict=True)

    def test_modules_to_save_missing_key_is_reported_and_tolerated(self):
        torch.manual_seed(0)
        config = LoraConfig(r=4, lora_alpha=8, target_modules=["lin1"], modules_to_save=["lin2"])
        model = get_peft_model(TinyMLP(), config)
        checkpoint = get_peft_model_state_dict(model)
        # drop the modules_to_save weights (saved without the "modules_to_save" infix)
        checkpoint = {
            key: value for key, value in checkpoint.items() if not key.endswith(("lin2.weight", "lin2.bias"))
        }

        report = check_peft_model_state_dict(model, checkpoint)
        assert report.missing_keys == [
            "base_model.model.lin2.modules_to_save.default.weight",
            "base_model.model.lin2.modules_to_save.default.bias",
        ]

        # non-strict loading tolerates the missing keys (previously this raised a bare KeyError)
        load_result = set_peft_model_state_dict(model, checkpoint)
        assert "base_model.model.lin2.modules_to_save.default.weight" in load_result.missing_keys

        # strict loading raises atomically
        before = snapshot_weights(model)
        with pytest.raises(RuntimeError, match="Missing key"):
            set_peft_model_state_dict(model, checkpoint, strict=True)
        assert_weights_equal(model, before)
