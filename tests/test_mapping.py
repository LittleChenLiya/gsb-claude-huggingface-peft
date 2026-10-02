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
import copy

import pytest
import torch
from transformers import PretrainedConfig

from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    PeftModelForCausalLM,
    PromptTuningConfig,
    explain_adapter_type_selection,
    get_peft_model,
)


class TestGetPeftModel:
    RELOAD_WARNING_EXPECTED_MATCH = r"You are trying to modify a model .*"

    @pytest.fixture
    def lora_config_0(self):
        return LoraConfig(target_modules="0")

    @pytest.fixture
    def base_model(self):
        return torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

    def test_get_peft_model_warns_when_reloading_model(self, lora_config_0, base_model):
        get_peft_model(base_model, lora_config_0)

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(base_model, lora_config_0)

    def test_get_peft_model_proposed_fix_in_warning_helps(self, lora_config_0, base_model, recwarn):
        peft_model = get_peft_model(base_model, lora_config_0)
        peft_model.unload()
        get_peft_model(base_model, lora_config_0)

        warning_checker = pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH)

        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Warning raised even though model was unloaded.")

    def test_get_peft_model_repeated_invocation(self, lora_config_0, base_model):
        peft_model = get_peft_model(base_model, lora_config_0)

        # use direct-addressing of the other layer to accommodate for the nested model
        lora_config_1 = LoraConfig(target_modules="base_model.model.1")

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(peft_model, lora_config_1)


class TestExplainAdapterTypeSelection:
    """Tests for peft.explain_adapter_type_selection, which explains which PEFT model wrapper class a config maps to
    without instantiating a model or modifying the config."""

    @pytest.fixture
    def base_model(self):
        return torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

    @pytest.fixture
    def causal_lm_model(self):
        # minimal stand-in for a causal LM: PeftModelForCausalLM expects `prepare_inputs_for_generation` and prompt
        # learning setup expects a transformers-style config
        class DummyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.config = PretrainedConfig(
                    vocab_size=100, hidden_size=10, num_hidden_layers=2, num_attention_heads=2
                )

            def forward(self, *args, **kwargs):
                return self.linear(torch.zeros(1, 10))

            @property
            def device(self):
                return self.linear.weight.device

            def prepare_inputs_for_generation(self, *args, **kwargs):
                return {}

        return DummyCausalLM()

    def test_lora_config_explains_generic_peft_model(self, base_model):
        config = LoraConfig(target_modules=["0"])

        explanation = explain_adapter_type_selection(config)

        assert "LORA" in explanation
        assert "PeftModel" in explanation

        # the explanation must match the wrapper that get_peft_model actually selects
        peft_model = get_peft_model(base_model, config)
        assert type(peft_model).__name__ in explanation
        assert isinstance(peft_model, PeftModel)

    def test_lora_config_with_task_type_explains_task_specific_wrapper(self, causal_lm_model):
        config = LoraConfig(target_modules=["linear"], task_type="CAUSAL_LM")

        explanation = explain_adapter_type_selection(config)

        assert "PeftModelForCausalLM" in explanation

        peft_model = get_peft_model(causal_lm_model, config)
        assert type(peft_model).__name__ in explanation
        assert isinstance(peft_model, PeftModelForCausalLM)

    def test_mixed_explains_mixed_model(self, base_model):
        config = LoraConfig(target_modules=["0"])

        explanation = explain_adapter_type_selection(config, mixed=True)

        assert "PeftMixedModel" in explanation

        peft_model = get_peft_model(base_model, config, mixed=True)
        assert type(peft_model).__name__ in explanation
        assert isinstance(peft_model, PeftMixedModel)

    def test_prompt_tuning_config_explains_task_specific_wrapper(self, causal_lm_model):
        config = PromptTuningConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=4, num_layers=2, token_dim=10, num_attention_heads=2
        )

        explanation = explain_adapter_type_selection(config)

        assert "PROMPT_TUNING" in explanation
        assert "PeftModelForCausalLM" in explanation

        # prompt learning configs are wrapped by the task-specific PEFT model
        peft_model = get_peft_model(causal_lm_model, config)
        assert type(peft_model).__name__ in explanation
        assert isinstance(peft_model, PeftModelForCausalLM)

    def test_unknown_peft_type_raises_readable_error(self, base_model):
        config = LoraConfig(target_modules=["0"])
        config.peft_type = "NOT_A_REAL_PEFT_TYPE"

        with pytest.raises(ValueError, match="Unknown PEFT type: 'NOT_A_REAL_PEFT_TYPE'"):
            explain_adapter_type_selection(config)

        # get_peft_model surfaces the same readable error instead of an opaque KeyError
        with pytest.raises(ValueError, match="Unknown PEFT type: 'NOT_A_REAL_PEFT_TYPE'"):
            get_peft_model(base_model, config)

    def test_explanation_does_not_modify_config(self):
        config = LoraConfig(target_modules=["0"], task_type="CAUSAL_LM")
        config_before = copy.deepcopy(config)

        explain_adapter_type_selection(config)
        explain_adapter_type_selection(config, mixed=True)

        assert config.to_dict() == config_before.to_dict()
