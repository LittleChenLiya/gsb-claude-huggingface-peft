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
Tests for the adapter type selection explanation wired into get_peft_model from mapping_func.py.

The explanation (explain_adapter_type_selection) describes which wrapper class a given PEFT config maps to, without
instantiating a model and without modifying the config. These tests verify that:

- the explanation matches the wrapper that get_peft_model actually returns, for LoRA and prompt tuning configs;
- unknown adapter types raise a readable error (both from the explanation and from get_peft_model);
- the explanation does not instantiate a model and does not modify the config;
- the explanation is attached to the created PEFT model and logged.

The tests use a tiny, randomly initialized transformers model built from a local config, so they are fully
reproducible offline (no model weights are downloaded).
"""

import logging

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from peft import (
    AdapterTypeSelection,
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    PeftModelForCausalLM,
    PromptTuningConfig,
    explain_adapter_type_selection,
    get_peft_model,
)


def _tiny_causal_lm():
    # A tiny GPT2 model built from a local config: no network access required.
    config = GPT2Config(vocab_size=128, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config)


class TestExplainAdapterTypeSelection:
    def test_lora_with_causal_lm_task_type_maps_to_peft_model_for_causal_lm(self):
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, target_modules=["c_attn"])

        selection = explain_adapter_type_selection(peft_config)

        assert isinstance(selection, AdapterTypeSelection)
        assert selection.peft_type == "LORA"
        assert selection.task_type == "CAUSAL_LM"
        assert selection.is_prompt_learning is False
        assert selection.mixed is False
        assert selection.wrapper_class is PeftModelForCausalLM
        assert selection.wrapper_name == "PeftModelForCausalLM"

        # The explanation must match the wrapper that get_peft_model actually returns.
        peft_model = get_peft_model(_tiny_causal_lm(), peft_config)
        assert isinstance(peft_model, PeftModelForCausalLM)
        assert type(peft_model).__name__ == selection.wrapper_name
        # The explanation is attached to the created model so callers can inspect it.
        assert peft_model.adapter_type_selection == selection

    def test_lora_without_task_type_maps_to_generic_peft_model(self):
        peft_config = LoraConfig(r=4, lora_alpha=16, target_modules=["c_attn"])

        selection = explain_adapter_type_selection(peft_config)

        assert selection.wrapper_class is PeftModel
        assert selection.task_type is None

        peft_model = get_peft_model(_tiny_causal_lm(), peft_config)
        assert isinstance(peft_model, PeftModel)
        assert not isinstance(peft_model, PeftModelForCausalLM)

    def test_prompt_tuning_maps_to_task_specific_wrapper(self):
        peft_config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=10)

        selection = explain_adapter_type_selection(peft_config)

        assert selection.peft_type == "PROMPT_TUNING"
        assert selection.task_type == "CAUSAL_LM"
        assert selection.is_prompt_learning is True
        assert selection.wrapper_class is PeftModelForCausalLM

        peft_model = get_peft_model(_tiny_causal_lm(), peft_config)
        assert isinstance(peft_model, PeftModelForCausalLM)
        assert type(peft_model).__name__ == selection.wrapper_name
        assert peft_model.adapter_type_selection == selection

    def test_mixed_flag_maps_to_peft_mixed_model(self):
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, target_modules=["c_attn"])

        selection = explain_adapter_type_selection(peft_config, mixed=True)

        assert selection.mixed is True
        assert selection.wrapper_class is PeftMixedModel

        peft_model = get_peft_model(_tiny_causal_lm(), peft_config, mixed=True)
        assert isinstance(peft_model, PeftMixedModel)
        assert type(peft_model).__name__ == selection.wrapper_name
        assert peft_model.adapter_type_selection == selection

    def test_unknown_adapter_type_raises_readable_error(self):
        peft_config = PeftConfig(peft_type="BOGUS_ADAPTER_TYPE")

        with pytest.raises(ValueError, match="Unknown PEFT adapter type") as exc_info:
            explain_adapter_type_selection(peft_config)

        # The error should be readable: name the unknown type and list the supported types.
        message = str(exc_info.value)
        assert "BOGUS_ADAPTER_TYPE" in message
        assert "LORA" in message
        assert "PROMPT_TUNING" in message

        # get_peft_model must raise the same readable error instead of a cryptic KeyError.
        with pytest.raises(ValueError, match="Unknown PEFT adapter type"):
            get_peft_model(_tiny_causal_lm(), peft_config)

    def test_missing_adapter_type_raises_readable_error(self):
        peft_config = PeftConfig()  # peft_type is None

        with pytest.raises(ValueError, match="Unknown PEFT adapter type"):
            explain_adapter_type_selection(peft_config)

    def test_explanation_does_not_instantiate_model(self):
        # The explanation takes no model argument; a model that must not be touched is irrelevant here.
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, target_modules=["c_attn"])

        selection = explain_adapter_type_selection(peft_config)

        assert selection.wrapper_class is PeftModelForCausalLM

    def test_explanation_does_not_modify_config(self):
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            target_modules=["c_attn"],
            base_model_name_or_path="keep-this-name",
            inference_mode=True,
        )
        config_before = (
            peft_config.peft_type,
            peft_config.task_type,
            peft_config.base_model_name_or_path,
            peft_config.inference_mode,
        )

        explain_adapter_type_selection(peft_config)

        config_after = (
            peft_config.peft_type,
            peft_config.task_type,
            peft_config.base_model_name_or_path,
            peft_config.inference_mode,
        )
        assert config_before == config_after

    def test_explanation_string_is_readable(self):
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, target_modules=["c_attn"])
        selection = explain_adapter_type_selection(peft_config)

        text = str(selection)
        assert "LORA" in text
        assert "CAUSAL_LM" in text
        assert "PeftModelForCausalLM" in text

    def test_get_peft_model_logs_explanation(self, caplog):
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=4, target_modules=["c_attn"])

        with caplog.at_level(logging.INFO, logger="peft.mapping_func"):
            get_peft_model(_tiny_causal_lm(), peft_config)

        assert any("Adapter type selection" in record.message for record in caplog.records)
        assert any("PeftModelForCausalLM" in record.message for record in caplog.records)
