"""Tests for the transactional adapter activation entry `PeftModel.set_adapters`.

These tests use a tiny CPU-only linear model with two LoRA adapters. They verify that:

1. multiple adapters can be activated in one transaction (`set_adapters`),
2. passing an unknown adapter name raises `ValueError` and leaves the previously active adapter set
   (including the adapter weights and `requires_grad` flags) completely untouched,
3. the old single-adapter interface `PeftModel.set_adapter` keeps working unchanged,
4. a failure mid-switch rolls back the active adapter weights and `requires_grad` flags instead of
   leaving a partially switched state.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 2)

    def forward(self, X):
        return self.lin1(self.relu(self.lin0(X)))


def build_model():
    torch.manual_seed(0)
    base_model = MLP()
    config = LoraConfig(target_modules=["lin0", "lin1"], r=4, lora_alpha=4, bias="none")
    peft_model = get_peft_model(base_model, config, adapter_name="adapter_1")
    peft_model.add_adapter("adapter_2", config)
    # LoRA initializes lora_B to zero, so the adapters would have no effect on the output.
    # Make both adapters contribute non-trivially so that switching adapters changes the output.
    for module in peft_model.modules():
        if hasattr(module, "lora_B"):
            for adapter_name in module.lora_B:
                module.lora_B[adapter_name].weight.data = torch.randn_like(module.lora_B[adapter_name].weight)
    peft_model.eval()
    return peft_model


def lora_param_state(peft_model):
    """Snapshot the active adapter selection and requires_grad flags of every LoRA layer."""
    state = {}
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            state[name] = (
                list(module.active_adapters),
                {
                    adapter_name: (
                        module.lora_A[adapter_name].weight.requires_grad,
                        module.lora_B[adapter_name].weight.requires_grad,
                    )
                    for adapter_name in module.lora_A
                },
            )
    return state


def test_set_adapters_successful_switch():
    model = build_model()
    X = torch.arange(90, dtype=torch.float32).view(9, 10)

    # activate both adapters in one transaction
    model.set_adapters(["adapter_1", "adapter_2"])
    assert model.active_adapters == ["adapter_1", "adapter_2"]
    for _, (active, _) in lora_param_state(model).items():
        assert active == ["adapter_1", "adapter_2"]

    # both adapters participate in the forward pass
    output_both = model(X)

    # switch to a single adapter via the new entry
    model.set_adapters(["adapter_2"])
    assert model.active_adapters == ["adapter_2"]
    output_adapter_2 = model(X)

    # the output must actually change depending on which adapters are active
    assert not torch.allclose(output_both, output_adapter_2, atol=1e-5)

    # switch back to both
    model.set_adapters(["adapter_1", "adapter_2"])
    assert model.active_adapters == ["adapter_1", "adapter_2"]
    assert torch.allclose(model(X), output_both, atol=1e-5)


def test_set_adapters_unknown_name_keeps_previous_state():
    model = build_model()
    X = torch.arange(90, dtype=torch.float32).view(9, 10)

    model.set_adapters(["adapter_1"])
    before = lora_param_state(model)
    output_before = model(X)

    # a single unknown name must be rejected ...
    with pytest.raises(ValueError, match="not found"):
        model.set_adapters(["adapter_1", "does_not_exist"])

    # ... and the original activation set must be fully preserved
    assert model.active_adapters == ["adapter_1"]
    assert lora_param_state(model) == before
    assert torch.allclose(model(X), output_before, atol=1e-5)

    # same when none of the names exist
    with pytest.raises(ValueError, match="not found"):
        model.set_adapters(["nope_1", "nope_2"])
    assert model.active_adapters == ["adapter_1"]
    assert lora_param_state(model) == before
    assert torch.allclose(model(X), output_before, atol=1e-5)


def test_set_adapters_does_not_partially_switch_requires_grad():
    model = build_model()

    # trainable adapters: both LoRA adapters require grads
    model.set_adapters(["adapter_1", "adapter_2"], inference_mode=False)
    for _, (_, grads) in lora_param_state(model).items():
        assert grads["adapter_1"] == (True, True)
        assert grads["adapter_2"] == (True, True)

    # switching to inference mode must freeze every adapter, not just some
    model.set_adapters(["adapter_1", "adapter_2"], inference_mode=True)
    for _, (_, grads) in lora_param_state(model).items():
        assert grads["adapter_1"] == (False, False)
        assert grads["adapter_2"] == (False, False)

    # switching to a subset must only activate that subset
    model.set_adapters(["adapter_1"], inference_mode=False)
    for _, (_, grads) in lora_param_state(model).items():
        assert grads["adapter_1"] == (True, True)
        assert grads["adapter_2"] == (False, False)


def test_set_adapters_rolls_back_on_mid_switch_failure():
    model = build_model()
    X = torch.arange(90, dtype=torch.float32).view(9, 10)

    model.set_adapters(["adapter_1"])
    before = lora_param_state(model)
    output_before = model(X)

    # force a failure while the switch is in progress: the first LoRA layer succeeds,
    # the second one raises. The transaction must roll back the first layer as well.
    lora_layers = [m for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B")]
    real_set_adapter = type(lora_layers[1]).set_adapter

    def failing_set_adapter(self, adapter_names, inference_mode=False):
        if self is lora_layers[1]:
            raise RuntimeError("boom")
        return real_set_adapter(self, adapter_names, inference_mode=inference_mode)

    type(lora_layers[1]).set_adapter = failing_set_adapter
    try:
        with pytest.raises(RuntimeError, match="boom"):
            model.set_adapters(["adapter_1", "adapter_2"])
    finally:
        type(lora_layers[1]).set_adapter = real_set_adapter

    # the active adapter weights and requires_grad flags must be restored exactly
    assert model.active_adapters == ["adapter_1"]
    assert lora_param_state(model) == before
    assert torch.allclose(model(X), output_before, atol=1e-5)


def test_legacy_set_adapter_unchanged():
    model = build_model()
    X = torch.arange(90, dtype=torch.float32).view(9, 10)

    # the old single-adapter interface still works as before
    model.set_adapter("adapter_2")
    assert model.active_adapters == ["adapter_2"]
    output_adapter_2 = model(X)

    with pytest.raises(ValueError, match="not found"):
        model.set_adapter("does_not_exist")
    assert model.active_adapters == ["adapter_2"]
    assert torch.allclose(model(X), output_adapter_2, atol=1e-5)
