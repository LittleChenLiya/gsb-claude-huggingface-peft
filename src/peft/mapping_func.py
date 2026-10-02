# Copyright 2024-present the HuggingFace Inc. team.
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

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, overload

from torch import nn

from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .config import PeftConfig, PromptLearningConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from .mixed_model import PeftMixedModel
from .peft_model import PeftModel
from .tuners.tuners_utils import BaseTuner, BaseTunerLayer


logger = logging.getLogger(__name__)


@dataclass
class AdapterTypeSelection:
    """
    Explanation of which PEFT wrapper class a config maps to when calling [`get_peft_model`].

    Attributes:
        peft_type: The adapter type of the config (e.g. `"LORA"`), or `None` if the config does not specify one.
        task_type: The task type of the config (e.g. `"CAUSAL_LM"`), or `None`.
        is_prompt_learning: Whether the config is a prompt-learning method (e.g. prompt tuning).
        mixed: Whether the mixed-model path was selected.
        wrapper_class: The wrapper class that would be instantiated.
        wrapper_name: The name of the wrapper class.
        reason: A human-readable explanation of why `wrapper_class` was selected.
    """

    peft_type: Optional[str]
    task_type: Optional[str]
    is_prompt_learning: bool
    mixed: bool
    wrapper_class: type
    wrapper_name: str
    reason: str

    def __str__(self) -> str:
        return (
            f"Adapter type selection: peft_type={self.peft_type!r}, task_type={self.task_type!r}, "
            f"is_prompt_learning={self.is_prompt_learning}, mixed={self.mixed} -> {self.wrapper_name}. "
            f"{self.reason}"
        )


def explain_adapter_type_selection(peft_config: PeftConfig, mixed: bool = False) -> AdapterTypeSelection:
    """
    Explain which wrapper class [`get_peft_model`] would select for `peft_config`.

    This is a pure, non-mutating description of the selection logic: it never instantiates a model and never modifies
    `peft_config`. It can be used to inspect or log the expected wrapper before (or instead of) creating a PEFT model.

    Args:
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the PEFT model.
        mixed (`bool`, *optional*, defaults to `False`):
            Whether the mixed-model path should be considered (see [`get_peft_model`]).

    Returns:
        `AdapterTypeSelection`: An explanation of the wrapper that would be instantiated.

    Raises:
        ValueError: If `peft_config.peft_type` is not a known PEFT adapter type.
    """
    peft_type = peft_config.peft_type
    known_peft_types = sorted(
        getattr(peft_type, "value", str(peft_type)) for peft_type in PEFT_TYPE_TO_CONFIG_MAPPING
    )
    if peft_type not in PEFT_TYPE_TO_CONFIG_MAPPING:
        raise ValueError(
            f"Unknown PEFT adapter type {peft_type!r} in {type(peft_config).__name__}. "
            f"Supported adapter types are: {', '.join(known_peft_types)}. "
            "Please provide a valid PEFT config, e.g. `LoraConfig`, `PromptTuningConfig`, etc."
        )

    task_type = peft_config.task_type
    is_prompt_learning = peft_config.is_prompt_learning

    if mixed:
        wrapper_class = PeftMixedModel
        reason = (
            "`mixed=True` was passed, so the config is wrapped in `PeftMixedModel`, which allows mixing compatible "
            "adapter types."
        )
    elif task_type in MODEL_TYPE_TO_PEFT_MODEL_MAPPING:
        wrapper_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
        reason = (
            f"`task_type={task_type!r}` maps to `{wrapper_class.__name__}` via `MODEL_TYPE_TO_PEFT_MODEL_MAPPING`."
        )
    elif is_prompt_learning:
        wrapper_class = PeftModel
        reason = (
            "The config is a prompt-learning config without a supported `task_type`; prompt learning is handled "
            "directly by `PeftModel`, as the base model is not wrapped in a tuner model."
        )
    else:
        wrapper_class = PeftModel
        reason = (
            f"`task_type={task_type!r}` is not in `MODEL_TYPE_TO_PEFT_MODEL_MAPPING` and the config is not a "
            "prompt-learning config, so it is wrapped in the generic `PeftModel`."
        )

    # `peft_type`/`task_type` may be plain strings or str-Enum members (PeftType/TaskType); normalize to their value.
    peft_type_value = getattr(peft_type, "value", peft_type)
    task_type_value = getattr(task_type, "value", task_type)

    return AdapterTypeSelection(
        peft_type=peft_type_value if peft_type_value is not None else None,
        task_type=task_type_value if task_type_value is not None else None,
        is_prompt_learning=is_prompt_learning,
        mixed=mixed,
        wrapper_class=wrapper_class,
        wrapper_name=wrapper_class.__name__,
        reason=reason,
    )


if TYPE_CHECKING:
    from .tuners import LoraConfig, LoraModel

    # A PeftModel forwards missing attributes to the wrapped tuner model via __getattr__, so its interface is
    # effectively the intersection of both classes. Python's type system cannot express intersections, so we emulate
    # them for static type checkers with classes that inherit from both, with PeftModel first in the MRO to mirror the
    # runtime attribute lookup order. These classes only exist for type checkers; at runtime, get_peft_model returns a
    # plain PeftModel (subclass).
    class _LoraPeftModel(PeftModel, LoraModel):  # type: ignore[misc]
        ...

    class _TunerPeftModel(PeftModel, BaseTuner):  # type: ignore[misc]
        ...

    # The overloads only affect static type checking, the returned types are chosen based on what attributes are
    # reachable through __getattr__ forwarding: LoRA has extra methods like add_weighted_adapter, other tuners provide
    # the BaseTuner API (e.g. merge_and_unload), and prompt learning wraps the base model directly, i.e. there is no
    # tuner model whose attributes could be forwarded. Overloads need to be defined before the actual function.
    @overload
    def get_peft_model(
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = ...,
        *,
        mixed: Literal[True],
        autocast_adapter_dtype: bool = ...,
        revision: Optional[str] = ...,
        low_cpu_mem_usage: bool = ...,
    ) -> PeftMixedModel: ...

    @overload
    def get_peft_model(
        model: nn.Module,
        peft_config: LoraConfig,
        adapter_name: str = ...,
        mixed: Literal[False] = ...,
        autocast_adapter_dtype: bool = ...,
        revision: Optional[str] = ...,
        low_cpu_mem_usage: bool = ...,
    ) -> _LoraPeftModel: ...

    @overload
    def get_peft_model(
        model: nn.Module,
        peft_config: PromptLearningConfig,
        adapter_name: str = ...,
        mixed: Literal[False] = ...,
        autocast_adapter_dtype: bool = ...,
        revision: Optional[str] = ...,
        low_cpu_mem_usage: bool = ...,
    ) -> PeftModel: ...

    @overload
    def get_peft_model(
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = ...,
        mixed: Literal[False] = ...,
        autocast_adapter_dtype: bool = ...,
        revision: Optional[str] = ...,
        low_cpu_mem_usage: bool = ...,
    ) -> _TunerPeftModel: ...

    @overload
    def get_peft_model(
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = ...,
        mixed: bool = ...,
        autocast_adapter_dtype: bool = ...,
        revision: Optional[str] = ...,
        low_cpu_mem_usage: bool = ...,
    ) -> PeftModel | PeftMixedModel: ...


def get_peft_model(
    model: nn.Module,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.

    Args:
        model ([`torch.nn.Module`]):
            Model to be wrapped. Typically this is a Transformers model but any `nn.Module` can work, with the caveat
            that task-specific features (`peft_config.task_type`) require the model to follow Transformers conventions.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process. Leave this setting as
            False if you intend on training the model, unless the adapter weights will be replaced by different weights
            before training starts.
    """
    old_name = peft_config.base_model_name_or_path
    # Transformers from-config models use name_or_path == ""; that is not a Hub id.
    new_name = model.__dict__.get("name_or_path", None) or None
    peft_config.base_model_name_or_path = new_name

    # Especially in notebook environments there could be a case that a user wants to experiment with different
    # configuration values. However, it is likely that there won't be any changes for new configs on an already
    # initialized PEFT model. The best we can do is warn the user about it.
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload the model with a "
            "different config, make sure to call `.unload()` before."
        )

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
    if prefix and adapter_name in prefix:
        warnings.warn(
            f"Adapter name '{adapter_name}' should not be contained in the prefix '{prefix}'. "
            "This may lead to reinitialization of the adapter weights during loading."
        )

    # Explain which wrapper the config maps to before instantiating anything. This also validates the adapter type
    # and raises a readable error for unknown types, without touching the model or the config.
    selection = explain_adapter_type_selection(peft_config, mixed=mixed)
    logger.info("%s", selection)

    if mixed:
        # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
        peft_model = selection.wrapper_class(model, peft_config, adapter_name=adapter_name)
    else:
        peft_model = selection.wrapper_class(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    # Attach the explanation so callers can inspect which wrapper was selected and why.
    peft_model.adapter_type_selection = selection
    return peft_model
