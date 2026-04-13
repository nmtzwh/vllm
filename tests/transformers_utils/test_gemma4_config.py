# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.config import _normalize_text_only_architectures

pytestmark = pytest.mark.skip_global_cleanup


class _DummyTextConfig:
    model_type = "gemma4_text"
    num_attention_heads = 8


class _DummyConfig:
    def __init__(self, architectures):
        self.model_type = "gemma4"
        self.architectures = architectures
        self.text_config = _DummyTextConfig()

    def update(self, values):
        for key, value in values.items():
            setattr(self, key, value)

    def get_text_config(self):
        return self.text_config


def test_normalize_gemma4_conditional_generation_to_text():
    config = _DummyConfig(["Gemma4ForConditionalGeneration"])

    _normalize_text_only_architectures(config)

    assert config.architectures == ["Gemma4ForCausalLM"]


def test_fill_missing_gemma4_architecture_with_text_model():
    config = _DummyConfig(None)

    _normalize_text_only_architectures(config)

    assert config.architectures == ["Gemma4ForCausalLM"]
