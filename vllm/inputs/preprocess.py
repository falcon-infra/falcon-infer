# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


from vllm.config import VllmConfig
from vllm.inference_profile import validate_text_prompt
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs import (
    DecoderOnlyDictPrompt,
)
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.tokenizers import TokenizerLike

from .data import (
    DecoderOnlyInputs,
    ProcessorInputs,
    PromptType,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    token_inputs,
)

logger = init_logger(__name__)


class InputPreprocessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        mm_registry=None,
    ) -> None:
        super().__init__()

        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)
        self.mm_registry = mm_registry

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_tokens(
        self, parsed_content: TokensPrompt, tokenization_kwargs: dict[str, Any]
    ) -> TokenInputs:
        validate_text_prompt(parsed_content)
        tokens = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )
        result = token_inputs(tokens)
        for key in ("prompt", "cache_salt"):
            if key in parsed_content:
                result[key] = parsed_content[key]
        return result

    def _process_text(
        self, parsed_content: TextPrompt, tokenization_kwargs: dict[str, Any]
    ) -> TokenInputs:
        validate_text_prompt(parsed_content)
        result = token_inputs(
            self._tokenize_prompt(parsed_content["prompt"], tokenization_kwargs)
        )
        result["prompt"] = parsed_content["prompt"]
        if "cache_salt" in parsed_content:
            result["cache_salt"] = parsed_content["cache_salt"]
        return result

    def _prompt_to_llm_inputs(self, prompt, tokenization_kwargs=None) -> TokenInputs:
        validate_text_prompt(prompt)
        if "prompt_token_ids" in prompt:
            return self._process_tokens(prompt, tokenization_kwargs or {})
        return self._process_text(prompt, tokenization_kwargs or {})

    def _process_decoder_only_prompt(
        self,
        prompt: DecoderOnlyDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> DecoderOnlyInputs:
        """
        For decoder-only models:
        Process an input prompt into a
        [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance.

        Arguments:

        * prompt: input prompt

        Returns:

        * [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance
        """
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        from vllm.inference_profile import validate_text_prompt

        validate_text_prompt(prompt)

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            tokenization_kwargs=tokenization_kwargs,
        )
