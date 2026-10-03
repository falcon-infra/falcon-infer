# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar

from vllm.inference_profile import validate_text_prompt
from vllm.inputs import (
    EmbedsPrompt,
    ProcessorInputs,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
)
from vllm.inputs.data import token_inputs
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.v1.metrics.stats import MultiModalCacheStats

from .inputs import (
    DictPrompt,
    SingletonDictPrompt,
    SingletonTokPrompt,
    TokPrompt,
)
from .inputs.preprocess import extract_target_prompt
from .params import ChatParams, TokenizeParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )

logger = init_logger(__name__)


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


class BaseRenderer(ABC, Generic[_T]):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "VllmConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        raise NotImplementedError

    def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
        super().__init__()

        self.config = config
        self.model_config = config.model_config
        self.api_process_rank = config.parallel_config._api_process_rank

        self.tokenizer = tokenizer

        # Lazy initialization since offline LLM doesn't use async
        self._async_tokenizer: AsyncMicrobatchTokenizer | None = None

        self.mm_processor = None
        self._mm_cache_stats: MultiModalCacheStats | None = None
        pass  # Unsupported P4 branch removed.

    def get_tokenizer(self) -> _T:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        if self._async_tokenizer is None:
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())

        return self._async_tokenizer

    def get_mm_processor(self):
        raise ValueError("Ascend P4 is text-only; no multimodal processor is installed")

    @property
    def mm_processor_cache(self) -> None:
        return None

    def stat_mm_cache(self) -> None:
        return None

    def update_mm_cache_stats(self) -> None:
        """Compatibility hook: text rendering has no media cache."""

    def clear_mm_cache(self) -> None:
        """Compatibility hook: text rendering has no media cache."""

    def warmup(self, chat_params: ChatParams) -> None:
        """
        Warm up this renderer to avoid first-request latency.

        For chat requests:
        - Jinja2 template compilation

        """
        from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

        try:
            logger.debug("Warming up chat template processing...")
            start_time = time.perf_counter()

            self.render_chat([[{"role": "user", "content": "warmup"}]], chat_params)

            elapsed = time.perf_counter() - start_time
            logger.debug("Chat template warmup completed in %.3fs", elapsed)
        except ChatTemplateResolutionError:
            logger.debug("This model does not support chat template.")
        except Exception:
            logger.warning("Chat template warmup failed", exc_info=True)

    def shutdown(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.close()

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    # Step 1: Convert raw inputs to prompts
    def render_prompt(self, prompt: DictPrompt | bytes) -> DictPrompt:
        validate_text_prompt(prompt)
        return prompt

    def render_prompts(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_prompt(prompt) for prompt in prompts]

    async def render_prompts_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        return self.render_prompts(prompts)

    @abstractmethod
    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt_token_ids = tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    async def _tokenize_prompt_async(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt_token_ids = await tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    def _detokenize_prompt(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt["prompt"] = tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    async def _detokenize_prompt_async(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt["prompt"] = await tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    @overload
    def _tokenize_singleton_prompt(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    def _tokenize_singleton_prompt(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    def _tokenize_singleton_prompt(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        validate_text_prompt(prompt)
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    @overload
    async def _tokenize_singleton_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    async def _tokenize_singleton_prompt_async(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    async def _tokenize_singleton_prompt_async(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        validate_text_prompt(prompt)
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def tokenize_prompt(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        validate_text_prompt(prompt)

        return self._tokenize_singleton_prompt(prompt, params)

    def tokenize_prompts(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        validate_text_prompt(prompt)

        return await self._tokenize_singleton_prompt_async(prompt, params)

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )

    # Step 3: Add extra keys to the prompts
    def _apply_prompt_extras(
        self,
        prompts: Sequence[TokPrompt],
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]

    # Step 4: Convert to engine inputs

    # TODO: Remove str and tokenization_kwargs after deprecating InputPreprocessor

    def _process_tokens(self, prompt: TokensPrompt) -> TokenInputs:
        validate_text_prompt(prompt)
        inputs = token_inputs(prompt["prompt_token_ids"])
        if "prompt" in prompt:
            inputs["prompt"] = prompt["prompt"]
        if "cache_salt" in prompt:
            inputs["cache_salt"] = prompt["cache_salt"]
        return inputs

    def _process_singleton(self, prompt: SingletonTokPrompt) -> TokenInputs:
        return self._process_tokens(prompt)

    def process_for_engine(
        self, prompt: TokPrompt, arrival_time: float
    ) -> ProcessorInputs:
        validate_text_prompt(prompt)
        result = self._process_tokens(prompt)
        result["arrival_time"] = arrival_time
        return result

    # Top-level methods
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]

    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]

    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts

    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts
