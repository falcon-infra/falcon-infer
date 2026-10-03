# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import json
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from fastapi import Request
from openai.types.responses import (
    ToolChoiceFunction,
)
from pydantic import ConfigDict, TypeAdapter, ValidationError
from starlette.datastructures import Headers

from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    GenerationError,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest, GenerateResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.utils import create_error_response
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import (
    ProcessorInputs,
    PromptType,
    SingletonPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator,
)

logger = init_logger(__name__)


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest | TokenizeCompletionRequest | DetokenizeRequest
)

ChatLikeRequest: TypeAlias = ChatCompletionRequest | TokenizeChatRequest


AnyRequest: TypeAlias = CompletionLikeRequest | ChatLikeRequest | GenerateRequest

AnyResponse: TypeAlias = (
    CompletionResponse | ChatCompletionResponse | TokenizeResponse | GenerateResponse
)

RequestT = TypeVar("RequestT", bound=AnyRequest)


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_prompts: list[ProcessorInputs] | None = None

    result_generator: AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None = (
        None
    )
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenAIServing:
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__()

        self.engine_client = engine_client

        self.models = models

        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids

        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.io_processor = engine_client.io_processor
        self.input_processor = engine_client.input_processor

    async def beam_search(
        self,
        prompt: ProcessorInputs,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        if prompt["type"] == "embeds":
            raise NotImplementedError("Embedding prompt not supported for beam search")

        # Extract prompt tokens and text based on model type
        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        prompt_text = decoder_prompt.get("prompt")
        prompt_token_ids = decoder_prompt["prompt_token_ids"]

        tokenized_length = len(prompt_token_ids)

        logprobs_num = 2 * beam_width
        sampling_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(
                orig_prompt=prompt,
                tokens=prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                lora_request=lora_request,
            )
        ]
        completed = []

        for _ in range(max_tokens):
            tasks = []
            request_id_batch = f"{request_id}-{random_uuid()}"

            for i, beam in enumerate(all_beams):
                prompt_item = beam.get_prompt()
                lora_request_item = beam.lora_request
                request_id_item = f"{request_id_batch}-beam-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate(
                            prompt_item,
                            sampling_params,
                            request_id_item,
                            lora_request=lora_request_item,
                            trace_headers=trace_headers,
                        )
                    )
                )
                tasks.append(task)

            output = [x[0] for x in await asyncio.gather(*tasks)]

            new_beams = []
            # Store all new tokens generated by beam
            all_beams_token_id = []
            # Store the cumulative probability of all tokens
            # generated by beam search
            all_beams_logprob = []
            # Iterate through all beam inference results
            for i, result in enumerate(output):
                current_beam = all_beams[i]

                # check for error finish reason and abort beam search
                if result.outputs[0].finish_reason == "error":
                    # yield error output and terminate beam search
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                        finished=True,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                    )
                    return

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend(
                        [
                            current_beam.cum_logprob + obj.logprob
                            for obj in logprobs.values()
                        ]
                    )

            # Handle the token for the end of sentence (EOS)
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)

            if not ignore_eos:
                # Get the index position of eos token in all generated results
                eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    assert result.outputs[0].logprobs is not None
                    logprobs_entry = result.outputs[0].logprobs[0]
                    completed.append(
                        BeamSearchSequence(
                            orig_prompt=prompt,
                            tokens=current_beam.tokens + [eos_token_id]
                            if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs + [logprobs_entry],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=eos_token_id,
                        )
                    )
                # After processing, set the log probability of the eos condition
                # to negative infinity.
                all_beams_logprob[eos_idx] = -np.inf

            # Processing non-EOS tokens
            # Get indices of the top beam_width probabilities
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[
                :beam_width
            ]

            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]
                token_id = int(all_beams_token_id[idx])
                assert result.outputs[0].logprobs is not None
                logprobs_entry = result.outputs[0].logprobs[0]
                new_beams.append(
                    BeamSearchSequence(
                        orig_prompt=prompt,
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs + [logprobs_entry],
                        lora_request=current_beam.lora_request,
                        cum_logprob=float(all_beams_logprob[idx]),
                    )
                )

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,  # type: ignore
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=beam.finish_reason
                    if beam.finish_reason is not None
                    else "length",
                    stop_reason=beam.stop_reason,
                )
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
        )

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        """
        Default preprocessing hook. Subclasses may override to prepare `ctx`.
        """
        return None

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> AnyResponse | ErrorResponse:
        """
        Default response builder. Subclass may override this method
        to return the appropriate response object.
        """
        return self.create_error_response("unimplemented endpoint")

    def _validate_request(self, ctx: ServeContext) -> ErrorResponse | None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.model_config.max_model_len
        ):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size."
            )
        return None

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _check_model(self, request: AnyRequest) -> ErrorResponse | None:
        return await self.models.check_model(request.model)

    def _maybe_get_adapters(
        self, request: AnyRequest, supports_default_mm_loras: bool = False
    ) -> None:
        # Retained internal call signature; no adapter lookup or mutation.
        if not self._is_model_supported(request.model):
            raise ValueError(f"The model `{request.model}` does not exist.")
        return None

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)
        max_model_len = self.model_config.max_model_len

        # Note: ScoreRequest doesn't have max_tokens

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages.",
                parameter="input_tokens",
                value=token_num,
            )

        if max_tokens is not None and token_num + max_tokens > max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{max_model_len} tokens. However, you requested "
                f"{max_tokens} output tokens and your prompt contains "
                f"{token_num} input tokens, for a total of "
                f"{token_num + max_tokens} tokens "
                f"({token_num} + {max_tokens} = "
                f"{token_num + max_tokens} > {max_model_len}). "
                f"Please reduce the length of the input prompt or the "
                f"number of requested output tokens.",
                parameter="max_tokens",
                value=max_tokens,
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    @staticmethod
    def _prepare_extra_chat_template_kwargs(
        request_chat_template_kwargs: dict[str, Any] | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Helper to merge server-default and request-specific chat template kwargs."""
        request_chat_template_kwargs = request_chat_template_kwargs or {}
        if default_chat_template_kwargs is None:
            return request_chat_template_kwargs
        # Apply server defaults first, then request kwargs override.
        return default_chat_template_kwargs | request_chat_template_kwargs

    async def _preprocess_completion(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[ProcessorInputs]:
        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))

        return await self._preprocess_cmpl(request, prompts)

    async def _preprocess_cmpl(
        self,
        request: RendererRequest,
        prompts: Sequence[PromptType | bytes],
    ) -> list[ProcessorInputs]:
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

    async def _preprocess_chat(
        self,
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]]:
        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=False,
            ),
        )

        mm_config = self.model_config.multimodal_config

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        if tool_parser is not None:
            tool_choice = getattr(request, "tool_choice", "none")
            if tool_choice != "none":
                if not isinstance(request, ChatCompletionRequest):
                    msg = (
                        "Tool usage is only supported for Chat Completions API "
                        "or Responses API requests."
                    )
                    raise NotImplementedError(msg)

                # TODO: Update adjust_request to accept ResponsesRequest
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

        return conversation, [engine_prompt]

    def _extract_prompt_components(self, prompt: PromptType | ProcessorInputs):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: ProcessorInputs):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: ProcessorInputs):
        return extract_prompt_len(self.model_config, prompt)

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
            lora_request=lora_request,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    @staticmethod
    def _parse_tool_calls_from_content(
        request: ChatCompletionRequest,
        tokenizer: TokenizerLike | None,
        enable_auto_tools: bool,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        content: str | None = None,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        function_calls = list[FunctionCall]()
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            assert content is not None
            # Forced Function Call
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            content = None  # Clear content since tool is called.
        elif request.tool_choice == "required":
            tool_calls = []
            with contextlib.suppress(ValidationError):
                content = content or ""
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(
                    content
                )
            for tool_call in tool_calls:
                function_calls.append(
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                )
            content = None  # Clear content since tool is called.
        elif (
            tool_parser_cls
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer not available when `skip_tokenizer_init=True`"
                )

            # Automatic Tool Call Parsing
            try:
                tool_parser = tool_parser_cls(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in tool parser creation.")
                raise e
            tool_call_info = tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                # extract_tool_calls() returns a list of tool calls.
                function_calls.extend(
                    FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None
            else:
                # No tool calls.
                return None, content

        return function_calls, content

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
