import json
import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, cast

import openai
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.shared_params import FunctionDefinition
from openai.types.responses.web_search_tool_param import UserLocation
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OpenAIConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    CONF_DISABLE_AUTH,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
    DEFAULT_BASE_URL,
    CONF_USE_RESPONSES_ENDPOINT,
    CONF_USE_CHAT_STREAMING,
    RECOMMENDED_USE_CHAT_STREAMING
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )

def _format_tool_legacy(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)

def _convert_content_to_param(
    content: conversation.Content,
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    if isinstance(content, conversation.ToolResultContent):
        return [
            FunctionCallOutput(
                type="function_call_output",
                call_id=content.tool_call_id,
                output=json.dumps(content.tool_result),
            )
        ]

    if content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "system":
            role = "developer"
        messages.append(
            EasyInputMessageParam(type="message", role=role, content=content.content)
        )

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        messages.extend(
            ResponseFunctionToolCallParam(
                type="function_call",
                name=tool_call.tool_name,
                arguments=json.dumps(tool_call.tool_args),
                call_id=tool_call.id,
            )
            for tool_call in content.tool_calls
        )
    return messages

async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ResponseStreamEvent],
    messages: ResponseInputParam,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    async for event in result:
        LOGGER.debug("Received event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseOutputMessage):
                yield {"role": event.item.role}
            elif isinstance(event.item, ResponseFunctionToolCall):
                current_tool_call = event.item
        elif isinstance(event, ResponseOutputItemDoneEvent):
            item = event.item.model_dump()
            item.pop("status", None)
            if isinstance(event.item, ResponseReasoningItem):
                messages.append(cast(ResponseReasoningItemParam, item))
            elif isinstance(event.item, ResponseOutputMessage):
                messages.append(cast(ResponseOutputMessageParam, item))
            elif isinstance(event.item, ResponseFunctionToolCall):
                messages.append(cast(ResponseFunctionToolCallParam, item))
        elif isinstance(event, ResponseTextDeltaEvent):
            yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            current_tool_call.status = "completed"
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call.call_id,
                        tool_name=current_tool_call.name,
                        tool_args=json.loads(current_tool_call.arguments),
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
        elif isinstance(event, ResponseIncompleteEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )

            if (
                event.response.incomplete_details
                and event.response.incomplete_details.reason
            ):
                reason: str = event.response.incomplete_details.reason
            else:
                reason = "unknown reason"

            if reason == "max_output_tokens":
                reason = "max output tokens reached"
            elif reason == "content_filter":
                reason = "content filter triggered"

            raise HomeAssistantError(f"OpenAI response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            if event.response.usage is not None:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    }
                )
            reason = "unknown reason"
            if event.response.error is not None:
                reason = event.response.error.message
            raise HomeAssistantError(f"OpenAI response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"OpenAI response error: {event.message}")


async def _chat_transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a chat/completions stream into HA format."""
    current_tool_call: dict[str, Any] | None = None

    async for chunk in result:
        LOGGER.debug("Received chat chunk: %s", chunk)
        choice = chunk.choices[0]
        delta = choice.delta

        # 1) Role token comes once, at the very start
        if delta.role is not None:
            yield {"role": delta.role}

        # 2) Text tokens come one by one
        if delta.content is not None:
            yield {"content": delta.content}

        # 3) Function-call arguments chunks
        if delta.function_call is not None:
            # start a new tool call
            if current_tool_call is None:
                current_tool_call = {
                    "id": delta.function_call.id,
                    "tool_name": delta.function_call.name,
                    "tool_args": delta.function_call.arguments or "",
                }
            else:
                # append to the same call
                current_tool_call["tool_args"] += delta.function_call.arguments or ""

        # 4) Finish a tool-call
        if current_tool_call and choice.finish_reason == "function_call":
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["tool_name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }
            current_tool_call = None

        # 5) If the model says “stop”, we’re done
        if choice.finish_reason:
            break


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model="ChatGPT",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ToolParam] | None = None
        if chat_log.llm_api:
            # Format tools differently based on the endpoint type
            if options.get(CONF_USE_RESPONSES_ENDPOINT, True):
                tools = [
                    _format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]
            else:
                tools = [
                    _format_tool_legacy(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]

        if options.get(CONF_WEB_SEARCH):
            web_search = WebSearchToolParam(
                type="web_search_preview",
                search_context_size=options.get(
                    CONF_WEB_SEARCH_CONTEXT_SIZE, RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE
                ),
            )
            if options.get(CONF_WEB_SEARCH_USER_LOCATION):
                web_search["user_location"] = UserLocation(
                    type="approximate",
                    city=options.get(CONF_WEB_SEARCH_CITY, ""),
                    region=options.get(CONF_WEB_SEARCH_REGION, ""),
                    country=options.get(CONF_WEB_SEARCH_COUNTRY, ""),
                    timezone=options.get(CONF_WEB_SEARCH_TIMEZONE, ""),
                )
            if tools is None:
                tools = []
            tools.append(web_search)

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        # Get the custom base URL (from config entry)
        base_url = self.entry.data.get("base_url", DEFAULT_BASE_URL)

        # Decide whether to use the /responses endpoint or /chat/completions
        endpoint = "/responses" if options.get(CONF_USE_RESPONSES_ENDPOINT, True) else "/chat/completions"
        full_url = f"{base_url}{endpoint}"  # Final endpoint

        LOGGER.debug(f"Making API call to: {full_url}")

        # Check if the API_KEY is missing when auth is required
        if not self.entry.data.get(CONF_API_KEY) and not self.entry.options.get(CONF_DISABLE_AUTH, False):
            LOGGER.error("API key is missing or authentication is disabled.")
            raise HomeAssistantError("API key is missing or authentication is disabled.")

        client = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": True,
            }

            if options.get(CONF_USE_RESPONSES_ENDPOINT, True):
                model_args["max_output_tokens"] = options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                )
                model_args["input"] = messages
            else:
                model_args["max_completion_tokens"] = options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                )
                model_args["messages"] = messages

            if options.get(CONF_USE_CHAT_STREAMING, True):
                LOGGER.debug("Using chat streaming.")
                model_args["stream"] = True
            else:
                model_args["stream"] = False

            if tools:
                model_args["tools"] = tools

            if model.startswith("o"):
                model_args["reasoning"] = {
                    "effort": options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    )
                }
            else:
                model_args["store"] = False

            try:
                if options.get(CONF_USE_RESPONSES_ENDPOINT, True):
                    # Use the /responses endpoint
                    result = await client.responses.create(**model_args)
                else:
                    # Use the /chat/completions endpoint
                    result = await client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "I’m temporarily rate-limited by OpenAI. Please try again shortly."
                )
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                    continue_conversation=False,
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                error_message = str(err)  
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, error_message)
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                    continue_conversation=False,
                )
            
            if options.get(CONF_USE_CHAT_STREAMING, True):
                if options.get(CONF_USE_RESPONSES_ENDPOINT, True):
                # existing Responses path
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        _transform_stream(chat_log, result, messages),
                    ):
                        if not isinstance(content, conversation.AssistantContent):
                            messages.extend(_convert_content_to_param(content))
                else:
                # new chat/completions path
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        _chat_transform_stream(chat_log, result),
                    ):
                        # for non-AssistantContent deltas, feed back into messages
                        if not isinstance(content, conversation.AssistantContent):
                            messages.extend(_convert_content_to_param(content))
            else:
                # Handle non-streaming response (directly)
                chat_log.content.append(conversation.AssistantContent(content=result.choices[0].message.content, agent_id=user_input.agent_id))

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        last_content = chat_log.content[-1]

        if isinstance(last_content, conversation.UserContent):
            LOGGER.debug("Waiting for assistant's response...")
            retries = 0
            max_retries = 5
            while retries < max_retries:
                if isinstance(chat_log.content[-1], conversation.AssistantContent):
                    break
        
                LOGGER.debug(f"Attempt {retries + 1}: Waiting for assistant's response...")
                await asyncio.sleep(2)  # Retry interval
                retries += 1

            if retries == max_retries:
                LOGGER.warning("Timeout reached waiting for assistant's response.")
                intent_response.async_set_speech("Sorry, I didn't get a response from the assistant.")

        elif isinstance(last_content, conversation.AssistantContent):
            intent_response.async_set_speech(last_content.content or "")
        else:
            LOGGER.error(f"Unexpected content type: {type(last_content)}")
            raise AssertionError(f"Unexpected content type: {type(last_content)}")
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
