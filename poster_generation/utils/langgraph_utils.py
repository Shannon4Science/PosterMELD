"""LangGraph utilities"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json
import json_repair

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.state.poster_state import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
if not ENV_PATH.exists():
    ENV_PATH = PROJECT_ROOT.parent / ".env"
load_dotenv(ENV_PATH, override=True)


def _status_code_from_exception(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def is_non_retryable_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    # Multi-channel relay/proxy deployments intermittently route a request to an
    # unavailable upstream channel and return a 400/500 such as "operation not
    # allowed in this deployment" or "channel ... does not exist / 可用渠道不存在".
    # These are transient (a retry usually lands on a healthy channel), so they must
    # stay retryable even though they carry a 4xx status code.
    transient_relay_tokens = (
        "operation is not allowed in this deployment",
        "operation not allowed in this deployment",
        "get_channel_failed",
        "channel_failed",
        "可用渠道不存在",
        "渠道",
    )
    if any(token in message for token in transient_relay_tokens):
        return False
    status_code = _status_code_from_exception(exc)
    if status_code in {400, 401, 403, 404}:
        return True
    return any(
        token in message
        for token in (
            "authentication",
            "unauthorized",
            "invalid api key",
            "incorrect api key",
            "api key",
            "permission denied",
            "forbidden",
            "model not found",
            "无效的令牌",
        )
    )


def is_retryable_model_error(exc: Exception) -> bool:
    return not is_non_retryable_model_error(exc)


def create_model(config: ModelConfig):
    """create chat model from config"""
    # common timeout settings for all providers
    timeout_settings = {
        'request_timeout': 500,  # 2 minutes for request timeout
        'max_retries': 2,        # reduce retries at model level since we have tenacity
    }
    
    if config.provider == 'openai':
        # gpt-5 / o-series reasoning models only accept the default temperature (1);
        # sending any other value is rejected (OpenAI errors, and some relay
        # deployments return 400 "operation not allowed in this deployment").
        is_reasoning_model = str(config.model_name).startswith(("gpt-5", "o1", "o3", "o4"))
        openai_kwargs = {
            'model_name': config.model_name,
            'temperature': 1 if is_reasoning_model else config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'request_timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
            # Stream the completion. Reasoning models can spend 30-40s before the
            # first token, and non-streaming requests get killed by relay/proxy
            # gateway timeouts (~30s) with a 502; streaming keeps the connection
            # alive with incremental chunks so heavy calls complete.
            'streaming': True,
            'stream_usage': True,
        }
        base_url = os.getenv('OPENAI_BASE_URL')
        if base_url:
            openai_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**openai_kwargs)
    elif config.provider == 'anthropic':
        anthropic_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ANTHROPIC_BASE_URL')
        if base_url:
            anthropic_kwargs['base_url'] = base_url
            
        return ChatAnthropic(**anthropic_kwargs)
    elif config.provider == 'google':
        google_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_output_tokens': config.max_tokens,
            'google_api_key': os.getenv('GOOGLE_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('GOOGLE_BASE_URL')
        if base_url:
            google_kwargs['base_url'] = base_url
            
        return ChatGoogleGenerativeAI(**google_kwargs)
    elif config.provider == 'zhipu':
        zhipu_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ZHIPU_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ZHIPU_BASE_URL')
        if base_url:
            zhipu_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**zhipu_kwargs)
    elif config.provider == 'moonshot':
        moonshot_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('MOONSHOT_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('MOONSHOT_BASE_URL')
        if base_url:
            moonshot_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**moonshot_kwargs)
    elif config.provider == 'Minimax':
        minimax_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('MINIMAX_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('MINIMAX_BASE_URL')
        if base_url:
            minimax_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**minimax_kwargs)
    elif config.provider == 'Alibaba':
        alibaba_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ALIBABA_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ALIBABA_BASE_URL')
        if base_url:
            alibaba_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**alibaba_kwargs)
    elif config.provider == 'openai_responses':
        return ResponsesAPIModel(config)
    else:
        raise ValueError(f"unsupported provider: {config.provider}")


class ResponsesAPIModel:
    """Minimal OpenAI Responses API adapter for the existing LangGraphAgent wrapper."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.endpoint = (
            os.getenv("OPENAI_RESPONSES_BASE_URL")
            or os.getenv("VLM_BASE_URL")
            or ""
        ).rstrip("/")
        self.api_key = os.getenv("OPENAI_RESPONSES_API_KEY") or os.getenv("VLM_API_KEY")
        if self.endpoint and not self.endpoint.endswith("/responses"):
            self.endpoint = f"{self.endpoint}/responses"

    def invoke(self, messages):
        if not self.endpoint or not self.api_key:
            raise ValueError("OPENAI_RESPONSES_BASE_URL/VLM_BASE_URL and OPENAI_RESPONSES_API_KEY/VLM_API_KEY are required")

        import requests
        from langchain_core.messages import AIMessage

        payload = {
            "model": self.config.model_name,
            "store": False,
            "stream": True,
            "max_output_tokens": self.config.max_tokens,
            "input": [self._convert_message(message) for message in messages],
        }
        # reasoning models (gpt-5 / o-series) only accept the default temperature
        if not str(self.config.model_name).startswith(("gpt-5", "o1", "o3", "o4")):
            payload["temperature"] = self.config.temperature
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=500,
            stream=True,
        )
        response.raise_for_status()
        return AIMessage(content=self._extract_stream_text(response))

    def _convert_message(self, message):
        role = "user"
        if isinstance(message, SystemMessage):
            role = "system"

        content = message.content
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "input_text", "text": content}]}

        converted = []
        for item in content:
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                converted.append({"type": "input_image", "image_url": image_url.get("url", "")})
        return {"role": role, "content": converted}

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if data.get("output_text"):
            return data["output_text"]

        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)

        raise ValueError(f"unsupported Responses API schema: {list(data.keys())}")

    def _extract_stream_text(self, response) -> str:
        chunks = []
        done_text = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            raw = line.split("data:", 1)[1].strip()
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif event_type == "response.output_text.done":
                done_text = event.get("text") or done_text
            elif event_type == "response.failed":
                raise ValueError(event.get("response", {}).get("error") or event)
        text = done_text or "".join(chunks)
        if not text:
            raise ValueError("Responses API stream completed without text output")
        return text


class LangGraphAgent:
    """langgraph agent wrapper"""

    def __init__(self, system_msg: str, config: ModelConfig, state=None, agent_name: str = "unknown"):
        self.system_msg = system_msg
        self.config = config
        self.model = create_model(config)
        self.history = [SystemMessage(content=system_msg)]
        self.state = state
        self.agent_name = agent_name

    def reset(self):
        """reset conversation"""
        self.history = [SystemMessage(content=self.system_msg)]
    
    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_model_error),
    )
    def step(self, message: str) -> 'AgentResponse':
        """process message and return response"""
        # check if message is json with image data
        try:
            msg_data = json.loads(message)
            if isinstance(msg_data, list) and any("image_url" in item for item in msg_data):
                # vision model call
                return self._step_vision(msg_data)
        except:
            pass
        
        # regular text call
        self.history.append(HumanMessage(content=message))
        
        # keep conversation window
        if len(self.history) > 10:
            self.history = [self.history[0]] + self.history[-9:]
        
        # get response with token tracking
        input_tokens, output_tokens = 0, 0
        try:
            if self.config.provider in ('openai', 'zhipu'):
                with get_openai_callback() as cb:
                    response = self.model.invoke(self.history)
                    input_tokens = cb.prompt_tokens or 0
                    output_tokens = cb.completion_tokens or 0
            else:
                response = self.model.invoke(self.history)
                # estimate tokens for non-openai
                input_tokens = len(message.split()) * 1.3
                output_tokens = len(response.content.split()) * 1.3
        except Exception as e:
            error_msg = f"model call failed: {e}"
            print(error_msg)
            
            # provide more specific error information
            if "timeout" in str(e).lower() or "read operation timed out" in str(e).lower():
                print(f"⚠️  Timeout error detected for {self.config.provider} {self.config.model_name}")
                print("💡 Possible solutions:")
                print("   - Check your internet connection")
                print("   - Verify API key is valid")
                print("   - Try using a different model provider")
                print("   - Consider increasing timeout settings")
            elif "rate limit" in str(e).lower():
                print(f"⚠️  Rate limit exceeded for {self.config.provider}")
                print("💡 Consider adding delays between requests")
            elif is_non_retryable_model_error(e):
                print(f"⚠️  Authentication error for {self.config.provider}")
                print("💡 Check your API key configuration")
            
            input_tokens = len(message.split()) * 1.3
            output_tokens = 100
            raise
        
        self.history.append(response)

        if self.state is not None and hasattr(self.state.get('timing_metrics'), 'add_api_call'):
            self.state['timing_metrics'].add_api_call(self.agent_name, 'text', int(input_tokens), int(output_tokens))

        return AgentResponse(response.content, input_tokens, output_tokens)
    
    def _step_vision(self, messages: List[Dict]) -> 'AgentResponse':
        """handle vision model calls"""
        # convert to proper format
        content = []
        for msg in messages:
            if msg.get("type") == "text":
                content.append({"type": "text", "text": msg["text"]})
            elif msg.get("type") == "image_url":
                content.append({
                    "type": "image_url",
                    "image_url": msg["image_url"]
                })
        
        human_msg = HumanMessage(content=content)
        
        # get response
        input_tokens, output_tokens = 0, 0
        try:
            if self.config.provider in ('openai', 'zhipu'):
                with get_openai_callback() as cb:
                    response = self.model.invoke([self.history[0], human_msg])
                    input_tokens = cb.prompt_tokens or 0
                    output_tokens = cb.completion_tokens or 0
            else:
                response = self.model.invoke([self.history[0], human_msg])
                # estimate tokens
                input_tokens = 200  # rough estimate for image
                output_tokens = len(response.content.split()) * 1.3
        except Exception as e:
            error_msg = f"vision model call failed: {e}"
            print(error_msg)
            
            # provide more specific error information for vision calls
            if "timeout" in str(e).lower() or "read operation timed out" in str(e).lower():
                print(f"⚠️  Vision timeout error detected for {self.config.provider} {self.config.model_name}")
                print("💡 Vision calls may take longer due to image processing")
                print("   - Consider using a different vision model")
                print("   - Check image size and format")
            elif "rate limit" in str(e).lower():
                print(f"⚠️  Rate limit exceeded for vision calls on {self.config.provider}")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                print(f"⚠️  Authentication error for vision calls on {self.config.provider}")


            raise

        if self.state is not None and hasattr(self.state.get('timing_metrics'), 'add_api_call'):
            self.state['timing_metrics'].add_api_call(self.agent_name, 'vision', int(input_tokens), int(output_tokens))

        return AgentResponse(response.content, input_tokens, output_tokens)


class AgentResponse:
    """agent response with token tracking"""
    def __init__(self, content: str, input_tokens: int, output_tokens: int):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def extract_json(response: str) -> Dict[str, Any]:
    """extract json from model response"""
    
    # find json code block
    start = response.find("```json")
    end = response.rfind("```")
    
    if start != -1 and end != -1 and end > start:
        json_content = response[start + 7:end].strip()
    else:
        json_content = response.strip()
    
    try:
        return json_repair.loads(json_content)
    except Exception as e:
        raise ValueError(f"failed to parse json: {e}")


def load_prompt(path: str) -> str:
    """load prompt template from file"""
    prompt_path = Path(path)
    if not prompt_path.is_absolute() and not prompt_path.exists():
        prompt_path = Path(__file__).resolve().parents[1] / prompt_path
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()
