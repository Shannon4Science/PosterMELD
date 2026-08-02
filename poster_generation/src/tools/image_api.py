import io
import os
import requests
import json
import base64
import re
import time
from pathlib import Path
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple


class ImageQuotaError(RuntimeError):
    """Raised when an image provider reports exhausted credit or hard quota."""


class ImageTools:
    """
    图像操作工具类，封装基于 nanobanana/qwen-image 或者 gemini-2.5-flash-image 的视觉能力，
    同时也混合了针对本地底层预处理的基础工具（如 Pillow 的裁剪与缩放）。
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        """
        初始化视觉工具
        :param api_key: 视觉服务 API 密钥
        :param base_url: 视觉服务网关的请求基础路径
        :param model: 默认使用的模型名称
        """
        self.api_key = api_key or os.getenv("IMAGE_API_KEY") or os.getenv("VLM_API_KEY")
        self.base_urls = self._resolve_base_urls(base_url)
        self.base_url = self.base_urls[0] if self.base_urls else ""
        self.models = self._resolve_models(model, fallback_models)
        self.model = self.models[0] if self.models else "gemini-2.5-flash-image"
        self.retry_attempts = max(1, int(retry_attempts if retry_attempts is not None else os.getenv("IMAGE_RETRY_ATTEMPTS", "5")))
        self.retry_delay = max(0.0, float(retry_delay if retry_delay is not None else os.getenv("IMAGE_RETRY_DELAY_SECONDS", "6")))
        self.request_timeout = max(1.0, float(os.getenv("IMAGE_REQUEST_TIMEOUT_SECONDS", "120")))

    def _resolve_models(self, model: Optional[str], fallback_models: Optional[List[str]]) -> List[str]:
        if os.getenv("IMAGE_MODELS"):
            values = self._split_base_urls(os.getenv("IMAGE_MODELS"))
        else:
            primary = model or os.getenv("IMAGE_MODEL") or "gemini-2.5-flash-image"
            values = [primary]
            configured_fallbacks = fallback_models
            if configured_fallbacks is None:
                configured_fallbacks = self._split_base_urls(os.getenv("IMAGE_FALLBACK_MODELS"))
            values.extend(configured_fallbacks or [])
            if primary == "gpt-image-2":
                values.append("gemini-3.1-flash-image-preview")

        models: List[str] = []
        seen = set()
        for value in values:
            value = str(value or "").strip()
            if value and value not in seen:
                models.append(value)
                seen.add(value)
        return models

    def _resolve_base_urls(self, base_url: Optional[str]) -> List[str]:
        if base_url:
            values = [base_url]
        elif os.getenv("IMAGE_BASE_URLS"):
            values = [os.getenv("IMAGE_BASE_URLS")]
        else:
            values = [os.getenv("IMAGE_BASE_URL"), os.getenv("VLM_BASE_URL")]
        urls: List[str] = []
        seen = set()
        for value in values:
            for candidate in self._split_base_urls(value):
                key = candidate.rstrip("/")
                if key and key not in seen:
                    urls.append(key)
                    seen.add(key)
        return urls

    def _split_base_urls(self, value: Optional[str]) -> List[str]:
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value]
        else:
            parts = [part.strip() for part in re.split(r"[\s,;]+", str(value)) if part.strip()]
        return [part.rstrip("/") for part in parts if part]

    def _headers(self, *, json_content: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _require_api_config(self, purpose: str) -> None:
        if not self.api_key or not self.base_urls:
            raise ValueError(f"IMAGE_API_KEY/VLM_API_KEY and IMAGE_BASE_URLS/IMAGE_BASE_URL/VLM_BASE_URL are required for {purpose}")

    def _request_with_failover(self, label: str, operation: Callable[[str], Any]) -> Any:
        errors = []
        for model in self.models:
            self.model = model
            for base_url in self.base_urls:
                for attempt in range(1, self.retry_attempts + 1):
                    try:
                        return operation(base_url)
                    except Exception as exc:
                        errors.append(f"model={model} {base_url} attempt {attempt}/{self.retry_attempts}: {exc}")
                        if self._is_hard_quota_error(exc):
                            message = (
                                f"{label} 检测到生图余额或硬额度错误，立即停止重试，"
                                f"model={model}, base_url={base_url}: {exc}"
                            )
                            print(message)
                            raise ImageQuotaError(message) from exc
                        if self._is_non_retryable_model_error(exc):
                            print(f"{label} 模型/渠道不可用，切换下一个 URL 或模型，model={model}, base_url={base_url}: {exc}")
                            break
                        if attempt < self.retry_attempts:
                            print(f"{label} 调用失败，{self.retry_delay:g}s 后重试 ({attempt}/{self.retry_attempts})，model={model}, base_url={base_url}: {exc}")
                            time.sleep(self.retry_delay)
                        else:
                            print(f"{label} 在 model={model}, base_url={base_url} 已失败 {self.retry_attempts} 次，切换下一个 URL")
        tail = "; ".join(errors[-5:])
        raise RuntimeError(f"{label} failed for all configured base URLs. Last errors: {tail}")

    def _is_hard_quota_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = [
            "insufficient_quota",
            "insufficient quota",
            "quota exceeded",
            "quota exhausted",
            "billing hard limit",
            "insufficient balance",
            "insufficient account balance",
            "balance is insufficient",
            "account balance is insufficient",
            "balance exhausted",
            "credits exhausted",
            "credit balance",
            "exceeded your current quota",
            "exceeded current quota",
            "payment required",
            "402 client error",
            "http 402",
            "余额不足",
            "额度不足",
            "账户欠费",
            "余额已用完",
            "需要充值",
        ]
        return any(marker in text for marker in markers)

    def _is_non_retryable_model_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = [
            "model_not_found",
            "get_channel_failed",
            "no available channel",
            "no access to model",
            "this token has no access",
            "model not found",
            "requested operation is unsupported",
            "operation is unsupported",
            "unsupported operation",
        ]
        return any(marker in text for marker in markers)

    def _is_gpt_image_model(self) -> bool:
        return str(self.model or "").startswith("gpt-image")

    def _raise_for_status_with_body(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = str(getattr(response, "text", "") or "")[:500]
            raise requests.HTTPError(f"{exc}; body={body}") from exc

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, output_path: str = "generated_img.png") -> str:
        """
        使用配置好的模型和 API 获取图像，并保存到本地。
        由于通常的标准接口走的是 OpenAI 风格的 /images/generations 路径：
        Returns:
            生成的图片对应的本地文件路径
        """
        print(f"正在调用图像生成 API (模型: {self.model})，提示词: {prompt}...")

        try:
            self._require_api_config("image generation")
            headers = self._headers()
            try:
                return self._request_with_failover(
                    "images/generations",
                    lambda base_url: self._generate_with_images_endpoint(prompt, width, height, output_path, headers, base_url),
                )
            except ImageQuotaError:
                raise
            except Exception as image_endpoint_error:
                print(f"标准图片接口失败，回退到 chat/completions: {image_endpoint_error}")
                return self._request_with_failover(
                    "chat/completions image generation",
                    lambda base_url: self._generate_with_chat_endpoint(prompt, output_path, headers, base_url),
                )
        except ImageQuotaError:
            raise
        except Exception as e:
            print(f"生成图像失败，可能因为网络或者服务接口变更错误: {e}")
            raise RuntimeError(f"image generation failed: {e}") from e

    def _generate_with_images_endpoint(self, prompt: str, width: int, height: int, output_path: str, headers: dict, base_url: str) -> str:
        url = f"{base_url.rstrip('/')}/images/generations"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": self._request_size(width, height),
        }
        if not self._is_gpt_image_model():
            payload["response_format"] = "b64_json"
        response = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
        self._raise_for_status_with_body(response)
        data = response.json()
        return self._write_image_response(data, output_path)

    def _request_size(self, width: int, height: int) -> str:
        override = os.getenv("IMAGE_SIZE") or os.getenv("IMAGE_REQUEST_SIZE")
        if override:
            return override
        if self._is_gpt_image_model():
            if width > height * 1.2:
                return "1536x1024"
            if height > width * 1.2:
                return "1024x1536"
            return "1024x1024"
        return f"{width}x{height}"

    def _write_image_response(self, data: dict, output_path: str) -> str:
        item = (data.get("data") or [{}])[0]

        if item.get("b64_json"):
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(item["b64_json"]))
            print(f"图像生成成功，路径: {output_path}")
            return output_path

        if item.get("url"):
            image_response = requests.get(item["url"], timeout=self.request_timeout)
            image_response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(image_response.content)
            print(f"图像生成成功，路径: {output_path}")
            return output_path

        raise Exception(f"图片接口返回数据结构异常: {data}")

    def _generate_with_chat_endpoint(self, prompt: str, output_path: str, headers: dict, base_url: str) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
        self._raise_for_status_with_body(response)

        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise Exception(f"API 返回数据结构异常: {data}")

        content = data["choices"][0]["message"]["content"]
        match = re.search(r"data:image/[^;]+;base64,([^)]+)", content)
        if not match:
            raise Exception(f"API 回复中未能提取到合法的 Base64 Markdown 图片标签。返回的原始数据为:\n{content[:200]}...")

        img_bytes = base64.b64decode(match.group(1))
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"图像生成成功，路径: {output_path}")
        return output_path

    def edit_image(self, image_path: str, prompt: str, output_path: str = "edited_img.png") -> str:
        """
        根据提示词，通过视觉 API 编辑目标图像。
        Returns:
            编辑后的图像的本地保存路径
        """
        print(f"正在请求视觉 API 编辑图像 {image_path}，提示词: {prompt}...")
        
        try:
            self._require_api_config("image editing")
            try:
                return self._request_with_failover(
                    "images/edits",
                    lambda base_url: self._edit_with_images_endpoint(image_path, prompt, output_path, base_url),
                )
            except ImageQuotaError:
                raise
            except Exception as image_endpoint_error:
                print(f"标准图片编辑接口失败，回退到 chat/completions: {image_endpoint_error}")
                return self._request_with_failover(
                    "chat/completions image editing",
                    lambda base_url: self._edit_with_chat_endpoint(image_path, prompt, output_path, base_url),
                )
        except ImageQuotaError:
            raise
        except Exception as e:
            print(f"编辑图像服务请求失败，直接返回原图: {e}")
            return image_path

    def _edit_with_images_endpoint(self, image_path: str, prompt: str, output_path: str, base_url: str) -> str:
        url = f"{base_url.rstrip('/')}/images/edits"
        with open(image_path, "rb") as image_file:
            files = {
                "image": (Path(image_path).name, image_file, "image/png"),
            }
            data = {
                "model": self.model,
                "prompt": prompt,
                "size": os.getenv("IMAGE_EDIT_SIZE") or "1024x1024",
            }
            if not self._is_gpt_image_model():
                data["response_format"] = "b64_json"
            response = requests.post(url, headers=self._headers(json_content=False), data=data, files=files, timeout=self.request_timeout)
        self._raise_for_status_with_body(response)
        return self._write_image_response(response.json(), output_path)

    def _edit_with_chat_endpoint(self, image_path: str, prompt: str, output_path: str, base_url: str) -> str:
        headers = self._headers()
        url = f"{base_url.rstrip('/')}/chat/completions"
        # 将原图转为 Base64 以多模态方式发送
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + " 请只输出编辑后的图片结果。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
        self._raise_for_status_with_body(response)

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]

            # 兼容旧逻辑，通过更健壮的正则表达式提取 Base64
            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\n]+)", content)
            if match:
                b64_str = match.group(1)
                img_bytes = base64.b64decode(b64_str)

                with open(output_path, "wb") as f:
                    f.write(img_bytes)
                print(f"图像编辑成功: {output_path}")
                return output_path
            raise Exception(f"未能在结果中提取到 Base64 图像: {content[:200]}...")
        raise Exception(f"API 返回数据结构异常: {data}")

    def crop_and_resize(self, image_path: str, target_width: int, target_height: int, output_path: str) -> str:
        """
        本地预处理后备方案：强行裁剪和按比例缩放图像，确保其绝对符合布局中的 (宽度 x 高度)
        保持了图片的纵横比不变，通过中间裁剪居中处理对齐。
        """
        with Image.open(image_path) as img:
            img_aspect = img.width / img.height
            target_aspect = target_width / target_height
            
            if img_aspect > target_aspect:
                # 原图片比目标的要“扁宽”，所以砍掉两边的宽
                new_width = int(img.height * target_aspect)
                left = (img.width - new_width) / 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # 原图片比目标的要“高瘦”，所以砍掉上下的高
                new_height = int(img.width / target_aspect)
                top = (img.height - new_height) / 2
                img = img.crop((0, top, img.width, top + new_height))
                
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            img.save(output_path)
            
        return output_path

    def fit_and_resize(
        self,
        image_path: str,
        target_width: int,
        target_height: int,
        output_path: str,
        background: Tuple[int, int, int] = (255, 255, 255),
    ) -> str:
        """
        Resize the full image into the target box without cropping.

        This is the safer default for tables because row labels, column headers,
        and border rules are semantically important and should not be center-cropped.
        """
        with Image.open(image_path) as img:
            if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
                canvas = Image.new("RGBA", img.size, (*background, 255))
                img = Image.alpha_composite(canvas, img.convert("RGBA")).convert("RGB")
            else:
                img = img.convert("RGB")

            scale = min(target_width / img.width, target_height / img.height)
            resized_width = max(1, int(round(img.width * scale)))
            resized_height = max(1, int(round(img.height * scale)))
            resized = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

            canvas = Image.new("RGB", (target_width, target_height), background)
            left = max((target_width - resized_width) // 2, 0)
            top = max((target_height - resized_height) // 2, 0)
            canvas.paste(resized, (left, top))
            canvas.save(output_path)

        return output_path
