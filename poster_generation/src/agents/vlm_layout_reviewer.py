"""
VLM-based poster screenshot review with bounded layout patching.

The VLM is only allowed to diagnose visual issues and propose small geometry
patches. Deterministic validation decides whether a patch can be applied.
"""

import base64
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import json_repair
import requests

from src.agents.micro_layout_refiner import MicroLayoutRefiner
from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class VLMLayoutReviewer:
    def __init__(self):
        self.name = "vlm_layout_reviewer"
        self.config = load_config()
        self.review_config = self.config.get("vlm_layout_review", {})
        self._last_usage = {"input_tokens": 0, "output_tokens": 0}

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_vlm_layout_review", False):
            return state

        log_agent_info(self.name, "reviewing rendered poster screenshot with VLM")
        state["vlm_reflow_required"] = False
        state["vlm_patch_applied"] = False
        state["template_repair_required"] = False

        try:
            review = self._review_or_fallback(state)
            if review.get("degraded"):
                state.setdefault("degraded_quality_states", []).append(
                    {
                        "component": self.name,
                        "category": "vlm_layout_review",
                        "reason": "; ".join(str(item) for item in review.get("warnings", [])),
                        "fallback": "deterministic_acceptance",
                    }
                )
            fast_mode = bool(state.get("template_fast_mode"))
            if not fast_mode:
                review = self._enforce_template_acceptance_gate(review, state)
            patch = self._extract_patch(review)
            state["vlm_layout_review"] = review
            state["vlm_layout_patch"] = patch

            if fast_mode:
                if patch:
                    review.setdefault("warnings", []).append(
                        "Fast template-first mode recorded the VLM patch but did not apply automatic global relayout."
                    )
                if self._has_high_overflow(review) and state.get("template_repair_count", 0) < int(self.review_config.get("template_prior_max_repairs", 1)):
                    state["template_repair_required"] = True
                    state["template_repair_decision"] = {
                        "source": self.name,
                        "reason": "High overflow reported by VLM in fast template-first mode.",
                        "review": review,
                    }
                state["vlm_layout_review"] = review
                self._save_outputs(state)
                state["current_agent"] = self.name
                log_agent_success(self.name, "VLM layout review completed in fast report-only mode")
                return state

            max_iterations = int(self.review_config.get("max_iterations", 1))
            if patch and state.get("vlm_review_count", 0) < max_iterations:
                patched_layout = self._apply_safe_patch(state.get("styled_layout") or [], patch, state)
                if patched_layout:
                    state["styled_layout"] = patched_layout
                    state["vlm_review_count"] = state.get("vlm_review_count", 0) + 1
                    state["vlm_reflow_required"] = True
                    state["vlm_patch_applied"] = True
                    self._save_styled_layout(state)
                    log_agent_success(self.name, f"applied {len(patch)} safe VLM layout patch(es)")
                else:
                    review.setdefault("warnings", []).append("VLM patch rejected by deterministic safety validation.")
                    log_agent_warning(self.name, "VLM patch rejected by deterministic safety validation")

            review = self._accept_after_max_template_repair(review, state)
            state["vlm_layout_review"] = review

            if state.get("template_layout_mode") == "template_prior" and not review.get("accept", True):
                if not state.get("vlm_patch_applied", False):
                    state["template_repair_required"] = True
                    state["template_repair_decision"] = {
                        "source": self.name,
                        "reason": self._review_reason(review),
                        "review": review,
                    }

            self._save_outputs(state)
            state["current_agent"] = self.name
            log_agent_success(self.name, "VLM layout review completed")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _has_high_overflow(self, review: Dict[str, Any]) -> bool:
        issues = review.get("issues") if isinstance(review.get("issues"), list) else []
        return any(
            str(issue.get("severity", "")).lower() == "high"
            and str(issue.get("category", "")).lower() == "overflow"
            for issue in issues
            if isinstance(issue, dict)
        )

    def _accept_after_max_template_repair(self, review: Dict[str, Any], state: PosterState) -> Dict[str, Any]:
        if review.get("degraded") or review.get("review_available") is False:
            return review
        if state.get("template_layout_mode") != "template_prior" or review.get("accept", True):
            return review
        if not bool(self.review_config.get("template_prior_accept_after_max_repair", True)):
            return review

        max_repairs = int(self.review_config.get("template_prior_max_repairs", 1))
        if state.get("template_repair_count", 0) < max_repairs:
            return review

        micro_report = self._load_content_json(state, "micro_layout_report.json")
        validation_issues = ((micro_report.get("validation") or {}).get("issues") or [])
        if validation_issues:
            return review

        accepted = dict(review)
        accepted["accept"] = True
        accepted.setdefault("warnings", [])
        accepted["warnings"].append(
            "Template-prior review accepted after max repair count because deterministic micro-layout validation has no issues."
        )
        return accepted

    def _enforce_template_acceptance_gate(self, review: Dict[str, Any], state: PosterState) -> Dict[str, Any]:
        if state.get("template_layout_mode") != "template_prior":
            return review
        if review.get("degraded") or review.get("review_available") is False:
            unavailable = dict(review)
            unavailable["accept"] = False
            return unavailable

        min_score = int(self.review_config.get("template_prior_min_accept_score", 82))
        issues = review.get("issues") if isinstance(review.get("issues"), list) else []
        score = review.get("overall_score")
        high_whitespace = any(
            str(issue.get("severity", "")).lower() == "high"
            and str(issue.get("category", "")).lower() == "whitespace"
            for issue in issues
        )
        high_visual_asset = any(
            str(issue.get("severity", "")).lower() == "high"
            and str(issue.get("category", "")).lower() == "visual_asset"
            for issue in issues
        )
        high_overflow = any(
            str(issue.get("severity", "")).lower() == "high"
            and str(issue.get("category", "")).lower() == "overflow"
            for issue in issues
        )
        unresolved_whitespace = self._has_unresolved_template_whitespace(issues, state)
        too_low = isinstance(score, (int, float)) and score < min_score

        if not (too_low or high_visual_asset or high_overflow or unresolved_whitespace):
            accepted = dict(review)
            accepted["accept"] = True
            accepted.setdefault("warnings", [])
            if review.get("accept") is False:
                accepted["warnings"].append(
                    f"Template-prior review accepted after gate normalization: score={score}, no high-risk issues."
                )
            return accepted

        gated = dict(review)
        gated["accept"] = False
        gated.setdefault("warnings", [])
        gated["warnings"].append(
            f"Template-prior hard gate rejected this poster: score={score}, "
            f"high_whitespace={high_whitespace}, high_visual_asset={high_visual_asset}, "
            f"high_overflow={high_overflow}, unresolved_whitespace={unresolved_whitespace}."
        )
        gated.setdefault("patch", [])
        return gated

    def _has_unresolved_template_whitespace(self, issues: List[Dict[str, Any]], state: PosterState) -> bool:
        if state.get("template_layout_mode") != "template_prior":
            return False
        block_settings = self.config.get("block_refinement", {})
        acceptable_min = float(block_settings.get("acceptable_min", 0.90))
        micro_report = self._load_content_json(state, "micro_layout_report.json")
        low_lanes = {
            str(lane.get("lane_id"))
            for lane in micro_report.get("lanes", [])
            if float(lane.get("final_utilization") or 0.0) < acceptable_min
        }
        if not low_lanes:
            return False

        section_to_lane = {}
        for element in state.get("styled_layout") or []:
            if element.get("type") == "section_container" and element.get("section_id"):
                section_to_lane[str(element.get("section_id"))] = str(element.get("lane_id") or element.get("slot_id") or "")

        for issue in issues:
            severity = str(issue.get("severity", "")).lower()
            category = str(issue.get("category", "")).lower()
            if severity not in {"medium", "high"} or category != "whitespace":
                continue
            target = str(issue.get("target") or issue.get("target_section") or issue.get("slot_id") or "")
            if target in low_lanes:
                return True
            if section_to_lane.get(target) in low_lanes:
                return True
        return False

    def _review_or_fallback(self, state: PosterState) -> Dict[str, Any]:
        preview_path = state.get("poster_preview_path")
        if not preview_path or not Path(preview_path).exists():
            return self._fallback_review("poster preview PNG is unavailable; skipping VLM review")

        base_url = os.getenv("VLM_BASE_URL")
        api_key = os.getenv("VLM_API_KEY")
        model = state.get("vlm_model") or os.getenv("VLM_MODEL")
        if not base_url or not api_key or not model:
            return self._fallback_review("VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL are required for VLM review")

        prompt = self._build_prompt(state)
        image_data = self._encode_image(preview_path)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            content = self._request_vlm_text(base_url, headers, model, prompt, image_data)
            self._record_usage(state, self.name)
            review = self._parse_json(content)
            review.setdefault("source", "vlm")
            review.setdefault("review_available", True)
            review.setdefault("degraded", False)
            review.setdefault("patch", [])
            review.setdefault("warnings", [])
            return review
        except Exception as exc:
            return self._fallback_review(f"VLM layout request failed ({exc}); using deterministic acceptance fallback")

    def _post_vlm_request(
        self,
        base_url: str,
        headers: Dict[str, str],
        model: str,
        prompt: str,
        image_data: str,
        *,
        transport: Optional[str] = None,
    ) -> requests.Response:
        timeout = int(self.review_config.get("timeout_seconds", 120))
        endpoint = base_url.rstrip("/")
        # gpt-5 / o-series reasoning models only accept the default temperature (1);
        # some deployments reject any other value with 400 "operation not allowed" or a
        # failed streamed response, so omit temperature entirely for those models.
        is_reasoning_model = str(model).startswith(("gpt-5", "o1", "o3", "o4"))
        temperature = self.review_config.get("temperature", 0.1)
        if endpoint.endswith("/responses"):
            root_endpoint = endpoint[: -len("/responses")]
            inferred_transport = "responses"
        elif endpoint.endswith("/chat/completions"):
            root_endpoint = endpoint[: -len("/chat/completions")]
            inferred_transport = "chat"
        else:
            root_endpoint = endpoint
            inferred_transport = "responses" if str(model).startswith("gpt-5") else "chat"
        selected_transport = transport or inferred_transport

        if selected_transport == "responses":
            response_endpoint = f"{root_endpoint}/responses"
            payload = {
                "model": model,
                "store": False,
                "stream": True,
                "max_output_tokens": self.review_config.get("max_tokens", 1600),
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data},
                        ],
                    }
                ],
            }
            if not is_reasoning_model:
                payload["temperature"] = temperature
            return requests.post(response_endpoint, headers=headers, json=payload, timeout=timeout, stream=True)

        payload = {
            "model": model,
            "max_tokens": self.review_config.get("max_tokens", 1600),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ],
        }
        if not is_reasoning_model:
            payload["temperature"] = temperature
        return requests.post(f"{root_endpoint}/chat/completions", headers=headers, json=payload, timeout=timeout)

    def _request_vlm_text(self, base_url: str, headers: Dict[str, str], model: str, prompt: str, image_data: str) -> str:
        """Post a VLM request and return its text, retrying transient failures.

        The relay endpoints are flaky and occasionally return a failed streamed
        response or a 5xx even though the model is reachable, so retry a few
        times before letting the caller fall back to the deterministic path.
        """
        self._last_usage = {"input_tokens": 0, "output_tokens": 0}
        attempts = max(int(self.review_config.get("request_attempts", 3)), 1)
        retry_delay = float(self.review_config.get("retry_delay_seconds", 2.0))
        last_exc: Optional[Exception] = None
        endpoint = base_url.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            transports = ["chat"]
        elif endpoint.endswith("/responses"):
            transports = ["responses", "chat"]
        elif str(model).startswith("gpt-5"):
            transports = ["responses", "chat"]
        else:
            transports = ["chat"]

        for attempt in range(attempts):
            for transport in transports:
                try:
                    response = self._post_vlm_request(
                        base_url,
                        headers,
                        model,
                        prompt,
                        image_data,
                        transport=transport,
                    )
                    response.raise_for_status()
                    return self._extract_response_text(response)
                except Exception as exc:  # noqa: BLE001 - retry any transient VLM failure
                    last_exc = exc
            if attempt < attempts - 1:
                time.sleep(min(retry_delay * (attempt + 1), 6.0))
        raise last_exc if last_exc else RuntimeError("VLM request failed with no exception")

    def _extract_response_text(self, response: requests.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._extract_stream_text(response)

        data = response.json()
        self._capture_usage(data.get("usage"))
        if data.get("output_text"):
            return data["output_text"]

        if data.get("choices"):
            return data["choices"][0]["message"]["content"]

        output_chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    output_chunks.append(text)
        if output_chunks:
            return "\n".join(output_chunks)

        raise ValueError(f"unsupported VLM response schema: {list(data.keys())}")

    def _extract_stream_text(self, response: requests.Response) -> str:
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
            if event_type in {"response.completed", "response.done"}:
                self._capture_usage((event.get("response") or {}).get("usage") or event.get("usage"))
            if event_type == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif event_type == "response.output_text.done":
                done_text = event.get("text") or done_text
            elif event_type == "response.failed":
                raise ValueError(event.get("response", {}).get("error") or event)
        text = done_text or "".join(chunks)
        if not text:
            raise ValueError("VLM stream completed without text output")
        return text

    def _capture_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        self._last_usage = {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }

    def _record_usage(self, state: PosterState, agent_name: str) -> None:
        usage = dict(self._last_usage)
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        if input_tokens <= 0 and output_tokens <= 0:
            return
        state["timing_metrics"].add_api_call(agent_name, "vision", input_tokens, output_tokens)
        state["tokens"].add_vision(input_tokens, output_tokens)
        self._last_usage = {"input_tokens": 0, "output_tokens": 0}

    def _build_prompt(self, state: PosterState) -> str:
        layout_summary = self._layout_summary(state.get("styled_layout") or [])
        micro_report = self._load_content_json(state, "micro_layout_report.json")
        resolved_assets = state.get("resolved_visual_assets") or {}
        is_template_prior = state.get("template_layout_mode") == "template_prior"
        patch_ops = (
            "move|move_up|move_down|increase_visual_scale|decrease_visual_scale|decrease_font_size"
            if is_template_prior
            else "move|move_up|move_down|increase_visual_scale|decrease_visual_scale|increase_font_size|decrease_font_size"
        )
        return f"""
You are reviewing a generated academic poster screenshot.

Goal:
- Identify visual layout problems visible to a human reviewer.
- Propose only safe, small geometry patches. Do not rewrite scientific content.

Check these issues:
- overlap or overflow
- excessive whitespace or unbalanced columns
- cramped elements
- weak visual hierarchy
- unclear reading flow
- images too small/large for their section
- mismatched visual emphasis
- title readability, including whether the title is visibly larger than body blocks
- major whitespace regions that make any template block look unfinished

Return strict JSON only:
{{
  "overall_score": 0-100,
  "accept": true/false,
  "global_assessment": {{
    "title_readability": "ok|too_small|crowded|unclear",
    "layout_balance": "ok|left_heavy|right_heavy|top_heavy|bottom_heavy|fragmented",
    "reading_order": "ok|unclear",
    "major_whitespace_regions": [
      {{
        "slot_id": "slot id or section id",
        "severity": "low|medium|high",
        "description": "short diagnosis"
      }}
    ],
    "visual_hierarchy": "ok|weak|confusing"
  }},
  "issues": [
    {{
      "severity": "low|medium|high",
      "category": "overlap|overflow|whitespace|hierarchy|visual_asset|reading_flow|style",
      "target": "element id, slot id, visual id, or section id",
      "description": "short diagnosis"
    }}
  ],
  "patch": [
    {{
      "target": "element id, slot id, visual id, or section id",
      "op": "{patch_ops}",
      "value": 0.1,
      "dx": 0.0,
      "dy": 0.0,
      "reason": "why this small patch helps"
    }}
  ],
  "visual_asset_recommendations": [
    {{
      "slot_id": "optional slot id",
      "action": "keep|crop_only|edit|generate_new",
      "reason": "short reason"
    }}
  ]
}}

Patch constraints:
- Only propose small adjustments.
- For move/move_up/move_down, use inches and keep absolute movement <= 0.4.
- For visual scale, use factors between 0.9 and 1.15.
- For font changes, use point deltas between 1 and 4.
- Never propose deleting sections or changing scientific data.
- If the template prior clearly does not suit the current content distribution, set accept=false and leave patch=[].

Layout summary:
{json.dumps(layout_summary, indent=2)}

Micro-layout validation report:
{json.dumps(micro_report, indent=2)}

Resolved visual assets:
{json.dumps(resolved_assets, indent=2)}
""".strip()

    def _layout_summary(self, layout: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for element in layout:
            summary.append({
                "id": element.get("id"),
                "type": element.get("type"),
                "section_id": element.get("section_id"),
                "slot_id": element.get("slot_id"),
                "visual_id": element.get("visual_id"),
                "lane_id": element.get("lane_id"),
                "x": element.get("x"),
                "y": element.get("y"),
                "width": element.get("width"),
                "height": element.get("height"),
                "font_size": element.get("font_size"),
            })
        return summary

    def _extract_patch(self, review: Dict[str, Any]) -> List[Dict[str, Any]]:
        patch = review.get("patch") or []
        return patch if isinstance(patch, list) else []

    def _apply_safe_patch(self, layout: List[Dict[str, Any]], patch: List[Dict[str, Any]], state: PosterState) -> Optional[List[Dict[str, Any]]]:
        patched = deepcopy(layout)
        applied_count = 0
        for operation in patch:
            candidate = deepcopy(patched)
            operation_count = self._apply_operation(candidate, operation, state)
            if operation_count == 0:
                continue
            self._sync_section_containers(candidate)
            if not self._validate_geometry(candidate, state):
                continue
            patched = candidate
            applied_count += operation_count

        if applied_count == 0:
            return None

        self._sync_section_containers(patched)
        return patched

    def _apply_operation(self, layout: List[Dict[str, Any]], operation: Dict[str, Any], state: PosterState) -> int:
        op = str(operation.get("op", "")).strip()
        target = str(operation.get("target", "")).strip()
        if not op or not target:
            return 0

        targets = self._find_targets(layout, target, op)
        if not targets:
            return 0

        if op == "move":
            dx = self._clamp(float(operation.get("dx", 0.0)), -self._max_move(), self._max_move())
            dy = self._clamp(float(operation.get("dy", 0.0)), -self._max_move(), self._max_move())
            return self._move_elements(targets, dx, dy, state)
        if op in {"move_up", "move_down"}:
            value = self._clamp(float(operation.get("value", 0.15)), 0.0, self._max_move())
            dy = -value if op == "move_up" else value
            return self._move_elements(targets, 0.0, dy, state)
        if op in {"increase_visual_scale", "decrease_visual_scale"}:
            factor = float(operation.get("value", 1.08))
            if op == "decrease_visual_scale" and factor > 1:
                factor = 1 / factor
            factor = self._clamp(factor, 0.9, 1.15)
            return self._scale_visuals(targets, factor, state)
        if op in {"increase_font_size", "decrease_font_size"}:
            value = float(operation.get("value", 2.0))
            delta = self._clamp(value, 1.0, 4.0)
            if op == "decrease_font_size":
                delta = -delta
            return self._change_font_size(targets, delta)
        return 0

    def _find_targets(self, layout: List[Dict[str, Any]], target: str, op: str) -> List[Dict[str, Any]]:
        direct = [
            element for element in layout
            if target in {
                str(element.get("id", "")),
                str(element.get("slot_id", "")),
                str(element.get("visual_id", "")),
                str(element.get("section_id", "")),
            }
        ]
        if direct:
            if op.startswith("move") and any(element.get("type") == "section_container" for element in direct):
                section_ids = {element.get("section_id") for element in direct if element.get("section_id")}
                return [
                    element for element in layout
                    if element.get("section_id") in section_ids or any(str(element.get("id", "")).startswith(f"{sid}_") for sid in section_ids)
                ]
            return direct

        prefix_matches = [element for element in layout if str(element.get("id", "")).startswith(f"{target}_")]
        return prefix_matches

    def _move_elements(self, elements: List[Dict[str, Any]], dx: float, dy: float, state: PosterState) -> int:
        applied = 0
        for element in elements:
            new_x = float(element.get("x", 0.0)) + dx
            new_y = float(element.get("y", 0.0)) + dy
            if self._inside_slide(new_x, new_y, float(element.get("width", 0.0)), float(element.get("height", 0.0)), state):
                element["x"] = new_x
                element["y"] = new_y
                applied += 1
        return applied

    def _scale_visuals(self, elements: List[Dict[str, Any]], factor: float, state: PosterState) -> int:
        applied = 0
        for element in elements:
            if element.get("type") != "visual":
                continue
            old_w = float(element.get("width", 0.0))
            old_h = float(element.get("height", 0.0))
            new_w = old_w * factor
            new_h = old_h * factor
            new_x = float(element.get("x", 0.0)) - (new_w - old_w) / 2
            new_y = float(element.get("y", 0.0)) - (new_h - old_h) / 2
            if self._inside_slide(new_x, new_y, new_w, new_h, state):
                element["x"] = new_x
                element["y"] = new_y
                element["width"] = new_w
                element["height"] = new_h
                applied += 1
        return applied

    def _change_font_size(self, elements: List[Dict[str, Any]], delta: float) -> int:
        applied = 0
        min_font = int(self.review_config.get("min_font_size", 24))
        max_font = int(self.review_config.get("max_font_size", 120))
        for element in elements:
            if "font_size" not in element:
                continue
            new_size = self._clamp(float(element.get("font_size", 0.0)) + delta, min_font, max_font)
            element["font_size"] = new_size
            applied += 1
        return applied

    def _validate_geometry(self, layout: List[Dict[str, Any]], state: PosterState) -> bool:
        refiner = MicroLayoutRefiner()
        template_layout = state.get("layout_template_metadata") or refiner._resolve_template_layout(state)
        lane_map = {lane["id"]: lane for lane in template_layout["lanes"]}
        validation = refiner._validate_refined_layout(layout, lane_map, state)
        return not validation.get("issues")

    def _sync_section_containers(self, layout: List[Dict[str, Any]]):
        containers = {
            element.get("section_id"): element
            for element in layout
            if element.get("type") == "section_container" and element.get("section_id")
        }
        if not containers:
            return

        for section_id, container in containers.items():
            children = [
                element for element in layout
                if element.get("type") != "section_container"
                and (
                    element.get("section_id") == section_id
                    or str(element.get("id") or element.get("slot_id") or "").startswith(f"{section_id}_")
                )
            ]
            if not children:
                continue
            bottom = max(float(child.get("y", 0.0)) + float(child.get("height", 0.0)) for child in children)
            container["height"] = max(
                float(container.get("height", 0.0)),
                bottom - float(container.get("y", 0.0)) + self.review_config.get("container_bottom_padding", 0.2),
            )

    def _inside_slide(self, x: float, y: float, width: float, height: float, state: PosterState) -> bool:
        return (
            x >= 0
            and y >= 0
            and x + width <= float(state["poster_width"]) + 1e-6
            and y + height <= float(state["poster_height"]) + 1e-6
        )

    def _fallback_review(self, warning: str) -> Dict[str, Any]:
        log_agent_warning(self.name, warning)
        return {
            "source": "fallback",
            "review_available": False,
            "degraded": True,
            "fallback": "deterministic_acceptance",
            "overall_score": None,
            "accept": False,
            "issues": [],
            "patch": [],
            "visual_asset_recommendations": [],
            "warnings": [warning],
        }

    def _review_reason(self, review: Dict[str, Any]) -> str:
        issues = review.get("issues") or []
        if issues:
            return str(issues[0].get("description") or "VLM rejected the draft poster.")
        warnings = review.get("warnings") or []
        if warnings:
            return str(warnings[0])
        return "VLM rejected the draft poster."

    def _encode_image(self, image_path: str) -> str:
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"

    def _parse_json(self, content: str) -> Dict[str, Any]:
        start = content.find("```json")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            content = content[start + 7:end].strip()
        return json_repair.loads(content)

    def _load_content_json(self, state: PosterState, filename: str) -> Dict[str, Any]:
        path = Path(state["output_dir"]) / "content" / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_styled_layout(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "styled_layout.json", "w", encoding="utf-8") as f:
            json.dump(state.get("styled_layout", []), f, indent=2)

    def _save_outputs(self, state: PosterState):
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "vlm_layout_review.json", "w", encoding="utf-8") as f:
            json.dump(state.get("vlm_layout_review", {}), f, indent=2)
        with open(output_dir / "vlm_layout_patch.json", "w", encoding="utf-8") as f:
            json.dump(state.get("vlm_layout_patch", []), f, indent=2)

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _max_move(self) -> float:
        return float(self.review_config.get("max_move_inches", 0.4))


def vlm_layout_reviewer_node(state: PosterState) -> Dict[str, Any]:
    result = VLMLayoutReviewer()(state)
    return {
        **state,
        "styled_layout": result.get("styled_layout"),
        "vlm_layout_review": result.get("vlm_layout_review"),
        "vlm_layout_patch": result.get("vlm_layout_patch"),
        "vlm_review_count": result.get("vlm_review_count", 0),
        "vlm_reflow_required": result.get("vlm_reflow_required", False),
        "vlm_patch_applied": result.get("vlm_patch_applied", False),
        "template_repair_required": result.get("template_repair_required", False),
        "template_repair_decision": result.get("template_repair_decision"),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
