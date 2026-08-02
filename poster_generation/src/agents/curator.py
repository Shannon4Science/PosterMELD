"""
spatial content planning and story board curation
"""

import json
import math
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config
from src.layout.template_selector import TemplateSelector
from src.tools.layout_api import LayoutTemplates
from src.template_extraction.block_template_registry import get_block_template_info, is_block_template_id
from src.utils.text_cleanup import normalize_text_for_poster, repair_possessive_title_apostrophe
from src.utils.visual_footprint import visual_slot_is_feasible
from jinja2 import Template

class StoryBoardCurator:
    """creates spatial content plan and story board"""
    
    def __init__(self):
        self.name = "spatial_content_planner"
        self.spatial_planning_prompt = load_prompt("config/prompts/spatial_content_planner.txt")
        self.config = load_config()
        self.validation_config = self.config["validation"]
        self.utilization_config = self.config["utilization_thresholds"]

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "creating spatial content plan")
        
        try:
            structured_sections = state.get("structured_sections")
            narrative_content = state.get("narrative_content")
            classified_visuals = state.get("classified_visuals")

            if not structured_sections:
                log_agent_error(self.name, "missing structured_sections from parser")
                raise ValueError("missing structured_sections from parser")
            if not narrative_content:
                log_agent_error(self.name, "missing narrative_content from parser")
                raise ValueError("missing narrative_content from parser")
            if not classified_visuals:
                log_agent_error(self.name, "missing classified_visuals from parser")
                raise ValueError("missing classified_visuals from parser")
            
            # prepare visual height context for spatial planning
            selection_report = self._resolve_layout_template(
                state,
                structured_sections,
                classified_visuals,
            )
            resolved_template = selection_report["selected_template"]
            self._apply_selected_template_state_defaults(state, selection_report)
            visual_context = self._prepare_visual_context_for_curator(state, resolved_template)
            
            story_board, inp, out = self._create_story_board(
                structured_sections, narrative_content, classified_visuals,
                state.get("visual_assets", {}),
                visual_context, state["text_model"], state
            )
            state["tokens"].add_text(inp, out)
            
            # validate height distribution
            validation_result = self._validate_height_distribution(story_board, visual_context)
            if validation_result["warnings"]:
                log_agent_warning(self.name, f"height validation warnings: {validation_result['warnings']}")
            log_agent_info(self.name, f"column utilizations: {validation_result['column_utilizations']}")
            
            state["story_board"] = story_board
            state["resolved_layout_template"] = resolved_template
            state["layout_template_metadata"] = visual_context["template_layout"]
            state["template_selection_report"] = selection_report
            state["current_agent"] = self.name
            
            self._save_story_board(state)
            
            # log story board summary
            sections = story_board.get("spatial_content_plan", {}).get("sections", [])
            total_visuals = sum(len(section.get("visual_assets", [])) for section in sections)
            
            log_agent_success(self.name, f"created story board with {len(sections)} sections")
            log_agent_success(self.name, f"selected {total_visuals} visual assets")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _create_story_board(self, structured_sections, narrative_content, classified_visuals, visual_assets, visual_context, config, state):
        log_agent_info(self.name, "generating spatial content plan")
        agent = LangGraphAgent("expert spatial poster designer", config, state, "curator")

        images = {
            asset_id.replace("figure_", ""): {
                "caption": asset.get("caption", ""),
                "aspect": asset.get("aspect", 1.0),
            }
            for asset_id, asset in visual_assets.items()
            if asset.get("asset_type") == "figure"
        }
        tables = {
            asset_id.replace("table_", ""): {
                "caption": asset.get("caption", ""),
                "aspect": asset.get("aspect", 1.0),
            }
            for asset_id, asset in visual_assets.items()
            if asset.get("asset_type") == "table"
        }
        
        template_data = {
            "structured_sections": json.dumps(structured_sections, indent=2),
            "narrative_content": json.dumps(narrative_content, indent=2),
            "classified_visuals": json.dumps(classified_visuals, indent=2),
            "paper_poster_keypoints": json.dumps(state.get("paper_poster_keypoints") or [], indent=2, ensure_ascii=False),
            "poster_reading_order": json.dumps(state.get("poster_reading_order") or [], indent=2),
            "available_images": json.dumps({k: {"caption": v.get("caption", ""), "aspect": v.get("aspect", 1.0)} 
                                          for k, v in images.items()}, indent=2),
            "available_tables": json.dumps({k: {"caption": v.get("caption", ""), "aspect": v.get("aspect", 1.0)} 
                                          for k, v in tables.items()}, indent=2),
            "available_height_per_column": visual_context["available_height_per_column"],
            "visual_heights_info": json.dumps(visual_context["visual_assets_heights"], indent=2),
            "template_layout_guidance": self._template_layout_guidance(visual_context),
            "section_count_guidance": self._section_count_guidance(visual_context),
        }
        
        max_attempts = self.validation_config["max_llm_attempts"]
        for attempt in range(max_attempts):
            try:
                prompt = Template(self.spatial_planning_prompt).render(**template_data)
                agent.reset()
                response = agent.step(prompt)
                
                story_board = self._coerce_story_board_payload(extract_json(response.content))
                self._remove_unknown_visual_references(story_board, visual_context)
                self._align_sections_to_keypoints(story_board, state, visual_context, classified_visuals)
                self._compact_portrait_text_content(story_board, visual_context)
                
                if self._validate_story_board(story_board, classified_visuals, visual_context):
                    log_agent_success(self.name, f"successfully created story board on attempt {attempt + 1}")
                    return story_board, response.input_tokens, response.output_tokens
                else:
                    log_agent_warning(self.name, f"attempt {attempt + 1}: validation failed, retrying")
                    
            except Exception as e:
                log_agent_warning(
                    self.name,
                    f"story board attempt {attempt + 1} failed: {e}\n{traceback.format_exc(limit=3)}",
                )
                if int((visual_context or {}).get("keypoint_target_count") or 0):
                    log_agent_warning(self.name, "using deterministic keypoint-first story board fallback")
                    story_board = self._fallback_story_board_from_keypoints(state, visual_context, classified_visuals)
                    return story_board, 0, 0
        # All attempts exhausted (repeated validation failures and/or transient
        # endpoint errors). Fall back to a cached valid story board from a previous
        # run, then to the deterministic keypoint-first plan, before failing hard —
        # this keeps the pipeline resilient to a flaky text endpoint.
        cached = self._load_cached_story_board(state)
        if cached is not None and self._validate_story_board(cached, classified_visuals, visual_context):
            log_agent_warning(self.name, "using cached story board after repeated generation failures")
            self._record_degraded_state(
                state,
                component="curator",
                category="story_board",
                reason="story board generation failed repeatedly; reused cached story_board.json",
                fallback="cached_story_board",
            )
            return cached, 0, 0
        if int((visual_context or {}).get("keypoint_target_count") or 0):
            log_agent_warning(self.name, "using deterministic keypoint-first story board fallback")
            self._record_degraded_state(
                state,
                component="curator",
                category="story_board",
                reason="story board generation failed repeatedly; used deterministic keypoint-first fallback",
                fallback="keypoint_first_story_board",
            )
            story_board = self._fallback_story_board_from_keypoints(state, visual_context, classified_visuals)
            return story_board, 0, 0
        raise ValueError("failed to create story board")

    def _coerce_story_board_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            nested = extract_json(payload)
            if isinstance(nested, dict):
                return nested
            if isinstance(nested, list):
                return {"spatial_content_plan": {"sections": nested}}
        if isinstance(payload, list):
            return {"spatial_content_plan": {"sections": payload}}
        raise ValueError(f"invalid story_board payload type: {type(payload).__name__}")

    def _fallback_story_board_from_keypoints(
        self,
        state: PosterState,
        visual_context: Dict[str, Any],
        classified_visuals: Dict[str, Any],
    ) -> Dict[str, Any]:
        story_board: Dict[str, Any] = {
            "spatial_content_plan": {
                "poster_strategy": {
                    "narrative_flow": "Keypoint-first reading order from problem to method to results.",
                    "space_utilization_approach": (
                        "Keypoints are grouped only when the selected template has fewer high-quality content panels; "
                        "capacity planner expands or compresses each block."
                    ),
                    "column_balance_rationale": "Template block geometry controls final placement.",
                },
                "sections": [],
            },
            "column_distribution": {
                "left_column": {"focus": "Problem and motivation", "assigned_sections": [], "content_strategy": "Introduce context."},
                "middle_column": {"focus": "Method", "assigned_sections": [], "content_strategy": "Explain the framework."},
                "right_column": {"focus": "Results", "assigned_sections": [], "content_strategy": "Show evidence and takeaway."},
            },
        }
        self._align_sections_to_keypoints(story_board, state, visual_context, classified_visuals)
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        distribution_keys = {"left": "left_column", "middle": "middle_column", "right": "right_column"}
        for section in sections:
            column = distribution_keys.get(section.get("column_assignment"), "middle_column")
            story_board["column_distribution"][column]["assigned_sections"].append(section.get("section_id"))
        return story_board

    def _remove_unknown_visual_references(self, story_board: Dict, visual_context: Dict[str, Any]) -> None:
        valid_visual_ids = set((visual_context or {}).get("valid_visual_ids") or [])
        if not valid_visual_ids:
            return
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        for section in sections:
            visuals = section.get("visual_assets", [])
            if not isinstance(visuals, list):
                section["visual_assets"] = []
                continue
            normalized_visuals = [self._normalize_visual_reference(visual) for visual in visuals]
            normalized_visuals = [visual for visual in normalized_visuals if visual]
            kept = [visual for visual in normalized_visuals if visual.get("visual_id") in valid_visual_ids]
            removed = [
                visual.get("visual_id")
                for visual in normalized_visuals
                if visual.get("visual_id") not in valid_visual_ids
            ]
            if removed:
                log_agent_warning(
                    self.name,
                    f"removed unknown visual references from {section.get('section_id')}: {removed}",
                )
            section["visual_assets"] = kept

    def _compact_portrait_text_content(self, story_board: Dict, visual_context: Dict[str, Any]) -> None:
        if not self._is_portrait_or_vertical_template(visual_context):
            return
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        for section in sections:
            text_items = section.get("text_content", [])
            if not isinstance(text_items, list):
                continue
            compacted = []
            for item in text_items:
                text = str(item).strip()
                if not text:
                    continue
                if len(text) > 170:
                    text = text[:167].rstrip(" ,;:") + "."
                compacted.append(text)
                if len(compacted) >= 2:
                    break
            section["text_content"] = compacted

    def _validate_story_board(self, story_board: Dict, classified_visuals: Dict = None, visual_context: Dict = None) -> bool:
        """validate story board structure and constraints"""
        if "spatial_content_plan" not in story_board:
            log_agent_warning(self.name, "validation error: missing 'spatial_content_plan'")
            return False
        
        scp = story_board["spatial_content_plan"]
        
        # check sections
        if "sections" not in scp or not isinstance(scp["sections"], list):
            log_agent_warning(self.name, "validation error: missing or invalid 'sections'")
            return False
        
        sections = scp["sections"]
        min_sections = self.validation_config["min_section_count"]
        max_sections = self._max_section_count_for_template(visual_context)
        keypoint_target = int((visual_context or {}).get("keypoint_target_count") or 0)
        if keypoint_target:
            if self._group_keypoints_for_template(visual_context):
                grouped_target = self._keypoint_section_target_count(visual_context)
                min_sections = grouped_target
                max_sections = grouped_target
            else:
                min_sections = keypoint_target
                max_sections = keypoint_target
        if len(sections) < min_sections or len(sections) > max_sections:
            log_agent_warning(self.name, f"validation error: need {min_sections}-{max_sections} sections, got {len(sections)}")
            return False
        
        # validate each section
        for i, section in enumerate(sections):
            required_fields = ["section_id", "section_title", "column_assignment", "vertical_priority", "text_content"]
            for field in required_fields:
                if field not in section:
                    log_agent_warning(self.name, f"validation error: section {i} missing '{field}'")
                    return False
            if keypoint_target:
                if self._group_keypoints_for_template(visual_context):
                    source_keypoint_ids = section.get("source_keypoint_ids") or []
                    if not isinstance(source_keypoint_ids, list) or not source_keypoint_ids:
                        log_agent_warning(self.name, f"validation error: grouped keypoint section {i} missing source_keypoint_ids")
                        return False
                elif not section.get("keypoint_id"):
                    log_agent_warning(self.name, f"validation error: keypoint section {i} missing keypoint_id")
                    return False
            
            # check column assignment is valid
            if section["column_assignment"] not in ["left", "middle", "right"]:
                log_agent_warning(self.name, f"validation error: section {i} invalid column_assignment")
                return False
                
            # check vertical priority is valid  
            if section["vertical_priority"] not in ["top", "middle", "bottom"]:
                log_agent_warning(self.name, f"validation error: section {i} invalid vertical_priority")
                return False
            
            # check section title length (4 words max)
            title = section.get("section_title", "")
            title_words = len(title.split())
            max_words = self.validation_config["max_title_words"]
            if title_words > max_words:
                log_agent_warning(self.name, f"validation error: section {i} title too long ({title_words} words): '{title}'")
                return False
            
            # check text content is list of bullet points
            min_items = self._minimum_text_items_for_section(section)
            if not isinstance(section["text_content"], list) or len(section["text_content"]) < min_items:
                log_agent_warning(self.name, f"validation error: section {i} invalid text_content")
                return False
            
            # check for ellipsis in text content
            for j, text in enumerate(section["text_content"]):
                if "..." in text:
                    log_agent_warning(self.name, f"validation error: section {i} bullet {j} contains ellipsis")
                    return False

            valid_visual_ids = set((visual_context or {}).get("valid_visual_ids") or [])
            for visual in section.get("visual_assets", []):
                visual = self._normalize_visual_reference(visual)
                if not visual:
                    continue
                visual_id = visual.get("visual_id")
                if valid_visual_ids and visual_id not in valid_visual_ids:
                    log_agent_warning(self.name, f"validation error: section {i} references unknown visual_id '{visual_id}'")
                    return False
        
        # validate key_visual placement if classified_visuals provided
        if classified_visuals:
            key_visual = classified_visuals.get("key_visual")
            if key_visual:
                key_visual_found = False
                key_visual_in_middle_top = False
                
                for section in sections:
                    visual_assets = section.get("visual_assets", [])
                    for visual in visual_assets:
                        visual = self._normalize_visual_reference(visual)
                        if not visual:
                            continue
                        if visual.get("visual_id") == key_visual:
                            key_visual_found = True
                            if (section.get("column_assignment") == "middle" and 
                                section.get("vertical_priority") == "top"):
                                key_visual_in_middle_top = True
                            break
                    if key_visual_found:
                        break

                requested_template = str((visual_context or {}).get("requested_layout_template") or "")
                if (
                    key_visual_found
                    and self._group_keypoints_for_template(visual_context or {})
                    and is_block_template_id(requested_template)
                ):
                    key_visual_in_middle_top = True
                
                if not key_visual_found:
                    log_agent_warning(self.name, f"validation error: key_visual '{key_visual}' not found in any section")
                    return False
                    
                if not key_visual_in_middle_top:
                    log_agent_warning(self.name, f"validation error: key_visual '{key_visual}' not placed in required primary visual position")
                    return False

        if self._is_portrait_or_vertical_template(visual_context or {}):
            selected_visual_ids = [
                normalized.get("visual_id")
                for section in sections
                for normalized in [self._normalize_visual_reference(visual) for visual in section.get("visual_assets", [])]
                if normalized and normalized.get("visual_id")
            ]
            max_visuals = self._max_visuals_for_context(visual_context or {})
            if len(selected_visual_ids) > max_visuals:
                log_agent_warning(
                    self.name,
                    f"validation error: portrait template allows only {max_visuals} total visuals, got {selected_visual_ids}",
                )
                return False
        
        # validate height exclusion compliance if visual_context provided
        if visual_context:
            visual_heights = visual_context.get("visual_assets_heights", {})
            max_visual_height_percentage = self._max_visual_height_percentage(visual_context)
            oversized_visuals = []
            
            # check all visual assets in the story board
            for section in sections:
                visual_assets = section.get("visual_assets", [])
                for visual in visual_assets:
                    visual = self._normalize_visual_reference(visual)
                    if not visual:
                        continue
                    visual_id = visual.get("visual_id")
                    if visual_id in visual_heights:
                        height_info = visual_heights[visual_id]
                        # extract percentage value from string like "91%"
                        height_str = height_info.get("height_percentage", "0%")
                        height_percentage = float(height_str.rstrip('%'))
                        
                        if height_percentage > max_visual_height_percentage:
                            oversized_visuals.append(f"{visual_id} ({height_str})")
            
            if oversized_visuals:
                # check if only one oversized visual is selected
                if len(oversized_visuals) == 1:
                    # only one oversized visual selected, allow it as fallback
                    log_agent_info(self.name, f"fallback applied: allowing single oversized visual: {oversized_visuals[0]}")
                else:
                    # multiple oversized visuals selected, only allow the smallest
                    selected_oversized = []
                    for section in sections:
                        visual_assets = section.get("visual_assets", [])
                        for visual in visual_assets:
                            visual = self._normalize_visual_reference(visual)
                            if not visual:
                                continue
                            visual_id = visual.get("visual_id")
                            if visual_id in visual_heights:
                                height_info = visual_heights[visual_id]
                                height_str = height_info.get("height_percentage", "0%")
                                height_percentage = float(height_str.rstrip('%'))
                                if height_percentage > max_visual_height_percentage:
                                    selected_oversized.append((visual_id, height_percentage, height_str))
                    
                    smallest = min(selected_oversized, key=lambda x: x[1])
                    invalid_visuals = [f"{vid} ({h_str})" for vid, h, h_str in selected_oversized if vid != smallest[0]]
                    log_agent_warning(self.name, f"validation error: oversized visuals (>{max_visual_height_percentage:.0f}% height) selected: {invalid_visuals} (fallback: only smallest allowed: {smallest[0]} ({smallest[2]}))")
                    return False
        
        return True

    def _minimum_text_items_for_section(self, section: Dict[str, Any]) -> int:
        default_min = int(self.validation_config["min_text_content_items"])
        budget = section.get("capacity_budget") or {}
        target_bullets = int(section.get("target_bullets") or budget.get("target_bullets") or 0)
        visual_policy = str(budget.get("visual_policy") or "").lower()
        if target_bullets == 1 or "figure" in visual_policy or "table" in visual_policy:
            return 1
        return default_min

    def _section_count_guidance(self, visual_context: Dict[str, Any]) -> str:
        keypoint_target = int((visual_context or {}).get("keypoint_target_count") or 0)
        if keypoint_target:
            if self._group_keypoints_for_template(visual_context):
                grouped_target = self._keypoint_section_target_count(visual_context)
                return (
                    f"exactly {grouped_target} grouped poster sections; use Poster Keypoints as a content pool "
                    "and merge adjacent or related keypoints when needed by template geometry"
                )
            return (
                f"exactly {keypoint_target} keypoint-aligned sections; create one section per "
                "paper_poster_keypoint in poster_reading_order"
            )
        if self._is_portrait_or_vertical_template(visual_context):
            return "exactly 5 compact"
        return "5-8"

    def _template_layout_guidance(self, visual_context: Dict[str, Any]) -> str:
        template_layout = visual_context.get("template_layout") or {}
        keypoint_target = int((visual_context or {}).get("keypoint_target_count") or 0)
        if (visual_context or {}).get("template_fast_mode"):
            contract = (visual_context or {}).get("fast_block_contract") or {}
            visual_policy = (visual_context or {}).get("fast_visual_policy") or {}
            blocks = [
                {
                    "slot_id": block.get("slot_id"),
                    "role": block.get("slot_role"),
                    "target_chars": block.get("target_chars"),
                    "min_chars": block.get("min_chars"),
                    "max_chars": block.get("max_chars"),
                    "visual_policy": block.get("visual_policy"),
                    "keypoint_ids": block.get("source_keypoint_ids"),
                }
                for block in contract.get("blocks") or []
            ]
            section_count = len(blocks) or self._keypoint_section_target_count(visual_context)
            if self._is_portrait_or_vertical_template(visual_context):
                figure_count = int(visual_policy.get("figure_count") or 1)
                table_count = int(visual_policy.get("table_count") or 0)
                max_visuals = int(visual_policy.get("max_visuals_total") or max(1, figure_count + table_count))
                return (
                    f"FAST PORTRAIT TEMPLATE-FIRST MODE for {visual_context.get('requested_layout_template')}. "
                    f"Produce exactly {section_count} grouped poster sections from the keypoint pool. Preserve every "
                    f"keypoint id in source_keypoint_ids. Use up to {max_visuals} total visuals: prioritize one key "
                    f"method figure, then up to {table_count} readable result table/chart, then remaining method/result "
                    f"figures up to the {figure_count}-figure budget. Convert only unreadable or overflow-prone tables "
                    "and ablations into short factual text. "
                    "Match text_content length to each slot's min/target/max chars while keeping the portrait layout readable. "
                    f"Visual policy: {json.dumps(visual_policy, ensure_ascii=False)}. "
                    f"Slot contracts: {json.dumps(blocks, ensure_ascii=False)}"
                )
            figure_count = int(visual_policy.get("figure_count") or 2)
            table_count = int(visual_policy.get("table_count") or 1)
            return (
                f"FAST TEMPLATE-FIRST MODE for {visual_context.get('requested_layout_template')}. "
                f"Produce exactly {section_count} grouped poster sections from 10 keypoints. Preserve every keypoint id in "
                "source_keypoint_ids. Use the fixed slot plan below; do not create extra sections and do not "
                f"drop keypoints. Select up to {figure_count} figures and up to {table_count} tables overall when "
                "the selected visuals are readable and the slot contracts have capacity; for dense result papers, "
                "2 figures plus 2 tables is allowed. If a table is unreadable, keep a text summary and record the "
                "warning. Match text_content length to each slot's min/target/max "
                "chars. Text formatting must be clean and uniform: no literal bullet symbols, no nested bullets, "
                "no ordered lists, no empty strings, and no Table/Figure number references. Text-only slots should "
                "use 2-4 parallel callouts; figure/table slots should use 1-2 short interpretation lines. "
                f"Visual policy: {json.dumps(visual_policy, ensure_ascii=False)}. "
                f"Slot contracts: {json.dumps(blocks, ensure_ascii=False)}"
            )
        if keypoint_target:
            if self._group_keypoints_for_template(visual_context):
                grouped_target = self._keypoint_section_target_count(visual_context)
                return (
                    f"Grouped keypoint template mode for {visual_context.get('requested_layout_template')}. "
                    f"Produce exactly {grouped_target} clean poster sections from the keypoint pool. "
                    "Do not force one keypoint per block; group related keypoints into coherent sections, "
                    "preserve source_keypoint_ids, and keep visuals only for the strongest method/results blocks."
                )
            return (
                f"Keypoint-first block template mode. Produce exactly {keypoint_target} fine-grained sections, "
                "one per paper_poster_keypoint, preserving poster_reading_order. Do not merge keypoints into "
                "fewer broad sections. Include keypoint_id and source_section on every section. Use short titles "
                "and 1-3 factual bullets from the matching keypoint/paper facts. Use at most a few visuals overall, "
                "preferably for the central method and strongest result blocks."
            )
        if self._is_portrait_or_vertical_template(visual_context):
            return (
                "Portrait extracted template. Use exactly 5 compact sections across the three vertical bands. "
                "Use 1-2 total visuals: the key visual plus the most readable result table or chart when it fits. "
                "Convert secondary tables and ablations into short text bullets. Avoid splitting one "
                "idea into multiple small sections."
            )
        if template_layout.get("extracted_template"):
            return (
                "Extracted landscape template. Treat template panels as soft style guidance. "
                "Use normal three-lane flow and avoid overfilling any lane with decorative panels."
            )
        return "Standard three-lane landscape poster."

    def _max_section_count_for_template(self, visual_context: Dict[str, Any] = None) -> int:
        keypoint_target = int((visual_context or {}).get("keypoint_target_count") or 0)
        if keypoint_target:
            return keypoint_target
        if self._is_portrait_or_vertical_template(visual_context or {}):
            return 5
        return self.validation_config["max_section_count"]

    def _is_portrait_or_vertical_template(self, visual_context: Dict[str, Any]) -> bool:
        template_layout = (visual_context or {}).get("template_layout") or {}
        lanes = template_layout.get("lanes") or []
        is_vertical_stack = bool(lanes) and len({round(lane.get("x", 0), 3) for lane in lanes}) == 1
        return template_layout.get("orientation") == "portrait" or is_vertical_stack

    def _max_visuals_for_context(self, visual_context: Dict[str, Any]) -> int:
        visual_policy = (visual_context or {}).get("fast_visual_policy") or {}
        if visual_policy.get("max_visuals_total") is not None:
            try:
                return max(1, int(visual_policy.get("max_visuals_total")))
            except (TypeError, ValueError):
                pass
        return 2 if self._is_portrait_or_vertical_template(visual_context or {}) else 4

    def _max_visual_height_percentage(self, visual_context: Dict[str, Any]) -> float:
        template_layout = visual_context.get("template_layout") or {}
        template_name = template_layout.get("template_name", "")
        orientation = template_layout.get("orientation")
        lanes = template_layout.get("lanes") or []
        is_vertical_stack = bool(lanes) and len({round(lane.get("x", 0), 3) for lane in lanes}) == 1
        if orientation == "portrait" or is_vertical_stack:
            return 130.0
        if template_name.startswith("extracted_"):
            return 85.0
        return 50.0

    def _resolve_layout_template(
        self,
        state: PosterState,
        structured_sections: Dict[str, Any],
        classified_visuals: Dict[str, Any],
    ) -> Dict[str, Any]:
        if state.get("resolved_layout_template"):
            selected_template = str(state["resolved_layout_template"])
            selection_report = dict(state.get("template_selection_report") or {})
            selection_report.setdefault("selected_template", selected_template)
            selection_report.setdefault("selection_mode", "preselected")
            selection_report.setdefault("candidates", [])
            log_agent_info(self.name, f"template selection ({selection_report.get('selection_mode')}): {selected_template}")
            return selection_report

        selector = TemplateSelector(self.config)
        selection_report = selector.select(
            state=state,
            structured_sections=structured_sections,
            classified_visuals=classified_visuals,
            visual_assets=state.get("visual_assets", {}),
        )

        selected_template = selection_report["selected_template"]
        mode = selection_report.get("selection_mode", "auto")
        log_agent_info(self.name, f"template selection ({mode}): {selected_template}")
        if selection_report.get("candidates"):
            log_agent_info(
                self.name,
                "template candidates: "
                + ", ".join(
                    f"{candidate['template_name']}={candidate['score']:.2f}"
                    for candidate in selection_report["candidates"]
                ),
        )
        return selection_report

    def _apply_selected_template_state_defaults(self, state: PosterState, selection_report: Dict[str, Any]) -> None:
        selected_template = selection_report.get("selected_template")
        if not selected_template or not is_block_template_id(selected_template):
            return

        state["enable_visual_legibility_review"] = True
        state["enable_vlm_layout_review"] = True
        state["enable_block_vlm_review"] = True

        if state.get("layout_template") != "auto":
            return

        info = get_block_template_info(selected_template) or {}
        size = info.get("recommended_canvas_size") or {}
        try:
            width = float(size.get("width"))
            height = float(size.get("height"))
        except (TypeError, ValueError):
            return
        if width > 0 and height > 0:
            state["poster_width"] = width
            state["poster_height"] = height
            selection_report["auto_canvas_size"] = {"width": width, "height": height}
            log_agent_info(self.name, f"auto canvas adjusted to {width:g}\" x {height:g}\" for {selected_template}")

    def _prepare_visual_context_for_curator(self, state: PosterState, resolved_template: str) -> Dict[str, Any]:
        """prepare visual assets height information for curator's spatial planning"""
        config = load_config()
        
        # get poster dimensions
        poster_width = state["poster_width"] 
        poster_height = state["poster_height"]
        
        # calculate available height per column (18% of effective height for title region)
        poster_margins = 2 * config["layout"]["poster_margin"]
        effective_height = poster_height - poster_margins  # effective height after margins
        title_region_height = effective_height * config["layout"]["title_height_fraction"]  # 18% fixed region
        curator_template = "three_column_postergen" if is_block_template_id(resolved_template) else resolved_template
        layout = LayoutTemplates(
            poster_width,
            poster_height,
            margin=config["layout"]["poster_margin"],
            col_gap=config["layout"]["column_spacing"],
        ).get_template(curator_template, header_height=title_region_height)
        template_layout_for_context = (
            state.get("layout_template_metadata")
            if is_block_template_id(resolved_template) and state.get("layout_template_metadata")
            else layout
        )
        lanes = layout["lanes"]

        min_lane_height = min(lane["h"] for lane in lanes)
        min_lane_width = min(lane["w"] for lane in lanes)
        visual_width_cap = layout.get("visual_width_cap")

        # account for text padding within each lane
        text_padding = 2 * config["layout"]["text_padding"]["left_right"]
        effective_width = min_lane_width - text_padding
        if visual_width_cap:
            effective_width = min(effective_width, visual_width_cap)
        
        log_agent_info(
            self.name,
            f"visual context: template={resolved_template}, available_height={min_lane_height:.1f}\", effective_width={effective_width:.1f}\"",
        )
        
        visual_heights = {}
        
        for asset_id, asset in (state.get("visual_assets") or {}).items():
            aspect_ratio = asset.get("aspect", 1.0) or 1.0
            lane_estimates = {}
            for lane in lanes:
                lane_effective_width = max(lane["w"] - text_padding, 0.1)
                if visual_width_cap:
                    lane_effective_width = min(lane_effective_width, visual_width_cap)
                lane_visual_height = lane_effective_width / aspect_ratio
                lane_estimates[lane["id"]] = {
                    "height_inches": round(lane_visual_height, 1),
                    "height_percentage": f"{((lane_visual_height / lane['h']) * 100):.0f}%",
                }

            visual_height = effective_width / aspect_ratio
            height_percentage = (visual_height / min_lane_height) * 100
            
            visual_heights[asset_id] = {
                "height_inches": round(visual_height, 1),
                "height_percentage": f"{height_percentage:.0f}%",
                "type": asset.get("asset_type", "figure"), 
                "aspect_ratio": aspect_ratio,
                "lane_estimates": lane_estimates,
            }
            log_agent_info(self.name, f"{asset_id}: {visual_height:.1f}\" ({height_percentage:.0f}% of column)")
        
        return {
            "available_height_per_column": round(min_lane_height, 1),
            "visual_assets_heights": visual_heights,
            "column_width": round(min_lane_width, 1),
            "effective_width": round(effective_width, 1),
            "template_layout": template_layout_for_context,
            "valid_visual_ids": list((state.get("visual_assets") or {}).keys()),
            "requested_layout_template": resolved_template,
            "visual_density": state.get("visual_density"),
            "template_fast_mode": bool(state.get("template_fast_mode")),
            "fast_block_contract": state.get("fast_block_contract") or {},
            "fast_visual_policy": state.get("fast_visual_policy") or {},
            "visual_assets": state.get("visual_assets") or {},
            "paper_poster_keypoints": state.get("paper_poster_keypoints") or [],
            "poster_reading_order": state.get("poster_reading_order") or [],
            "keypoint_target_count": min(len(state.get("paper_poster_keypoints") or []), 10)
            if state.get("paper_poster_keypoints") else 0,
            "keypoint_section_target_count": self._default_keypoint_section_target(resolved_template, template_layout_for_context, state),
            "keypoint_grouping_mode": self._template_uses_grouped_keypoints(resolved_template),
        }

    def _align_sections_to_keypoints(
        self,
        story_board: Dict[str, Any],
        state: PosterState,
        visual_context: Dict[str, Any],
        classified_visuals: Dict[str, Any],
    ) -> None:
        keypoints = self._ordered_keypoints(state)
        if not keypoints:
            return
        if self._group_keypoints_for_template(visual_context):
            self._align_grouped_sections_to_keypoints(
                story_board,
                keypoints,
                visual_context,
                classified_visuals,
            )
            return

        scp = story_board.setdefault("spatial_content_plan", {})
        original_sections = scp.get("sections") or []
        if not isinstance(original_sections, list):
            original_sections = []

        by_keypoint_id: Dict[int, Dict[str, Any]] = {}
        remaining = []
        for section in original_sections:
            if not isinstance(section, dict):
                continue
            try:
                keypoint_id = int(section.get("keypoint_id"))
            except (TypeError, ValueError):
                remaining.append(section)
                continue
            by_keypoint_id[keypoint_id] = section

        aligned = []
        for index, keypoint in enumerate(keypoints):
            keypoint_id = int(keypoint["id"])
            source = by_keypoint_id.get(keypoint_id) or (remaining[index] if index < len(remaining) else {})
            section = deepcopy(source)
            keypoint_text = normalize_text_for_poster(str(keypoint.get("key_point") or "").strip())
            source_section = normalize_text_for_poster(str(keypoint.get("section") or "Paper").strip()) or "Paper"

            section["keypoint_id"] = keypoint_id
            section["source_section"] = source_section
            section["source_sections"] = list(section.get("source_sections") or [source_section])
            section["section_id"] = str(section.get("section_id") or f"keypoint_{keypoint_id}")
            title = normalize_text_for_poster(str(section.get("section_title") or "").strip())
            if not title or len(title.split()) > self.validation_config["max_title_words"]:
                title = self._short_title_for_keypoint(keypoint_text, source_section)
            section["section_title"] = self._clean_section_title(title)
            section["column_assignment"] = section.get("column_assignment") if section.get("column_assignment") in {"left", "middle", "right"} else self._keypoint_column(source_section, title, index, len(keypoints))
            section["vertical_priority"] = section.get("vertical_priority") if section.get("vertical_priority") in {"top", "middle", "bottom"} else self._keypoint_vertical_priority(index, len(keypoints))
            section["importance_level"] = int(section.get("importance_level") or (1 if index == 0 else 2 if index < 5 else 3))
            section["content_type"] = section.get("content_type") or self._keypoint_content_type(source_section, title)
            section["expected_content_density"] = section.get("expected_content_density") or "medium"
            section["text_content"] = self._keypoint_text_content(section.get("text_content"), keypoint_text)
            section["visual_assets"] = self._valid_visual_assets(section.get("visual_assets"), visual_context)
            section["spatial_rationale"] = section.get("spatial_rationale") or "Aligned to poster keypoint reading order."
            aligned.append(section)

        self._ensure_key_visual_for_keypoint_sections(aligned, classified_visuals, visual_context)
        self._limit_keypoint_visuals(aligned, classified_visuals)
        scp["sections"] = aligned

    def _align_grouped_sections_to_keypoints(
        self,
        story_board: Dict[str, Any],
        keypoints: List[Dict[str, Any]],
        visual_context: Dict[str, Any],
        classified_visuals: Dict[str, Any],
    ) -> None:
        scp = story_board.setdefault("spatial_content_plan", {})
        original_sections = scp.get("sections") or []
        if not isinstance(original_sections, list):
            original_sections = []

        target_count = self._keypoint_section_target_count(visual_context)
        template_name = str((visual_context or {}).get("requested_layout_template") or "")
        groups = self._partition_keypoints(keypoints, target_count, template_name)
        aligned: List[Dict[str, Any]] = []
        remaining = [section for section in original_sections if isinstance(section, dict)]

        for index, group in enumerate(groups):
            source = deepcopy(remaining[index]) if index < len(remaining) else {}
            texts = [
                normalize_text_for_poster(str(keypoint.get("key_point") or "").strip())
                for keypoint in group
                if str(keypoint.get("key_point") or "").strip()
            ]
            source_sections = self._unique_preserve_order(
                normalize_text_for_poster(str(keypoint.get("section") or "Paper").strip()) or "Paper"
                for keypoint in group
            )
            source_keypoint_ids = [int(keypoint["id"]) for keypoint in group]
            title = normalize_text_for_poster(str(source.get("section_title") or "").strip())
            if not title or len(title.split()) > self.validation_config["max_title_words"]:
                title = self._short_title_for_keypoint_group(texts, source_sections, index, len(groups))
            role = self._group_content_type(source_sections, title, texts, index, len(groups))

            section = deepcopy(source)
            template_defaults = {}
            if self._template_uses_grouped_keypoints(template_name):
                template_defaults = self._standard_group_defaults(index, title, role, texts, visual_context)
            title = template_defaults.get("section_title", title)
            role = template_defaults.get("content_type", role)
            section["section_id"] = str(source.get("section_id") or f"keypoint_group_{index + 1:02d}_{self._slugify(title)}")
            section["keypoint_id"] = source_keypoint_ids[0]
            section["source_keypoint_ids"] = source_keypoint_ids
            section["source_keypoints"] = texts
            section["source_section"] = source_sections[0] if source_sections else "Paper"
            section["source_sections"] = source_sections
            section["section_title"] = self._clean_section_title(title)
            section["column_assignment"] = (
                template_defaults.get("column_assignment")
                or (
                    section.get("column_assignment")
                    if section.get("column_assignment") in {"left", "middle", "right"}
                    else self._group_column(role, index, len(groups))
                )
            )
            section["vertical_priority"] = (
                template_defaults.get("vertical_priority")
                or (
                    section.get("vertical_priority")
                    if section.get("vertical_priority") in {"top", "middle", "bottom"}
                    else self._keypoint_vertical_priority(index, len(groups))
                )
            )
            section["importance_level"] = int(section.get("importance_level") or (1 if role == "method" else 2 if role == "results" else 3))
            section["content_type"] = role
            section["expected_content_density"] = (
                section.get("expected_content_density")
                or template_defaults.get("expected_content_density")
                or ("medium" if role in {"method", "results"} else "high")
            )
            standard_template = template_name in self._standard_template_ids()
            existing_text = section.get("text_content")
            section["text_content"] = self._grouped_keypoint_text_content(
                existing_text,
                texts,
                role=role,
                source_sections=source_sections,
                title=str(title or ""),
                prefer_existing=standard_template,
            )
            section["visual_assets"] = self._valid_visual_assets(section.get("visual_assets"), visual_context)
            if template_defaults.get("preferred_slot_id"):
                section["preferred_slot_id"] = template_defaults["preferred_slot_id"]
            fast_budget = self._fast_budget_for_slot(visual_context, section.get("preferred_slot_id"))
            if fast_budget:
                section["capacity_budget"] = fast_budget
                section["target_chars"] = fast_budget.get("target_chars")
                section["min_chars"] = fast_budget.get("min_chars")
                section["max_chars"] = fast_budget.get("max_chars")
                section["target_bullets"] = fast_budget.get("target_bullets")
                target_chars = int(fast_budget.get("target_chars") or 0)
                section["expected_content_density"] = (
                    "high" if target_chars >= 430 else "low" if target_chars <= 140 else "medium"
                )
            section["spatial_rationale"] = (
                section.get("spatial_rationale")
                or template_defaults.get("spatial_rationale")
                or "Grouped keypoints to match the selected template's visual block geometry."
            )
            aligned.append(section)

        if template_name in self._standard_template_ids():
            self._normalize_standard_grouped_section_titles(aligned)
            self._ensure_standard_template_grouped_visuals(aligned, classified_visuals, visual_context)
        else:
            self._ensure_grouped_keypoint_visuals(aligned, classified_visuals, visual_context)
        scp["sections"] = aligned

    def _ordered_keypoints(self, state: PosterState) -> List[Dict[str, Any]]:
        keypoints = state.get("paper_poster_keypoints") or []
        if not keypoints:
            return []
        by_id = {}
        for item in keypoints:
            try:
                by_id[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue
        order = []
        for value in state.get("poster_reading_order") or []:
            try:
                keypoint_id = int(value)
            except (TypeError, ValueError):
                continue
            if keypoint_id in by_id and keypoint_id not in order:
                order.append(keypoint_id)
        for keypoint_id in sorted(by_id):
            if keypoint_id not in order:
                order.append(keypoint_id)
        return [by_id[keypoint_id] for keypoint_id in order[:10]]

    def _short_title_for_keypoint(self, keypoint_text: str, source_section: str) -> str:
        section_title = re.sub(r"^\d+\.?\s*", "", source_section or "").strip()
        generic_sections = {"introduction", "method", "methods", "experiments", "results", "conclusion", "discussion"}
        if section_title and section_title.lower() not in generic_sections and len(section_title.split()) <= 4:
            return section_title
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", keypoint_text or "")
        stop = {"the", "and", "for", "with", "from", "that", "this", "using", "into"}
        kept = [word for word in words if word.lower() not in stop][:4]
        return " ".join(kept) or "Key Point"

    def _short_title_for_keypoint_group(
        self,
        keypoint_texts: List[str],
        source_sections: List[str],
        index: int,
        total: int,
    ) -> str:
        joined = " ".join(keypoint_texts)
        section_joined = " ".join(source_sections)
        role = self._group_content_type(source_sections, "", keypoint_texts, index, total)
        if role == "method":
            if re.search(r"\bam[- ]?elo\b", joined, re.I):
                return "am-ELO Method"
            return "Core Method"
        if role == "results":
            if re.search(r"robust|perturb|consisten", joined, re.I):
                return "Robustness"
            return "Key Results"
        if index == 0:
            return "Motivation"
        if re.search(r"stable|stability|mle|likelihood", joined + " " + section_joined, re.I):
            return "Stable Arena"
        if index >= total - 1:
            return "Takeaway"
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", joined or section_joined)
        stop = {"the", "and", "for", "with", "from", "that", "this", "using", "into", "based"}
        kept = [word for word in words if word.lower() not in stop][:3]
        return " ".join(kept) or "Key Point"

    def _keypoint_column(self, source_section: str, title: str, index: int, total: int) -> str:
        text = f"{source_section} {title}".lower()
        if any(token in text for token in ["result", "experiment", "evaluation", "analysis", "benchmark"]):
            return "right"
        if any(token in text for token in ["method", "approach", "framework", "model", "algorithm", "system"]):
            return "middle"
        if index >= max(total - 2, 0):
            return "right"
        if index >= max(total // 3, 1):
            return "middle"
        return "left"

    def _keypoint_vertical_priority(self, index: int, total: int) -> str:
        if index < max(total / 3, 1):
            return "top"
        if index < max((2 * total) / 3, 2):
            return "middle"
        return "bottom"

    def _keypoint_content_type(self, source_section: str, title: str) -> str:
        text = f"{source_section} {title}".lower()
        if any(token in text for token in ["result", "experiment", "evaluation", "analysis", "benchmark"]):
            return "results"
        if any(token in text for token in ["method", "approach", "framework", "model", "algorithm", "system"]):
            return "method"
        return "foundation"

    def _group_content_type(
        self,
        source_sections: List[str],
        title: str,
        keypoint_texts: List[str],
        index: int,
        total: int,
    ) -> str:
        text = f"{' '.join(source_sections)} {title} {' '.join(keypoint_texts)}".lower()
        if any(token in text for token in ["result", "experiment", "evaluation", "benchmark", "performance", "robust", "accuracy", "loss"]):
            return "results"
        if any(token in text for token in ["method", "approach", "framework", "model", "algorithm", "mle", "likelihood", "elo", "am-elo", "m-elo"]):
            return "method"
        if index >= max(total - 1, 0):
            return "takeaway"
        return "foundation"

    def _group_column(self, role: str, index: int, total: int) -> str:
        if role == "method":
            return "middle"
        if role in {"results", "takeaway"}:
            return "right"
        if index >= max(total - 2, 0):
            return "right"
        if index >= max(total // 3, 1):
            return "middle"
        return "left"

    def _keypoint_text_content(self, existing: Any, keypoint_text: str) -> List[str]:
        bullets = []
        if isinstance(existing, list):
            bullets = self._clean_poster_text_items(existing, max_items=3)
        if not bullets:
            return [keypoint_text]
        key = self._dedupe_key(keypoint_text)
        if key and all(key not in self._dedupe_key(item) and self._dedupe_key(item) not in key for item in bullets):
            bullets.insert(0, keypoint_text)
        return self._clean_poster_text_items(bullets, max_items=3)

    def _grouped_keypoint_text_content(
        self,
        existing: Any,
        keypoint_texts: List[str],
        *,
        role: str = "",
        source_sections: Optional[List[str]] = None,
        title: str = "",
        prefer_existing: bool = False,
    ) -> List[str]:
        keypoint_bullets = []
        for text in keypoint_texts:
            clean = normalize_text_for_poster(str(text or "").strip())
            if clean:
                keypoint_bullets.append(clean)
        existing_bullets = []
        if isinstance(existing, list):
            for clean in self._clean_poster_text_items(existing, max_items=4):
                if not self._existing_group_text_relevant(
                    clean,
                    keypoint_texts,
                    role=role,
                    source_sections=source_sections or [],
                    title=title,
                ):
                    continue
                existing_bullets.append(clean)

        bullets = existing_bullets + keypoint_bullets if prefer_existing and existing_bullets else keypoint_bullets + existing_bullets
        deduped = []
        for clean in bullets:
            key = self._dedupe_key(clean)
            if key and all(key not in self._dedupe_key(item) and self._dedupe_key(item) not in key for item in deduped):
                deduped.append(clean)
        return self._clean_poster_text_items(deduped, max_items=4) or ["Key paper finding."]

    def _existing_group_text_relevant(
        self,
        text: str,
        keypoint_texts: List[str],
        *,
        role: str,
        source_sections: List[str],
        title: str,
    ) -> bool:
        clean = normalize_text_for_poster(str(text or "").strip())
        if not clean:
            return False
        role_key = str(role or "").lower()
        source_key = " ".join(str(section or "") for section in source_sections).lower()
        title_key = str(title or "").lower()
        result_like_section = (
            role_key in {"results", "takeaway"}
            or any(token in source_key for token in ("result", "experiment", "evaluation", "benchmark"))
            or any(token in title_key for token in ("result", "evaluation", "takeaway"))
        )
        if self._looks_like_result_summary_text(clean) and not result_like_section:
            return False

        candidate_terms = self._content_terms(clean)
        group_terms = self._content_terms(
            " ".join(
                [
                    *[str(value or "") for value in keypoint_texts],
                    *[str(value or "") for value in source_sections],
                    str(title or ""),
                    str(role or ""),
                ]
            )
        )
        if not candidate_terms or not group_terms:
            return False
        overlap = candidate_terms & group_terms
        required_overlap = 1 if len(group_terms) <= 4 else 2
        return len(overlap) >= required_overlap

    def _looks_like_result_summary_text(self, text: str) -> bool:
        lowered = str(text or "").lower()
        return bool(
            re.search(
                r"\b(overall empirical conclusion|strongest method|outperforms?|best performing|"
                r"performance|target rates?|cost models?|budgets?|main result|key result)\b",
                lowered,
            )
        )

    def _content_terms(self, text: str) -> set[str]:
        stop = {
            "section",
            "result",
            "results",
            "method",
            "methods",
            "using",
            "with",
            "from",
            "this",
            "that",
            "paper",
            "main",
        }
        return {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text or "").lower())
            if term not in stop
        }

    def _clean_poster_text_items(self, items: Any, *, max_items: int) -> List[str]:
        if not isinstance(items, list):
            items = [items]
        cleaned: List[str] = []
        seen = set()
        for item in items:
            for raw_line in str(item or "").splitlines():
                text = normalize_text_for_poster(raw_line.strip())
                text = self._strip_text_item_marker(text)
                if not text:
                    continue
                key = self._dedupe_key(text)
                if not key or key in seen:
                    continue
                seen.add(key)
                cleaned.append(text)
                if len(cleaned) >= max_items:
                    return cleaned
        return cleaned

    def _strip_text_item_marker(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"^\s*[•●◦▪▫*\-]\s*", "", text)
        text = re.sub(r"^\s*(?:\d+[\.)]|step\s+\d+[\.:]?)\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _clean_section_title(self, title: Any) -> str:
        text = str(title or "").strip()
        text = text.replace("\u00a0", " ").replace("–", "-").replace("—", "-")
        text = self._repair_possessive_title_apostrophe(text)
        text = re.sub(r"\bwith\s+(?:a\s+)?table\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\btable\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*[•●◦▪▫*\-]\s*", "", text)
        text = re.sub(r"^\s*(?:\d+[\.)]|step\s+\d+[\.:]?)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -:|")
        if not text:
            return "Key Point"
        small_words = {"and", "or", "with", "for", "to", "in", "of", "the", "a", "an", "by"}

        def clean_word(word: str, index: int) -> str:
            raw = word.strip()
            if re.fullmatch(r"[A-Z]{2,}[A-Za-z0-9-]*", raw):
                return raw
            lower = raw.lower()
            if index > 0 and lower in small_words:
                return lower
            return "-".join(part[:1].upper() + part[1:].lower() for part in raw.split("-") if part)

        words = [clean_word(word, index) for index, word in enumerate(text.split())]
        return " ".join(word for word in words if word)

    def _repair_possessive_title_apostrophe(self, title: str) -> str:
        return repair_possessive_title_apostrophe(title)

    def _valid_visual_assets(self, visuals: Any, visual_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        valid_ids = set((visual_context or {}).get("valid_visual_ids") or [])
        result = []
        for visual in visuals or []:
            visual = self._normalize_visual_reference(visual)
            if not visual:
                continue
            visual_id = visual.get("visual_id")
            if valid_ids and visual_id not in valid_ids:
                continue
            result.append(visual)
        return result

    def _normalize_visual_reference(self, visual: Any) -> Dict[str, Any] | None:
        if isinstance(visual, dict):
            visual_id = visual.get("visual_id") or visual.get("id")
            if not visual_id:
                return None
            item = dict(visual)
            item["visual_id"] = str(visual_id)
            return item
        if isinstance(visual, str) and visual.strip():
            return {
                "visual_id": visual.strip(),
                "visual_purpose": "Referenced by spatial planner",
                "placement_rationale": "Normalized from compact visual id",
            }
        return None

    def _ensure_key_visual_for_keypoint_sections(
        self,
        sections: List[Dict[str, Any]],
        classified_visuals: Dict[str, Any],
        visual_context: Dict[str, Any],
    ) -> None:
        key_visual = (classified_visuals or {}).get("key_visual")
        if not key_visual or key_visual not in set((visual_context or {}).get("valid_visual_ids") or []):
            return
        holder = None
        for section in sections:
            if any(
                normalized and normalized.get("visual_id") == key_visual
                for normalized in [self._normalize_visual_reference(visual) for visual in section.get("visual_assets", [])]
            ):
                holder = section
                break
        if holder is None:
            holder = next(
                (
                    section for section in sections
                    if self._keypoint_content_type(section.get("source_section", ""), section.get("section_title", "")) == "method"
                ),
                sections[min(1, len(sections) - 1)] if sections else None,
            )
            if holder is None:
                return
            holder.setdefault("visual_assets", []).insert(0, {
                "visual_id": key_visual,
                "visual_purpose": "Primary method or contribution visual",
                "placement_rationale": "Anchors the core contribution block",
            })
        holder["column_assignment"] = "middle"
        holder["vertical_priority"] = "top"
        holder["importance_level"] = 1

    def _ensure_grouped_keypoint_visuals(
        self,
        sections: List[Dict[str, Any]],
        classified_visuals: Dict[str, Any],
        visual_context: Dict[str, Any],
    ) -> None:
        valid_ids = set((visual_context or {}).get("valid_visual_ids") or [])
        if not valid_ids or not sections:
            return
        for section in sections:
            section["visual_assets"] = []

        def valid_visual_id(visual_id: Any) -> str | None:
            visual_id = str(visual_id or "")
            return visual_id if visual_id in valid_ids else None

        method_visual = valid_visual_id((classified_visuals or {}).get("key_visual"))
        if not method_visual:
            method_visual = self._first_valid_visual((classified_visuals or {}).get("method_workflow"), valid_ids)
        result_visual = self._first_valid_visual(
            [
                visual_id
                for visual_id in ((classified_visuals or {}).get("main_results") or [])
                if str(visual_id).startswith("figure_")
            ],
            valid_ids,
            exclude={method_visual} if method_visual else set(),
        )
        table_visual = self._preferred_table_visual(classified_visuals or {}, valid_ids, visual_context)

        if method_visual:
            holder = self._first_section_by_role(sections, "method") or sections[min(1, len(sections) - 1)]
            self._set_single_visual(holder, method_visual, "Primary method or contribution visual")
            holder["column_assignment"] = "middle"
            holder["vertical_priority"] = "top"
            holder["importance_level"] = 1

        if result_visual:
            holder = self._first_section_by_role(sections, "results", exclude_visual=True)
            if holder:
                self._set_single_visual(holder, result_visual, "Primary empirical result visual")
                holder["column_assignment"] = "right"
                holder["importance_level"] = min(int(holder.get("importance_level") or 2), 2)

        if table_visual:
            holder = self._last_section_by_role(sections, "results", exclude_visual=True)
            if holder:
                self._set_single_visual(holder, table_visual, "Compact result table for quantitative evidence")
                holder["column_assignment"] = "right"
                holder["vertical_priority"] = holder.get("vertical_priority") if holder.get("vertical_priority") in {"middle", "bottom"} else "bottom"
                holder["importance_level"] = min(int(holder.get("importance_level") or 2), 2)

    def _standard_group_defaults(
        self,
        index: int,
        current_title: str,
        current_role: str,
        keypoint_texts: List[str],
        visual_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        contract = (visual_context or {}).get("fast_block_contract") or {}
        blocks = contract.get("blocks") or []
        block = blocks[index] if index < len(blocks) else {}
        slot_id = block.get("slot_id")
        role = block.get("content_role") or current_role
        title = current_title
        slot_role = str(block.get("slot_role") or "").strip()
        if slot_role and (not title or title.lower() in {"paper", "method", "results", "experiments"}):
            title = slot_role
        title = self._standard_group_title(title, role, keypoint_texts, slot_role)

        bbox = block.get("slot_bbox") or {}
        layout = (visual_context or {}).get("template_layout") or {}
        regions = layout.get("regions") or []
        max_right = max((float(region.get("x", 0.0)) + float(region.get("w", 0.0)) for region in regions), default=1.0)
        min_top = min((float(region.get("y", 0.0)) for region in regions), default=0.0)
        max_bottom = max((float(region.get("y", 0.0)) + float(region.get("h", 0.0)) for region in regions), default=1.0)
        center_x = float(bbox.get("x", 0.0) or 0.0) + float(bbox.get("w", 0.0) or 0.0) / 2
        center_y = float(bbox.get("y", 0.0) or 0.0) + float(bbox.get("h", 0.0) or 0.0) / 2
        if center_x < max_right / 3:
            column = "left"
        elif center_x > max_right * 2 / 3:
            column = "right"
        else:
            column = "middle"
        body_span = max(max_bottom - min_top, 0.1)
        relative_y = center_y - min_top
        if relative_y < body_span / 3:
            vertical = "top"
        elif relative_y > body_span * 2 / 3:
            vertical = "bottom"
        else:
            vertical = "middle"

        target_chars = int(block.get("target_chars") or 0)
        return {
            "section_title": title,
            "content_type": role,
            "preferred_slot_id": slot_id,
            "column_assignment": column,
            "vertical_priority": vertical,
            "expected_content_density": "high" if target_chars >= 430 else "low" if target_chars <= 160 else "medium",
            "spatial_rationale": "Standard-template grouped keypoint mapping keeps ten paper points in curated visual blocks.",
        }

    def _standard_group_title(
        self,
        current_title: str,
        role: str,
        keypoint_texts: List[str],
        slot_role: str,
    ) -> str:
        title = repair_possessive_title_apostrophe(str(current_title or "").strip())
        title_key = self._dedupe_key(title)
        joined = " ".join(str(text or "") for text in keypoint_texts).lower()
        slot_key = str(slot_role or "").strip().lower()
        generic_titles = {
            "",
            "paper",
            "paper s main",
            "paper's main",
            "core method",
            "method details",
            "method visual",
            "system flow",
            "system details",
            "key results",
            "results",
            "evaluation setup",
            "main results with table",
        }

        if re.search(r"\bpaper'?s?\s+main\b", title, flags=re.IGNORECASE) or "main application" in joined:
            return "Main Application"
        if re.search(r"\b(prior active search|visual active search|related work|baseline methods?)\b", joined):
            return "Prior Methods"
        if "previous eviction prediction" in joined:
            return "Prior Prediction"
        if re.search(r"\b(policy|prediction module|search module|remaining budget|sequential geospatial search)\b", joined):
            return "Search Policy"
        if re.search(r"\b(hags)\b", joined) and re.search(r"\b(introduced|scalable|hierarchical)\b", joined):
            return "HAGS Overview" if title_key in generic_titles or title_key == "hags is introduced" else title
        if role == "results" and re.search(r"\b(empirical|outperform|strongest|target rates?|budgets?|positive rate)\b", joined):
            return "Main Results"
        if slot_key.startswith("main results") and title_key in generic_titles:
            return "Main Results"
        if slot_key.startswith("evaluation") and title_key in generic_titles:
            return "Evaluation Setup"
        if slot_key.startswith("method details") and title_key in generic_titles:
            return "Method Details"
        if slot_key and title_key in generic_titles:
            return self._clean_section_title(slot_role)
        return title

    def _dedupe_grouped_section_titles(self, sections: List[Dict[str, Any]]) -> None:
        used: set[str] = set()
        for section in sections:
            title = self._clean_section_title(section.get("section_title") or "")
            key = self._dedupe_key(title)
            if key and key not in used:
                section["section_title"] = title
                used.add(key)
                continue
            replacement = self._alternate_grouped_section_title(section, used)
            section["section_title"] = replacement
            used.add(self._dedupe_key(replacement))

    def _alternate_grouped_section_title(self, section: Dict[str, Any], used: set[str]) -> str:
        text = " ".join(
            [
                *[str(value or "") for value in section.get("source_keypoints") or []],
                *[str(value or "") for value in section.get("text_content") or []],
                " ".join(str(value or "") for value in section.get("source_sections") or []),
            ]
        ).lower()
        role = str(section.get("content_type") or section.get("content_role") or "").lower()
        candidates: List[str] = []
        if re.search(r"\b(prior active search|visual active search|related work|baseline methods?)\b", text):
            candidates.append("Prior Methods")
        if "previous eviction prediction" in text:
            candidates.append("Prior Prediction")
        if re.search(r"\b(policy|prediction module|search module|remaining budget|sequential geospatial search)\b", text):
            candidates.append("Search Policy")
        if role == "results":
            candidates.extend(["Main Results", "Result Summary", "Takeaway"])
        elif role == "method":
            candidates.extend(["Method Details", "Method Workflow", "Core Mechanism"])
        else:
            candidates.extend(["Motivation", "Problem Setting", "Context"])

        for candidate in candidates:
            key = self._dedupe_key(candidate)
            if key and key not in used:
                return candidate
        slot_id = str(section.get("preferred_slot_id") or section.get("slot_id") or "")
        fallback = f"{self._clean_section_title(role or 'Section')} {slot_id.replace('_', ' ').title()}".strip()
        return fallback or "Section"

    def _normalize_standard_grouped_section_titles(self, sections: List[Dict[str, Any]]) -> None:
        for section in sections:
            budget = section.get("capacity_budget") or {}
            slot_role = str(budget.get("slot_role") or "")
            role = str(section.get("content_type") or section.get("content_role") or budget.get("content_role") or "")
            keypoint_texts = [
                str(value or "")
                for value in (section.get("source_keypoints") or section.get("text_content") or [])
            ]
            section["section_title"] = self._clean_section_title(
                self._standard_group_title(
                    str(section.get("section_title") or ""),
                    role,
                    keypoint_texts,
                    slot_role,
                )
            )
        self._dedupe_grouped_section_titles(sections)

    def _ensure_standard_template_grouped_visuals(
        self,
        sections: List[Dict[str, Any]],
        classified_visuals: Dict[str, Any],
        visual_context: Dict[str, Any],
    ) -> None:
        valid_ids = set((visual_context or {}).get("valid_visual_ids") or [])
        if not valid_ids or not sections:
            return
        protected_visuals_by_slot: Dict[str, List[Dict[str, Any]]] = {}
        for section in sections:
            slot_id = str(section.get("preferred_slot_id") or "")
            protected = []
            for visual in section.get("visual_assets") or []:
                normalized = self._normalize_visual_reference(visual)
                if not normalized:
                    continue
                visual_id = str(normalized.get("visual_id") or "")
                if visual_id.startswith("generated_teaser") and visual_id in valid_ids:
                    protected.append(normalized)
            if protected and slot_id:
                protected_visuals_by_slot[slot_id] = protected
            section["visual_assets"] = []

        slot_map = {str(section.get("preferred_slot_id") or ""): section for section in sections}
        fast_visual_policy = (visual_context or {}).get("fast_visual_policy") or {}
        figure_slots = [str(slot_id) for slot_id in fast_visual_policy.get("figure_slots") or []]
        table_slots = [str(slot_id) for slot_id in fast_visual_policy.get("table_slots") or []]
        figure_slot_set = set(figure_slots)
        table_slot_set = set(table_slots)
        figure_candidates = self._grouped_figure_candidates(classified_visuals or {}, valid_ids, visual_context)
        table_candidates = self._grouped_table_candidates(classified_visuals or {}, valid_ids, visual_context)

        is_portrait = self._is_portrait_or_vertical_template(visual_context)
        figure_count = int(fast_visual_policy.get("figure_count") or 2)
        table_count = int(fast_visual_policy.get("table_count") or 1)
        if not is_portrait:
            figure_count = min(max(figure_count, 0), len(figure_candidates))
            table_count = min(max(table_count, 0), len(table_candidates))
        max_visuals_total = int(fast_visual_policy.get("max_visuals_total") or (figure_count + table_count))
        if not is_portrait:
            max_visuals_total = min(max(max_visuals_total, 0), figure_count + table_count)
        used_visuals: set[str] = set()
        used_slots: set[str] = set()
        for slot_id, protected_visuals in protected_visuals_by_slot.items():
            holder = slot_map.get(slot_id)
            if not holder:
                continue
            holder["visual_assets"] = list(protected_visuals)
            holder["importance_level"] = min(int(holder.get("importance_level") or 1), 1)
            used_slots.add(slot_id)
            used_visuals.update(str(visual.get("visual_id") or "") for visual in protected_visuals)

        def place_visual(slot_id: str, visual_id: str, purpose: str, *, importance: int, append: bool = False) -> bool:
            paper_visual_count = sum(
                1 for used_visual_id in used_visuals
                if not str(used_visual_id).startswith("generated_teaser")
            )
            if paper_visual_count >= max_visuals_total:
                return False
            if not visual_id or visual_id in used_visuals:
                return False
            if slot_id in used_slots and not append:
                return False
            holder = slot_map.get(slot_id)
            if not holder:
                return False
            if visual_id.startswith("figure_") and figure_slot_set and slot_id not in figure_slot_set:
                return False
            if visual_id.startswith("table_") and table_slot_set and slot_id not in table_slot_set:
                return False
            if not self._slot_can_hold_visual(slot_id, visual_id, visual_context):
                log_agent_info(
                    self.name,
                    f"skip {visual_id} in {slot_id}: below visual footprint feasibility",
                )
                return False
            if append:
                self._append_visual(holder, visual_id, purpose)
            else:
                self._set_single_visual(holder, visual_id, purpose)
            holder["importance_level"] = min(int(holder.get("importance_level") or importance), importance)
            used_visuals.add(visual_id)
            used_slots.add(slot_id)
            return True

        if is_portrait:
            if figure_slots and figure_candidates:
                place_visual(
                    figure_slots[0],
                    figure_candidates[0],
                    "Primary method or contribution visual for the portrait template",
                    importance=1,
                )
            selected_tables = [visual_id for visual_id in table_candidates if visual_id not in used_visuals][:table_count]
            if table_slots and selected_tables:
                place_visual(table_slots[0], selected_tables[0], "Readable quantitative result table", importance=2)
            remaining_figures = [visual_id for visual_id in figure_candidates[1:figure_count] if visual_id not in used_visuals]
            for slot_id, visual_id in zip(figure_slots[1:figure_count], remaining_figures):
                place_visual(slot_id, visual_id, "Supporting method or result figure", importance=2)
            remaining_tables = [visual_id for visual_id in table_candidates if visual_id not in used_visuals][:table_count]
            for slot_id, selected_table in zip(table_slots[1:table_count], remaining_tables):
                place_visual(slot_id, selected_table, "Supporting quantitative result table", importance=2)
            return

        figure_slot_order = self._standard_visual_slot_order(figure_slots, slot_map, visual_context)
        table_slot_order = self._standard_visual_slot_order(table_slots, slot_map, visual_context)
        placed_figures = 0
        for visual_id in figure_candidates:
            if placed_figures >= figure_count:
                break
            for slot_id in figure_slot_order:
                if place_visual(slot_id, visual_id, "Primary method or system figure for the standard template", importance=1):
                    placed_figures += 1
                    break

        placed_tables = 0
        for selected_table in table_candidates:
            if placed_tables >= table_count:
                break
            if selected_table in used_visuals:
                continue
            for slot_id in table_slot_order:
                if place_visual(slot_id, selected_table, "Primary quantitative result table", importance=2):
                    placed_tables += 1
                    break

        if placed_figures < min(figure_count, len(figure_candidates)):
            multi_slot_order = figure_slot_order or self._standard_multi_visual_slot_order(slot_map, visual_context)
            for visual_id in figure_candidates:
                if placed_figures >= min(figure_count, len(figure_candidates)):
                    break
                if visual_id in used_visuals:
                    continue
                for slot_id in multi_slot_order:
                    if place_visual(slot_id, visual_id, "Supporting method or system figure for the standard template", importance=2, append=True):
                        placed_figures += 1
                        break

        if placed_tables < min(table_count, len(table_candidates)):
            multi_slot_order = table_slot_order or self._standard_multi_visual_slot_order(slot_map, visual_context)
            for selected_table in table_candidates:
                if placed_tables >= min(table_count, len(table_candidates)):
                    break
                if selected_table in used_visuals:
                    continue
                for slot_id in multi_slot_order:
                    if place_visual(slot_id, selected_table, "Supporting quantitative result table for a large visual block", importance=2, append=True):
                        placed_tables += 1
                        break

    def _standard_visual_slot_order(
        self,
        preferred_slot_ids: List[str],
        slot_map: Dict[str, Dict[str, Any]],
        visual_context: Dict[str, Any],
    ) -> List[str]:
        template_layout = (visual_context or {}).get("template_layout") or {}
        lane_by_id = {str(lane.get("id") or ""): lane for lane in template_layout.get("lanes") or []}
        region_by_id = {
            str(region.get("region_id") or region.get("slot_id") or region.get("id") or ""): region
            for region in template_layout.get("regions") or []
        }
        candidate_slot_ids = preferred_slot_ids if preferred_slot_ids else list(slot_map.keys())
        ordered = [
            slot_id
            for slot_id in self._unique_preserve_order(candidate_slot_ids)
            if slot_id in slot_map and self._slot_declares_visual_host(slot_id, visual_context)
        ]

        def area(slot_id: str) -> float:
            geometry = lane_by_id.get(slot_id) or region_by_id.get(slot_id) or {}
            return float(geometry.get("w", 0.0) or 0.0) * float(geometry.get("h", 0.0) or 0.0)

        original_index = {slot_id: index for index, slot_id in enumerate(ordered)}
        return sorted(ordered, key=lambda slot_id: (-area(slot_id), original_index.get(slot_id, 999)))

    def _standard_multi_visual_slot_order(
        self,
        slot_map: Dict[str, Dict[str, Any]],
        visual_context: Dict[str, Any],
    ) -> List[str]:
        slots_with_visuals = [
            slot_id
            for slot_id, section in slot_map.items()
            if section.get("visual_assets") and self._slot_declares_visual_host(slot_id, visual_context)
        ]
        return self._standard_visual_slot_order(slots_with_visuals, slot_map, visual_context)

    def _slot_declares_visual_host(self, slot_id: str, visual_context: Dict[str, Any]) -> bool:
        template_layout = (visual_context or {}).get("template_layout") or {}
        regions = template_layout.get("regions") or []
        for region in regions:
            region_id = str(region.get("region_id") or region.get("slot_id") or region.get("id") or "")
            if region_id == str(slot_id):
                return bool(region.get("can_host_visual", True))
        return True

    def _slot_can_hold_visual(self, slot_id: str, visual_id: str, visual_context: Dict[str, Any]) -> bool:
        if not self._slot_declares_visual_host(slot_id, visual_context):
            return False
        template_layout = (visual_context or {}).get("template_layout") or {}
        lanes = template_layout.get("lanes") or []
        lane = next((lane for lane in lanes if str(lane.get("id") or "") == str(slot_id)), None)
        if not lane:
            return True
        lane = dict(lane)
        lane.setdefault("poster_orientation", template_layout.get("orientation"))
        text_padding = 2 * float(self.config["layout"]["text_padding"]["left_right"])
        max_width = max(float(lane.get("w", 0.0) or 0.0) - text_padding, 0.1)
        visual_width_cap = template_layout.get("visual_width_cap")
        if visual_width_cap:
            max_width = min(max_width, float(visual_width_cap))
        visual_assets = dict((visual_context or {}).get("visual_assets") or {})
        if visual_id not in visual_assets:
            estimate = ((visual_context or {}).get("visual_assets_heights") or {}).get(visual_id) or {}
            if estimate:
                visual_assets[visual_id] = {
                    "asset_type": "table" if str(visual_id).startswith("table_") else "figure",
                    "aspect": estimate.get("aspect_ratio") or estimate.get("aspect"),
                }
        return visual_slot_is_feasible(
            visual_id,
            lane,
            visual_assets,
            self.config,
            max_width=max_width,
        )

    def _limit_keypoint_visuals(self, sections: List[Dict[str, Any]], classified_visuals: Dict[str, Any]) -> None:
        key_visual = (classified_visuals or {}).get("key_visual")
        kept_total = 0
        for section in sections:
            role = self._keypoint_content_type(section.get("source_section", ""), section.get("section_title", ""))
            visuals = []
            for visual in section.get("visual_assets") or []:
                visual = self._normalize_visual_reference(visual)
                if not visual:
                    continue
                visual_id = visual.get("visual_id")
                if visual_id == key_visual:
                    visuals.append(visual)
                    kept_total += 1
                    continue
                if role in {"method", "results"} and kept_total < 2:
                    visuals.append(visual)
                    kept_total += 1
            section["visual_assets"] = visuals[:1]

    def _first_valid_visual(self, visual_ids: Any, valid_ids: set[str], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        for visual_id in visual_ids or []:
            visual_id = str(visual_id or "")
            if visual_id in valid_ids and visual_id not in exclude:
                return visual_id
        return None

    def _grouped_figure_candidates(
        self,
        classified_visuals: Dict[str, Any],
        valid_ids: set[str],
        visual_context: Dict[str, Any],
    ) -> List[str]:
        ordered: List[str] = []
        key_visual = str((classified_visuals or {}).get("key_visual") or "")
        if key_visual.startswith("figure_"):
            ordered.append(key_visual)
        for bucket in ("method_workflow", "main_results", "comparative_results", "problem_illustration", "supporting"):
            ordered.extend(str(visual_id or "") for visual_id in (classified_visuals or {}).get(bucket) or [])
        ordered.extend(sorted(valid_ids, key=self._visual_sort_key))
        candidates = [
            visual_id
            for visual_id in self._unique_preserve_order(ordered)
            if visual_id in valid_ids and visual_id.startswith("figure_")
        ]
        return self._rank_readable_visuals(candidates, visual_context, prefer_existing_order=True)

    def _grouped_table_candidates(
        self,
        classified_visuals: Dict[str, Any],
        valid_ids: set[str],
        visual_context: Dict[str, Any],
    ) -> List[str]:
        ordered: List[str] = []
        for bucket in ("main_results", "comparative_results", "supporting", "problem_illustration", "method_workflow"):
            ordered.extend(str(visual_id or "") for visual_id in (classified_visuals or {}).get(bucket) or [])
        ordered.extend(sorted(valid_ids, key=self._visual_sort_key))
        candidates = [
            visual_id
            for visual_id in self._unique_preserve_order(ordered)
            if visual_id in valid_ids and visual_id.startswith("table_")
        ]
        return self._rank_readable_visuals(candidates, visual_context, prefer_existing_order=False)

    def _rank_readable_visuals(
        self,
        visual_ids: List[str],
        visual_context: Dict[str, Any],
        *,
        prefer_existing_order: bool,
    ) -> List[str]:
        visual_heights = (visual_context or {}).get("visual_assets_heights") or {}
        max_height = self._max_visual_height_percentage(visual_context)
        original_index = {visual_id: index for index, visual_id in enumerate(visual_ids)}

        def score(visual_id: str) -> tuple:
            aspect = float((visual_heights.get(visual_id) or {}).get("aspect_ratio") or 1.0)
            height_percentage = self._visual_height_percentage(visual_id, visual_heights)
            too_tall = height_percentage is not None and height_percentage > max_height
            too_wide = aspect > 3.2
            readability = int(too_tall) + int(too_wide)
            if prefer_existing_order:
                return (original_index.get(visual_id, 999), readability, abs(aspect - 1.8), aspect)
            return (readability, abs(aspect - 1.8), aspect, original_index.get(visual_id, 999))

        return sorted(visual_ids, key=score)

    def _visual_sort_key(self, visual_id: str) -> tuple:
        prefix = re.sub(r"\d+$", "", str(visual_id or ""))
        match = re.search(r"(\d+)$", str(visual_id or ""))
        number = int(match.group(1)) if match else 9999
        return (prefix, number, str(visual_id or ""))

    def _preferred_table_visual(
        self,
        classified_visuals: Dict[str, Any],
        valid_ids: set[str],
        visual_context: Dict[str, Any],
    ) -> str | None:
        candidates = []
        for bucket in ("main_results", "comparative_results", "supporting"):
            for visual_id in classified_visuals.get(bucket) or []:
                visual_id = str(visual_id or "")
                if visual_id.startswith("table_") and visual_id in valid_ids:
                    candidates.append(visual_id)
        if "table_3" in candidates:
            visual_heights = (visual_context or {}).get("visual_assets_heights") or {}
            table_3_height = self._visual_height_percentage("table_3", visual_heights)
            table_3_aspect = float((visual_heights.get("table_3") or {}).get("aspect_ratio") or 1.0)
            if table_3_height is None or (table_3_height <= self._max_visual_height_percentage(visual_context) and table_3_aspect <= 4.0):
                return "table_3"
        visual_heights = (visual_context or {}).get("visual_assets_heights") or {}
        max_height = self._max_visual_height_percentage(visual_context)
        readable = [
            visual_id
            for visual_id in candidates
            if float((visual_heights.get(visual_id) or {}).get("aspect_ratio") or 1.0) <= 4.0
            and (
                self._visual_height_percentage(visual_id, visual_heights) is None
                or self._visual_height_percentage(visual_id, visual_heights) <= max_height
            )
        ]
        return readable[0] if readable else None

    def _visual_height_percentage(self, visual_id: str, visual_heights: Dict[str, Any]) -> float | None:
        value = (visual_heights.get(visual_id) or {}).get("height_percentage")
        if value is None:
            return None
        try:
            return float(str(value).rstrip("%"))
        except (TypeError, ValueError):
            return None

    def _first_section_by_role(
        self,
        sections: List[Dict[str, Any]],
        role: str,
        *,
        exclude_visual: bool = False,
    ) -> Dict[str, Any] | None:
        for section in sections:
            if exclude_visual and section.get("visual_assets"):
                continue
            if section.get("content_type") == role:
                return section
        return None

    def _last_section_by_role(
        self,
        sections: List[Dict[str, Any]],
        role: str,
        *,
        exclude_visual: bool = False,
    ) -> Dict[str, Any] | None:
        for section in reversed(sections):
            if exclude_visual and section.get("visual_assets"):
                continue
            if section.get("content_type") == role:
                return section
        return None

    def _set_single_visual(self, section: Dict[str, Any], visual_id: str, purpose: str) -> None:
        section["visual_assets"] = [{
            "visual_id": visual_id,
            "visual_purpose": purpose,
            "placement_rationale": "Selected for grouped keypoint poster layout.",
        }]

    def _append_visual(self, section: Dict[str, Any], visual_id: str, purpose: str) -> None:
        visuals = list(section.get("visual_assets") or [])
        if any(str(visual.get("visual_id") or "") == visual_id for visual in visuals):
            return
        visuals.append({
            "visual_id": visual_id,
            "visual_purpose": purpose,
            "placement_rationale": "Backfilled into a large visual block to preserve the figure/table budget.",
        })
        section["visual_assets"] = visuals

    def _partition_keypoints(
        self,
        keypoints: List[Dict[str, Any]],
        target_count: int,
        template_name: str = "",
    ) -> List[List[Dict[str, Any]]]:
        if target_count <= 0 or len(keypoints) <= target_count:
            return [[keypoint] for keypoint in keypoints]
        if str(template_name or "") in self._standard_template_ids():
            if target_count == 4:
                slices = [(0, 2), (2, 5), (5, 7), (7, 10)]
            elif target_count == 5:
                slices = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
            elif target_count == 6:
                slices = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 10)]
            elif target_count == 7:
                slices = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 9), (9, 10)]
            else:
                slices = []
            if slices:
                groups = [
                    keypoints[start:min(end, len(keypoints))]
                    for start, end in slices
                    if start < len(keypoints)
                ]
                assigned = sum(len(group) for group in groups)
                if assigned < len(keypoints) and groups:
                    groups[-1].extend(keypoints[assigned:])
                return [group for group in groups if group]
        groups: List[List[Dict[str, Any]]] = []
        for index in range(target_count):
            start = math.floor(index * len(keypoints) / target_count)
            end = math.floor((index + 1) * len(keypoints) / target_count)
            if end <= start:
                end = start + 1
            groups.append(keypoints[start:end])
        return [group for group in groups if group]

    def _unique_preserve_order(self, values: Any) -> List[str]:
        result = []
        seen = set()
        for value in values:
            value = str(value or "").strip()
            key = value.lower()
            if value and key not in seen:
                result.append(value)
                seen.add(key)
        return result

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
        return slug[:32] or "section"

    def _template_uses_grouped_keypoints(self, template_name: str) -> bool:
        return str(template_name or "") in self._standard_template_ids()

    def _standard_template_ids(self) -> set[str]:
        return set((self.config.get("standard_template_policy") or {}).get("auto_templates") or [])

    def _group_keypoints_for_template(self, visual_context: Dict[str, Any] | None) -> bool:
        return bool((visual_context or {}).get("keypoint_grouping_mode"))

    def _fast_budget_for_slot(self, visual_context: Dict[str, Any] | None, slot_id: Any) -> Dict[str, Any]:
        if not (visual_context or {}).get("template_fast_mode"):
            return {}
        by_slot = ((visual_context or {}).get("fast_block_contract") or {}).get("by_slot") or {}
        budget = by_slot.get(str(slot_id or "")) or {}
        return dict(budget) if isinstance(budget, dict) else {}

    def _default_keypoint_section_target(
        self,
        template_name: str,
        layout: Dict[str, Any],
        state: PosterState,
    ) -> int:
        keypoint_count = min(len(state.get("paper_poster_keypoints") or []), 10)
        if not keypoint_count:
            return 0
        if not self._template_uses_grouped_keypoints(template_name):
            return keypoint_count
        content_slots = int(layout.get("slot_count") or len(layout.get("regions") or []) or 0)
        if content_slots <= 0:
            content_slots = 7
        if keypoint_count <= content_slots:
            return keypoint_count
        return min(content_slots, max(6, min(7, keypoint_count)))

    def _keypoint_section_target_count(self, visual_context: Dict[str, Any] | None) -> int:
        value = int((visual_context or {}).get("keypoint_section_target_count") or 0)
        keypoint_target = int((visual_context or {}).get("keypoint_target_count") or 0)
        if value:
            return value
        return keypoint_target

    def _dedupe_key(self, text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()[:140]

    def _validate_height_distribution(self, story_board: Dict, visual_context: Dict) -> Dict[str, Any]:
        """validate spatial plan for height constraints and generate warnings"""
        config = load_config()
        available_height = visual_context["available_height_per_column"]
        visual_heights = visual_context["visual_assets_heights"]
        
        # extract sections from story board
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        if not sections:
            return {"warnings": ["No sections found in story board"], "column_utilizations": {}}
        
        # organize sections by column
        columns = {"left": [], "middle": [], "right": []}
        for section in sections:
            column = section.get("column_assignment", "left")
            if column in columns:
                columns[column].append(section)
        
        # calculate estimated height for each section and column
        column_utilizations = {}
        warnings = []
        
        for column_name, column_sections in columns.items():
            total_height = 0
            total_visual_height = 0
            total_visuals = 0
            section_details = []
            
            for section in column_sections:
                section_height = self._estimate_section_height(section, visual_heights, config)
                total_height += section_height
                
                # calculate visual contribution for this section
                section_visual_height = 0
                visual_assets = section.get("visual_assets", [])
                for visual_asset in visual_assets:
                    visual_asset = self._normalize_visual_reference(visual_asset)
                    if not visual_asset:
                        continue
                    visual_id = visual_asset.get("visual_id", "")
                    if visual_id in visual_heights:
                        section_visual_height += visual_heights[visual_id]["height_inches"]
                        total_visuals += 1
                
                total_visual_height += section_visual_height
                section_details.append({
                    "section_id": section.get("section_id", "unknown"),
                    "estimated_height": section_height,
                    "visual_count": len(visual_assets),
                    "visual_height": round(section_visual_height, 1)
                })
            
            utilization = total_height / available_height if available_height > 0 else 0
            visual_density = total_visual_height / available_height if available_height > 0 else 0
            
            column_utilizations[column_name] = {
                "total_height": round(total_height, 1),
                "utilization_percent": f"{utilization*100:.0f}%",
                "visual_density_percent": f"{visual_density*100:.0f}%",
                "section_count": len(column_sections),
                "total_visuals": total_visuals,
                "sections": section_details,
                "status": "OK" if utilization <= self.utilization_config["overflow_critical"] else "OVERFLOW"
            }
            
            if utilization > self.utilization_config["overflow_critical"]:
                warnings.append(f"{column_name} column serious overflow: {utilization*100:.0f}% (visual density: {visual_density*100:.0f}%)")
            elif utilization > self.utilization_config["overflow_warning"]:
                warnings.append(f"{column_name} column minor overflow: {utilization*100:.0f}% (visual density: {visual_density*100:.0f}%)")
            elif utilization < self.utilization_config["underutilized"]:
                warnings.append(f"{column_name} column underutilized: {utilization*100:.0f}% (visual density: {visual_density*100:.0f}%)")
            
            if total_visuals == 0:
                warnings.append(f"{column_name} column has no visuals - add visual assets")
        
        return {
            "column_utilizations": column_utilizations,
            "warnings": warnings,
            "overall_status": "PASS" if not warnings else "NEEDS_OPTIMIZATION"
        }

    def _estimate_section_height(self, section: Dict, visual_heights: Dict, config: Dict) -> float:
        """estimate total height for a section including visuals and text"""
        total_height = 0
        
        # section title height (from config)
        section_title_height = config["section_estimation"]["base_title_height"]
        total_height += section_title_height
        
        # visual assets height
        visual_assets = section.get("visual_assets", [])
        for visual_asset in visual_assets:
            visual_asset = self._normalize_visual_reference(visual_asset)
            if not visual_asset:
                continue
            visual_id = visual_asset.get("visual_id", "")
            if visual_id in visual_heights:
                visual_height = visual_heights[visual_id]["height_inches"]
                visual_spacing = config["layout"]["visual_spacing"]["below_visual"]
                total_height += visual_height + visual_spacing
        
        # text content height (rough estimation)
        text_content = section.get("text_content", [])
        text_lines = len(text_content)
        bullet_height = config["section_estimation"]["bullet_point_height"]
        text_height = text_lines * bullet_height
        total_height += text_height
        
        # spacing between title and content
        title_spacing = config["layout"]["title_to_content_spacing"]
        total_height += title_spacing
        
        # section bottom spacing
        section_spacing = config["layout"]["section_spacing"]
        total_height += section_spacing
        
        return total_height

    def _load_cached_story_board(self, state: PosterState) -> Optional[Dict[str, Any]]:
        """load a previously saved story board (used as a fallback when generation fails)"""
        try:
            path = Path(state["output_dir"]) / "content" / "story_board.json"
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data or None
        except Exception:
            return None

    def _record_degraded_state(self, state: PosterState, *, component: str, category: str, reason: str, fallback: str):
        """record a non-blocking degraded quality state (ADR 0003/0006)"""
        state.setdefault("degraded_quality_states", []).append({
            "component": component,
            "category": category,
            "reason": reason,
            "fallback": fallback,
        })

    def _save_story_board(self, state: PosterState):
        """save story board to json file"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "story_board.json", "w", encoding='utf-8') as f:
            json.dump(state.get("story_board", {}), f, indent=2)
        if state.get("template_selection_report") is not None:
            with open(output_dir / "template_selection_report.json", "w", encoding="utf-8") as f:
                json.dump(state.get("template_selection_report", {}), f, indent=2)


def curator_node(state) -> Dict[str, Any]:
    result = StoryBoardCurator()(state)
    return {
        **state,
        "story_board": result["story_board"],
        "resolved_layout_template": result.get("resolved_layout_template"),
        "layout_template_metadata": result.get("layout_template_metadata"),
        "template_selection_report": result.get("template_selection_report"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }
