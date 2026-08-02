"""poster state management"""

from typing import Dict, Any, Optional, List, TypedDict
from dataclasses import dataclass, field
import os
import time


@dataclass
class ModelConfig:
    model_name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class TokenUsage:
    input_text: int = 0
    output_text: int = 0
    input_vision: int = 0
    output_vision: int = 0

    def add_text(self, inp: int, out: int):
        self.input_text += inp
        self.output_text += out

    def add_vision(self, inp: int, out: int):
        self.input_vision += inp
        self.output_vision += out


@dataclass
class APICall:
    agent: str
    call_type: str
    input_tokens: int
    output_tokens: int
    timestamp: float


@dataclass
class TimingMetrics:
    pipeline_start: float = 0.0
    pipeline_end: float = 0.0
    parser_time: float = 0.0
    standard_template_preselector_time: float = 0.0
    template_capacity_planner_time: float = 0.0
    poster_keypoint_selector_time: float = 0.0
    curator_time: float = 0.0
    template_block_planner_time: float = 0.0
    layout_optimizer_time: float = 0.0
    color_agent_time: float = 0.0
    header_planner_time: float = 0.0
    header_block_reviewer_time: float = 0.0
    font_agent_time: float = 0.0
    micro_layout_refiner_time: float = 0.0
    title_designer_time: float = 0.0
    visual_asset_agent_time: float = 0.0
    affiliation_logo_agent_time: float = 0.0
    renderer_time: float = 0.0
    vlm_layout_reviewer_time: float = 0.0
    visual_legibility_reviewer_time: float = 0.0
    generated_teaser_agent_time: float = 0.0
    background_image_agent_time: float = 0.0
    adaptive_column_relayout_time: float = 0.0
    template_region_relayout_time: float = 0.0
    block_occupancy_analyzer_time: float = 0.0
    block_vlm_reviewer_time: float = 0.0
    block_content_refiner_time: float = 0.0
    api_calls: List[APICall] = field(default_factory=list)

    def add_api_call(self, agent: str, call_type: str, input_tokens: int, output_tokens: int):
        self.api_calls.append(APICall(
            agent=agent,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=time.time()
        ))

    def get_total_time(self) -> float:
        if self.pipeline_start == 0.0 or self.pipeline_end == 0.0:
            return 0.0
        return round(self.pipeline_end - self.pipeline_start, 2)

    def get_api_call_count(self) -> int:
        return len(self.api_calls)

    def get_component_percentage(self, component_time: float) -> float:
        total = self.get_total_time()
        if total == 0:
            return 0.0
        return round((component_time / total) * 100, 2)


class PosterState(TypedDict):
    # core paths
    pdf_path: str
    output_dir: str
    poster_name: str

    # model configs
    text_model: ModelConfig
    vision_model: ModelConfig

    # processing results
    raw_text: Optional[str]
    images: Optional[Dict[str, Any]]
    tables: Optional[Dict[str, Any]]
    visual_assets: Optional[Dict[str, Any]]
    resolved_visual_assets: Optional[Dict[str, Any]]
    visual_plan: Optional[List[Dict[str, Any]]]
    affiliations: Optional[List[str]]
    affiliation_logos: Optional[List[Dict[str, Any]]]
    vlm_layout_review: Optional[Dict[str, Any]]
    vlm_layout_patch: Optional[List[Dict[str, Any]]]
    visual_legibility_review: Optional[Dict[str, Any]]
    block_occupancy_report: Optional[Dict[str, Any]]
    block_vlm_review: Optional[Dict[str, Any]]
    block_content_patch: Optional[Dict[str, Any]]
    block_refinement_history: Optional[Dict[str, Any]]
    adaptive_layout_decision: Optional[Dict[str, Any]]
    narrative: Optional[Dict[str, str]]
    poster_plan: Optional[List[Dict[str, Any]]]
    paper_poster_keypoints: Optional[List[Dict[str, Any]]]
    poster_reading_order: Optional[List[int]]
    poster_keypoint_selection_report: Optional[Dict[str, Any]]
    poster_width: int
    poster_height: int
    layout_template: str
    poster_variant: Optional[Dict[str, Any]]
    resolved_layout_template: Optional[str]
    layout_template_metadata: Optional[Dict[str, Any]]
    template_selection_report: Optional[Dict[str, Any]]
    standard_template_selection_report: Optional[Dict[str, Any]]
    adaptive_lane_widths: Optional[Dict[str, float]]
    template_layout_mode: Optional[str]
    template_block_plan: Optional[Dict[str, Any]]
    layout_intent: Optional[Dict[str, Any]]
    header_plan: Optional[Dict[str, Any]]
    header_block_review: Optional[Dict[str, Any]]
    header_block_patch_applied: bool
    template_prior_source_story_board: Optional[Dict[str, Any]]
    template_fast_mode: bool
    fast_block_contract: Optional[Dict[str, Any]]
    fast_visual_policy: Optional[Dict[str, Any]]
    fast_pipeline_report: Optional[Dict[str, Any]]
    block_capacity_contract: Optional[Dict[str, Any]]
    capacity_aware_story_board: Optional[Dict[str, Any]]
    capacity_planning_report: Optional[Dict[str, Any]]
    slot_pressure_report: Optional[Dict[str, Any]]
    wireframe_layout: Optional[List[Dict[str, Any]]]
    content_filled_layout: Optional[List[Dict[str, Any]]]
    final_layout: Optional[List[Dict[str, Any]]]

    narrative_content: Optional[Dict[str, Any]]
    classified_visuals: Optional[Dict[str, Any]]
    structured_sections: Optional[Dict[str, Any]]
    story_board: Optional[Dict[str, Any]]
    curated_content: Optional[Dict[str, Any]]
    design_layout: Optional[List[Dict[str, Any]]]
    section_title_design: Optional[Dict[str, Any]]
    color_scheme: Optional[Dict[str, str]]
    keywords: Optional[Dict[str, Any]]
    styled_layout: Optional[List[Dict[str, Any]]]
    styling_interfaces: Optional[Dict[str, Any]]
    initial_layout_data: Optional[List[Dict[str, Any]]]
    column_analysis: Optional[Dict[str, Any]]
    optimized_story_board: Optional[Dict[str, Any]]
    optimized_column_assignment: Optional[List[Dict[str, Any]]]
    balancer_decisions: Optional[Dict[str, Any]]
    final_column_analysis: Optional[Dict[str, Any]]

    # poster assets
    url: str
    logo_path: str
    aff_logo_path: Optional[str]
    doi: Optional[str]
    conference_name: Optional[str]
    enable_visual_refinement: bool
    enable_affiliation_logos: bool
    affiliation_logo_mode: str
    enable_vlm_layout_review: bool
    enable_visual_legibility_review: bool
    enable_block_vlm_review: bool
    enable_adaptive_column_width: bool
    enable_generated_background: bool
    enable_generated_teaser: bool
    background_palette: Optional[str]
    background_style: Optional[str]
    poster_style_preset: Optional[str]
    visual_density: Optional[str]
    section_title_numbering: Optional[str]
    header_route: Optional[str]
    header_subtitle_policy: Optional[str]
    header_title_wrap_policy: Optional[str]
    header_seed: Optional[int]
    background_image_path: Optional[str]
    background_image_report: Optional[Dict[str, Any]]
    generated_teaser_report: Optional[Dict[str, Any]]
    vlm_model: Optional[str]
    render_stage: str
    draft_status: str
    draft_rejection_reason: Optional[str]
    final_poster_accepted: bool
    vlm_review_count: int
    vlm_reflow_required: bool
    vlm_patch_applied: bool
    adaptive_relayout_required: bool
    adaptive_relayout_count: int
    template_repair_required: bool
    template_repair_count: int
    template_repair_decision: Optional[Dict[str, Any]]
    poster_preview_path: Optional[str]
    pptx_output_path: Optional[str]
    visual_reflow_required: bool
    visual_reflow_count: int
    block_refinement_required: bool
    block_refinement_count: int

    # metadata
    tokens: TokenUsage
    timing_metrics: TimingMetrics
    degraded_quality_states: List[Dict[str, Any]]
    current_agent: str
    errors: List[str]


def create_state(
    pdf_path: str,
    text_model: str = "gpt-4.1-2025-04-14",
    vision_model: str = "gpt-4.1-2025-04-14",
    width: int = 84,
    height: int = 42,
    layout_template: str = "three_column_postergen",
    url: str = "",
    logo_path: str = "",
    aff_logo_path: str = "",
    enable_visual_refinement: bool = False,
    enable_affiliation_logos: bool = False,
    affiliation_logo_mode: str = "single",
    enable_vlm_layout_review: bool = False,
    enable_visual_legibility_review: bool = False,
    enable_block_vlm_review: bool = False,
    enable_adaptive_column_width: bool = False,
    enable_generated_background: bool = False,
    enable_generated_teaser: bool = False,
    background_palette: Optional[str] = None,
    background_style: Optional[str] = None,
    poster_style_preset: Optional[str] = None,
    visual_density: Optional[str] = None,
    section_title_numbering: Optional[str] = None,
    header_route: Optional[str] = None,
    header_subtitle_policy: Optional[str] = None,
    header_title_wrap_policy: Optional[str] = None,
    header_seed: Optional[int] = None,
    vlm_model: Optional[str] = None,
    conference_name: Optional[str] = None,
) -> PosterState:
    """create initial poster state"""
    from pathlib import Path

    poster_name = Path(pdf_path).parent.name or "test_poster"
    output_root = Path(os.getenv("PAPER2POSTER_OUTPUT_ROOT", "output")).expanduser()
    output_dir = str(output_root / poster_name)

    needs_post_render_pass = any(
        [
            enable_vlm_layout_review,
            enable_visual_legibility_review,
            enable_block_vlm_review,
            enable_generated_background,
        ]
    )

    return PosterState(
        pdf_path=pdf_path,
        output_dir=output_dir,
        poster_name=poster_name,
        text_model=_get_model_config(text_model),
        vision_model=_get_model_config(vision_model),
        raw_text=None,
        images=None,
        tables=None,
        visual_assets=None,
        resolved_visual_assets=None,
        visual_plan=None,
        affiliations=None,
        affiliation_logos=None,
        vlm_layout_review=None,
        vlm_layout_patch=None,
        visual_legibility_review=None,
        block_occupancy_report=None,
        block_vlm_review=None,
        block_content_patch=None,
        block_refinement_history=None,
        adaptive_layout_decision=None,
        narrative=None,
        poster_plan=None,
        paper_poster_keypoints=None,
        poster_reading_order=None,
        poster_keypoint_selection_report=None,
        poster_width=width,
        poster_height=height,
        layout_template=layout_template,
        poster_variant={
            "variant_id": "default_standard" if layout_template == "auto" else "manual",
            "requested_template": layout_template,
            "resolved_template": None,
            "style_profile": poster_style_preset,
            "visual_density": visual_density,
            "generated_teaser": {"enabled": enable_generated_teaser},
            "generated_background": {
                "enabled": enable_generated_background,
                "palette": background_palette,
                "style": background_style,
            },
            "seed": None,
        },
        resolved_layout_template=None,
        layout_template_metadata=None,
        template_selection_report=None,
        standard_template_selection_report=None,
        adaptive_lane_widths=None,
        template_layout_mode=None,
        template_block_plan=None,
        layout_intent=None,
        header_plan=None,
        header_block_review=None,
        header_block_patch_applied=False,
        template_prior_source_story_board=None,
        template_fast_mode=False,
        fast_block_contract=None,
        fast_visual_policy=None,
        fast_pipeline_report=None,
        block_capacity_contract=None,
        capacity_aware_story_board=None,
        capacity_planning_report=None,
        slot_pressure_report=None,
        wireframe_layout=None,
        content_filled_layout=None,
        final_layout=None,
        narrative_content=None,
        classified_visuals=None,
        structured_sections=None,
        story_board=None,
        curated_content=None,
        design_layout=None,
        section_title_design=None,
        color_scheme=None,
        keywords=None,
        styled_layout=None,
        styling_interfaces=None,
        initial_layout_data=None,
        column_analysis=None,
        optimized_story_board=None,
        optimized_column_assignment=None,
        balancer_decisions=None,
        final_column_analysis=None,
        url=url,
        logo_path=logo_path,
        aff_logo_path=aff_logo_path,
        doi=None,
        conference_name=conference_name,
        enable_visual_refinement=enable_visual_refinement,
        enable_affiliation_logos=enable_affiliation_logos,
        affiliation_logo_mode=affiliation_logo_mode,
        enable_vlm_layout_review=enable_vlm_layout_review,
        enable_visual_legibility_review=enable_visual_legibility_review,
        enable_block_vlm_review=enable_block_vlm_review,
        enable_adaptive_column_width=enable_adaptive_column_width,
        enable_generated_background=enable_generated_background,
        enable_generated_teaser=enable_generated_teaser,
        background_palette=background_palette,
        background_style=background_style,
        poster_style_preset=poster_style_preset,
        visual_density=visual_density,
        section_title_numbering=section_title_numbering,
        header_route=header_route,
        header_subtitle_policy=header_subtitle_policy,
        header_title_wrap_policy=header_title_wrap_policy,
        header_seed=header_seed,
        background_image_path=None,
        background_image_report=None,
        generated_teaser_report=None,
        vlm_model=vlm_model,
        render_stage="draft" if needs_post_render_pass else "final",
        draft_status="pending",
        draft_rejection_reason=None,
        final_poster_accepted=False,
        vlm_review_count=0,
        vlm_reflow_required=False,
        vlm_patch_applied=False,
        adaptive_relayout_required=False,
        adaptive_relayout_count=0,
        template_repair_required=False,
        template_repair_count=0,
        template_repair_decision=None,
        poster_preview_path=None,
        pptx_output_path=None,
        visual_reflow_required=False,
        visual_reflow_count=0,
        block_refinement_required=False,
        block_refinement_count=0,
        tokens=TokenUsage(),
        timing_metrics=TimingMetrics(),
        degraded_quality_states=[],
        current_agent="init",
        errors=[]
    )


def _get_model_config(model_id: str) -> ModelConfig:
    """Resolve a model id (typically from --text_model / --vision_model on the CLI,
    which already override env defaults) into a ModelConfig.

    Resolution order:
      1. exact known alias in the table below;
      2. explicit ``provider/model`` form -> ModelConfig(model, provider);
      3. OpenAI-family passthrough (gpt-*, o1/o3/o4, chatgpt-*) -> provider "openai",
         so any OpenAI model name (e.g. ``gpt-4o``) works even if not pre-listed
         instead of silently degrading to the gpt-4.1 fallback;
      4. otherwise fall back to gpt-4.1.
    """
    configs = {
        "claude": ModelConfig("claude-sonnet-4-20250514", "anthropic"),
        "claude-sonnet-4-20250514": ModelConfig("claude-sonnet-4-20250514", "anthropic"),
        "claude-opus-4.5": ModelConfig("claude-opus-4-5-20251101", "anthropic"),
        "claude-opus-4-5-20251101": ModelConfig("claude-opus-4-5-20251101", "anthropic"),
        "gemini": ModelConfig("gemini-2.5-pro", "google"),
        "gemini-2.5-pro": ModelConfig("gemini-2.5-pro", "google"),
        "gpt-4o-2024-08-06": ModelConfig("gpt-4o-2024-08-06", "openai"),
        "gpt-4.1-2025-04-14": ModelConfig("gpt-4.1-2025-04-14", "openai"),
        "gpt-4.1-mini-2025-04-14": ModelConfig("gpt-4.1-mini-2025-04-14", "openai"),
        "gpt-5": ModelConfig("gpt-5", "openai"),
        "gpt-5.1": ModelConfig("gpt-5.1", "openai"),
        "gpt-5.4-xhigh": ModelConfig("gpt-5.4-xhigh", "openai"),
        "gpt-5.5-xhigh": ModelConfig("gpt-5.5-xhigh", "openai"),
        "gpt-5.4": ModelConfig("gpt-5.4", "openai"),
        "glm-4.6": ModelConfig("glm-4.6", "zhipu"),
        "glm-4.6v": ModelConfig("glm-4.6v", "zhipu"),
        "glm-4.5": ModelConfig("glm-4.5", "zhipu"),
        "glm-4.5-air": ModelConfig("glm-4.5-air", "zhipu"),
        "glm-4.5v": ModelConfig("glm-4.5v", "zhipu"),
        "glm-4": ModelConfig("glm-4", "zhipu"),
        "glm-4v": ModelConfig("glm-4v", "zhipu"),
        "kimi-k2-turbo-preview": ModelConfig("kimi-k2-turbo-preview", "moonshot"),
        "moonshot-v1-8k-vision-preview": ModelConfig("moonshot-v1-8k-vision-preview", "moonshot"),
        "MiniMax-M2": ModelConfig("MiniMax-M2", "Minimax"),
        "qwen3-max": ModelConfig("qwen3-max", "Alibaba"),
        "qwen3-vl-plus": ModelConfig("qwen3-vl-plus", "Alibaba"),
        "gpt-4o": ModelConfig("gpt-4o", "openai"),
    }
    if not model_id:
        return configs["gpt-4.1-2025-04-14"]
    model_id = model_id.strip()
    if model_id in configs:
        return configs[model_id]
    if "/" in model_id:
        provider, _, name = model_id.partition("/")
        provider, name = provider.strip(), name.strip()
        if provider and name:
            return ModelConfig(name, provider)
    if model_id.lower().startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return ModelConfig(model_id, "openai")
    return configs["gpt-4.1-2025-04-14"]
