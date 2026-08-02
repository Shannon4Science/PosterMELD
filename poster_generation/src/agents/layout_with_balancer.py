"""
3-phase layout optimization orchestrator
"""

import json
from pathlib import Path
from typing import Dict, Any
from src.state.poster_state import PosterState
from src.agents.layout_agent import LayoutAgent
from src.agents.balancer_agent import BalancerAgent
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

class LayoutWithBalancerAgent:
    def __init__(self):
        self.name = "layout_with_balancer"
        self.layout_agent = LayoutAgent()
        self.balancer_agent = BalancerAgent()

    def __call__(self, state: PosterState) -> PosterState:
        """execute 3-phase layout optimization"""
        log_agent_info(self.name, "starting 3-phase layout optimization")
        
        try:
            prior_errors = list(state.get("errors") or [])

            # phase 1: initial layout generation
            log_agent_info(self.name, "phase 1: generating initial layout")
            initial_state = self.layout_agent(state, mode="initial")
            if self._new_errors(initial_state, prior_errors):
                return initial_state

            if initial_state.get("template_layout_mode") == "template_prior":
                log_agent_info(self.name, "template prior mode: skipping legacy three-column balancer")
                slot_pressure_report = self._build_slot_pressure_report(
                    initial_state["initial_layout_data"],
                    initial_state["column_analysis"],
                    initial_state.get("layout_template_metadata") or {},
                )
                initial_state["slot_pressure_report"] = slot_pressure_report
                initial_state["optimized_story_board"] = initial_state.get("story_board")
                initial_state["balancer_decisions"] = {
                    "mode": "template_block_passthrough",
                    "reason": "slot-driven templates already map one section per slot",
                }
                self._save_balancer_output(
                    {
                        "optimized_story_board": initial_state["optimized_story_board"],
                        "balancer_decisions": initial_state["balancer_decisions"],
                    },
                    initial_state,
                )
                final_state = self.layout_agent(initial_state, mode="final")
                final_state["slot_pressure_report"] = slot_pressure_report
                return final_state
            
            # phase 2: balancer optimization  
            log_agent_info(self.name, "phase 2: optimizing with balancer")
            balancer_result = self.balancer_agent(
                initial_layout_data=initial_state["initial_layout_data"],
                column_analysis=initial_state["column_analysis"],
                state=initial_state
            )
            
            # save balancer decisions
            self._save_balancer_output(balancer_result, initial_state)
            
            # update state with optimized story board
            initial_state["optimized_story_board"] = balancer_result["optimized_story_board"]
            initial_state["balancer_decisions"] = balancer_result["balancer_decisions"]
            
            # phase 3: final layout generation
            log_agent_info(self.name, "phase 3: generating final layout")
            before_final_errors = list(initial_state.get("errors") or [])
            final_state = self.layout_agent(initial_state, mode="final")
            if self._new_errors(final_state, before_final_errors):
                return final_state
            
            # update token counts
            final_state["tokens"].add_text(
                balancer_result.get("input_tokens", 0),
                balancer_result.get("output_tokens", 0)
            )
            
            log_agent_success(self.name, "3-phase layout optimization complete")
            return final_state
            
        except Exception as e:
            log_agent_error(self.name, f"3-phase optimization error: {e}")
            return {**state, "errors": state.get("errors", []) + [f"{self.name}: {e}"]}

    def _new_errors(self, state: PosterState, prior_errors: list[str]) -> list[str]:
        prior_counts: Dict[str, int] = {}
        for error in prior_errors:
            prior_counts[str(error)] = prior_counts.get(str(error), 0) + 1

        new_errors: list[str] = []
        seen_counts: Dict[str, int] = {}
        for error in state.get("errors") or []:
            key = str(error)
            seen_counts[key] = seen_counts.get(key, 0) + 1
            if seen_counts[key] > prior_counts.get(key, 0):
                new_errors.append(key)
        return new_errors

    def _save_balancer_output(self, balancer_result: Dict, state: PosterState):
        """save balancer optimization results"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "optimized_story_board.json", "w", encoding='utf-8') as f:
            json.dump(balancer_result["optimized_story_board"], f, indent=2)
        
        with open(output_dir / "balancer_decisions.json", "w", encoding='utf-8') as f:
            json.dump(balancer_result["balancer_decisions"], f, indent=2)

    def _build_slot_pressure_report(
        self,
        initial_layout_data: Any,
        column_analysis: Dict[str, Any],
        template_layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        slot_report: Dict[str, Any] = {}
        columns = column_analysis.get("columns", {})
        for lane in template_layout.get("lanes", []):
            lane_id = lane["id"]
            column = columns.get(lane_id, {})
            available = float(column.get("available_height", lane.get("h", 0.0)) or lane.get("h", 0.0) or 0.1)
            used = float(column.get("total_height", 0.0) or 0.0)
            slot_report[lane_id] = {
                "slot_id": lane_id,
                "pressure": round(used / max(available, 0.1), 4),
                "used_height": round(used, 4),
                "available_height": round(available, 4),
                "available_width": round(float(column.get("width", lane.get("w", 0.0))), 4),
            }
        return {"slots": slot_report}


def layout_with_balancer_node(state: PosterState) -> Dict[str, Any]:
    """layout with balancer node for langgraph"""
    try:
        agent = LayoutWithBalancerAgent()
        result = agent(state)
        
        return {
            **state,
            "initial_layout_data": result.get("initial_layout_data"),
            "column_analysis": result.get("column_analysis"),
            "design_layout": result.get("design_layout"),
            "final_column_analysis": result.get("final_column_analysis"),
            "optimized_column_assignment": result.get("optimized_column_assignment"),
            "optimized_story_board": result.get("optimized_story_board"),
            "balancer_decisions": result.get("balancer_decisions"),
            "resolved_layout_template": result.get("resolved_layout_template"),
            "layout_template_metadata": result.get("layout_template_metadata"),
            "template_selection_report": result.get("template_selection_report"),
            "adaptive_lane_widths": result.get("adaptive_lane_widths"),
            "template_layout_mode": result.get("template_layout_mode"),
            "template_block_plan": result.get("template_block_plan"),
            "layout_intent": result.get("layout_intent"),
            "slot_pressure_report": result.get("slot_pressure_report"),
            "tokens": result.get("tokens"),
            "current_agent": result.get("current_agent"),
            "errors": result.get("errors", [])
        }
    except Exception as e:
        log_agent_error("layout_with_balancer", f"node error: {e}")
        return {**state, "errors": state.get("errors", []) + [f"layout_with_balancer: {e}"]}
