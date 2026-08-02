"""
pdf text and asset extraction
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import fitz
from marker.converters.pdf import PdfConverter
from marker.renderers.markdown import MarkdownRenderer
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.schema import BlockTypes
from jinja2 import Template
from PIL import Image

from src.state.poster_state import PosterState
from src.tools.mineru_api import MinerUClient, MinerUExtraction
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config
from src.utils.text_cleanup import normalize_text_for_poster, normalize_title_for_poster


class Parser:
    def __init__(self):
        self.name = "parser"
        config_data = load_config()
        self.config_data = config_data
        batch_config = config_data["pdf_processing"]["batch_sizes"]
        self.marker_config = {
            "recognition_batch_size": batch_config["recognition"],
            "layout_batch_size": batch_config["layout"],
            "detection_batch_size": batch_config["detection"], 
            "table_rec_batch_size": batch_config["table_rec"],
            "ocr_error_batch_size": batch_config["ocr_error"],
            "equation_batch_size": batch_config["equation"],
            "disable_tqdm": False,
        }
        
        self.converter = None
        self.clean_pattern = re.compile(r"<!--[\s\S]*?-->")
        self.enhanced_abt_prompt = load_prompt("config/prompts/narrative_abt_extraction.txt")
        self.visual_classification_prompt = load_prompt("config/prompts/classify_visuals.txt")
        self.title_authors_prompt = load_prompt("config/prompts/extract_title_authors.txt")
        self.section_extraction_prompt = load_prompt("config/prompts/extract_structured_sections.txt")
    
    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting foundation building")
        
        try:
            output_dir = Path(state["output_dir"])
            content_dir = output_dir / "content"
            assets_dir = output_dir / "assets"
            content_dir.mkdir(parents=True, exist_ok=True)
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            # extract raw text and assets
            raw_text, raw_result = self._extract_raw_text(state["pdf_path"], content_dir)
            prompt_text = self._prompt_source_text(raw_text)

            figures, tables = self._extract_assets(raw_result, state["poster_name"], assets_dir)
            cached_narrative = self._load_cached_content(content_dir, "narrative_content.json")
            cached_classified_visuals = self._load_cached_content(content_dir, "classified_visuals.json")
            cached_structured_sections = self._load_cached_content(content_dir, "structured_sections.json")
            
            cached_title_authors = self._title_authors_from_cached_narrative(cached_narrative)
            try:
                title, authors = self._extract_title_authors(raw_text, state["text_model"], state)
                if self._is_missing_title_authors(title, authors) and cached_title_authors:
                    title, authors = cached_title_authors
                    log_agent_warning(self.name, "using cached title/authors after incomplete model extraction")
            except Exception as exc:
                if cached_title_authors:
                    title, authors = cached_title_authors
                    log_agent_warning(self.name, f"using cached title/authors after model failure: {exc}")
                else:
                    raise
            affiliations = self._extract_affiliations(raw_text)
            doi = self._extract_doi(raw_text)

            try:
                narrative_content, inp_tok, out_tok = self._generate_narrative_content(prompt_text, state["text_model"], state)
            except Exception as exc:
                if self._validate_narrative_content(cached_narrative):
                    log_agent_warning(self.name, f"using cached narrative content after model failure: {exc}")
                    narrative_content, inp_tok, out_tok = cached_narrative, 0, 0
                else:
                    raise
            state["tokens"].add_text(inp_tok, out_tok)

            try:
                classified_visuals, inp_tok2, out_tok2 = self._classify_visual_assets(figures, tables, raw_text, state["text_model"], state)
            except Exception as exc:
                if self._validate_classified_visuals(cached_classified_visuals):
                    log_agent_warning(self.name, f"using cached visual classification after model failure: {exc}")
                    classified_visuals, inp_tok2, out_tok2 = cached_classified_visuals, 0, 0
                else:
                    raise
            state["tokens"].add_text(inp_tok2, out_tok2)

            narrative_content["meta"] = {
                "poster_title": title,
                "authors": authors,
                "affiliations": affiliations,
            }

            try:
                structured_sections = self._extract_structured_sections(prompt_text, state["text_model"], state)
            except Exception as exc:
                if self._validate_structured_sections(cached_structured_sections):
                    log_agent_warning(self.name, f"using cached structured sections after model failure: {exc}")
                    structured_sections = cached_structured_sections
                else:
                    raise
            
            # save artifacts and update state
            self._save_content(narrative_content, "narrative_content.json", content_dir)
            self._save_content(classified_visuals, "classified_visuals.json", content_dir)
            self._save_content(structured_sections, "structured_sections.json", content_dir)
            self._save_raw_text(raw_text, content_dir)
            visual_assets = self._build_visual_registry(figures, tables)
            self._save_content(visual_assets, "visual_assets.json", content_dir)
            
            state["raw_text"] = prompt_text
            state["structured_sections"] = structured_sections
            state["narrative_content"] = narrative_content
            state["affiliations"] = affiliations
            state["doi"] = doi
            state["classified_visuals"] = classified_visuals
            state["images"] = figures
            state["tables"] = tables
            state["visual_assets"] = visual_assets
            state["current_agent"] = self.name
            
            log_agent_success(self.name, f"extracted raw text, {len(figures)} images, and {len(tables)} tables")
            log_agent_success(self.name, f"extracted title: {title}")
            log_agent_success(self.name, f"extracted affiliations: {', '.join(affiliations) if affiliations else 'none'}")
            log_agent_success(self.name, "generated enhanced abt narrative")
            log_agent_success(self.name, f"classified visuals: key={classified_visuals.get('key_visual', 'none')}, problem_ill={len(classified_visuals.get('problem_illustration', []))}, method_wf={len(classified_visuals.get('method_workflow', []))}, main_res={len(classified_visuals.get('main_results', []))}, comp_res={len(classified_visuals.get('comparative_results', []))}, support={len(classified_visuals.get('supporting', []))}")
            
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(str(e))
        
        return state

    def _load_cached_content(self, content_dir: Path, filename: str) -> Dict[str, Any]:
        path = content_dir / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return content if isinstance(content, dict) else {}
        except Exception as exc:
            log_agent_warning(self.name, f"ignored unreadable cached {filename}: {exc}")
            return {}

    def _validate_narrative_content(self, narrative: Dict[str, Any]) -> bool:
        return isinstance(narrative, dict) and all(key in narrative for key in ("and", "but", "therefore"))

    def _validate_classified_visuals(self, classified_visuals: Dict[str, Any]) -> bool:
        if not isinstance(classified_visuals, dict):
            return False
        expected_keys = {
            "key_visual",
            "problem_illustration",
            "method_workflow",
            "main_results",
            "comparative_results",
            "supporting",
        }
        return any(key in classified_visuals for key in expected_keys)

    def _is_missing_title_authors(self, title: str, authors: str) -> bool:
        return not title or not authors or title == "Untitled" or authors == "Authors not found"

    def _title_authors_from_cached_narrative(self, narrative: Dict[str, Any]) -> Tuple[str, str] | None:
        meta = narrative.get("meta") if isinstance(narrative, dict) else None
        if not isinstance(meta, dict):
            return None
        title = normalize_title_for_poster(str(meta.get("poster_title") or meta.get("title") or ""))
        authors = normalize_text_for_poster(str(meta.get("authors") or ""))
        if title and authors:
            return title, authors
        return None
    
    def _extract_raw_text(self, pdf_path: str, content_dir: Path) -> Tuple[str, Any]:
        pdf_config = self.config_data.get("pdf_processing", {}) if hasattr(self, "config_data") else {}
        backend = os.getenv("PDF_PARSER_BACKEND", str(pdf_config.get("backend") or "mineru")).strip().lower()
        fallback_backend = str(pdf_config.get("fallback_backend") or "marker").strip().lower()

        if backend == "mineru":
            try:
                extraction = self._extract_raw_text_with_mineru(pdf_path, content_dir)
                self._write_mineru_report(content_dir, extraction.report)
                return extraction.raw_text, extraction
            except Exception as exc:
                log_agent_warning(self.name, f"MinerU extraction failed; falling back to marker: {exc}")
                self._write_mineru_report(content_dir, {
                    "backend": "mineru",
                    "model_version": os.getenv("MINERU_MODEL_VERSION", str((pdf_config.get("mineru") or {}).get("model_version") or "vlm")),
                    "fallback_used": True,
                    "fallback_backend": fallback_backend,
                    "error_summary": str(exc)[:1000],
                })
                if fallback_backend != "marker":
                    raise

        return self._extract_raw_text_with_marker(pdf_path, content_dir)

    def _extract_raw_text_with_mineru(self, pdf_path: str, content_dir: Path) -> MinerUExtraction:
        mineru_config = (self.config_data.get("pdf_processing", {}) if hasattr(self, "config_data") else {}).get("mineru", {})
        client = MinerUClient.from_env(mineru_config)
        log_agent_info(self.name, f"converting pdf to raw text with MinerU ({client.model_version})")
        extraction = client.parse_pdf(pdf_path, content_dir)
        log_agent_info(self.name, f"MinerU extracted {len(extraction.raw_text)} chars")
        return extraction

    def _extract_raw_text_with_marker(self, pdf_path: str, content_dir: Path) -> Tuple[str, Any]:
        log_agent_info(self.name, "converting pdf to raw text")
        converter = self._marker_converter()
        document = converter.build_document(pdf_path)
        
        # create renderer and get rendered output from the existing document
        renderer = converter.resolve_dependencies(MarkdownRenderer)
        rendered = renderer(document)
        
        text, _, images = text_from_rendered(rendered)
        text = self.clean_pattern.sub("", text)
        
        (content_dir / "raw.md").write_text(text, encoding="utf-8")
        
        log_agent_info(self.name, f"extracted {len(text)} chars")
        
        raw_result = (document, rendered, images)
        return text, raw_result

    def _marker_converter(self):
        if self.converter is None:
            self.converter = PdfConverter(artifact_dict=create_model_dict(), config=self.marker_config)
        return self.converter

    def _write_mineru_report(self, content_dir: Path, report: Dict[str, Any]) -> None:
        sanitized = dict(report)
        sanitized.pop("api_key", None)
        with open(content_dir / "mineru_report.json", "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False)

    def _extract_doi(self, raw_text: str) -> str | None:
        """Extract a paper DOI from the header area, not from references."""
        header = self._paper_header_for_metadata(raw_text)
        patterns = [
            r"10\.\d{4,9}/[^\s\"'<>\]\)]+",
            r"doi\.org/(10\.\d{4,9}/[^\s\"'<>\]\)]+)",
            r"DOI[:\s]+(10\.\d{4,9}/[^\s\"'<>\]\)]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, header, flags=re.IGNORECASE)
            if m:
                doi = m.group(1) if m.lastindex else m.group(0)
                doi = doi.rstrip(".,;)")
                return doi
        return None

    def _extract_affiliations(self, raw_text: str) -> list[str]:
        """Extract likely institution names from the paper header before abstract."""
        header = self._paper_header_for_metadata(raw_text)
        header = re.sub(r"<sup>.*?</sup>", "", header)
        header = re.sub(r"\s+", " ", header)

        candidates: list[str] = []
        specific_patterns = [
            r"Brown School at Washington University in St\. Louis",
            r"Washington University in St\. Louis",
            r"George Mason University",
            r"Tsinghua University",
            r"Beijing University of Posts and Telecommunications",
            r"Beijing University\s+of\s+Posts\s+and\s+Telecommunications",
            r"The Chinese University of Hong ?Kong",
            r"University of Illinois at Chicago",
            r"University of Illinois Chicago",
            r"Hong ?Kong University of Science and Technology(?: \(Guangzhou\))?",
            r"Hong ?Kong University of Science and Technology",
        ]
        for pattern in specific_patterns:
            for match in re.finditer(pattern, header, flags=re.IGNORECASE):
                candidates.append(self._normalize_affiliation_name(match.group(0)))

        generic_pattern = r"([A-Z][A-Za-z&.\- ]{2,80}?(?:University|Institute|College|School)(?: in [A-Z][A-Za-z.\- ]{2,40})?)"
        for match in re.finditer(generic_pattern, header):
            name = self._normalize_affiliation_name(match.group(1))
            if self._looks_like_institution(name):
                candidates.append(name)

        deduped: list[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if len(candidate.split()) <= 3 and any(key in existing and key != existing for existing in seen):
                continue
            if key and key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped[:6]

    def _paper_header_for_metadata(self, raw_text: str) -> str:
        for marker in ("#### Abstract", "A BSTRACT", "\nAbstract", "\nABSTRACT"):
            if marker in raw_text:
                return raw_text.split(marker, 1)[0]
        return raw_text[:6000]

    def _prompt_source_text(self, raw_text: str) -> str:
        """Bound model-facing paper text while preserving the full raw.md artifact."""
        try:
            max_chars = int(os.getenv("PAPER2POSTER_PARSER_MAX_CHARS", "0") or 0)
        except ValueError:
            max_chars = 0
        if max_chars <= 0 or len(raw_text) <= max_chars:
            return raw_text

        head_chars = max(int(max_chars * 0.72), 1)
        tail_chars = max(max_chars - head_chars, 1)
        log_agent_warning(
            self.name,
            f"limiting model-facing paper text from {len(raw_text)} to {max_chars} chars",
        )
        return (
            raw_text[:head_chars].rstrip()
            + "\n\n[Middle sections omitted only for model context safety; full text remains in raw.md.]\n\n"
            + raw_text[-tail_chars:].lstrip()
        )

    def _normalize_affiliation_name(self, name: str) -> str:
        name = re.sub(r"^(Department|Division|School|College) of [^,]+,\s*", "", name.strip(), flags=re.IGNORECASE)
        name = re.sub(r"^(USA|United States|U\.S\.A\.|UK|Canada|China)\s+", "", name, flags=re.IGNORECASE)
        name = name.replace("Hongkong", "Hong Kong")
        name = re.sub(r"\s+", " ", name).strip(" ,.;")
        return name

    def _looks_like_institution(self, name: str) -> bool:
        lowered = name.lower()
        reject_terms = {"abstract", "copyright", "figure", "table"}
        reject_prefixes = ("usa ", "united states ", "u.s.a. ")
        return (
            1 < len(name.split()) <= 10
            and any(token in lowered for token in ("university", "institute", "college", "school"))
            and not any(term in lowered for term in reject_terms)
            and not lowered.startswith(reject_prefixes)
        )

    def _generate_narrative_content(self, text: str, config, state) -> Tuple[Dict, int, int]:
        log_agent_info(self.name, "generating abt narrative")
        agent = LangGraphAgent("expert poster design consultant", config, state, "parser")
        
        for attempt in range(3):
            try:
                prompt = Template(self.enhanced_abt_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)
                
                narrative = extract_json(response.content)
                
                if "and" in narrative and "but" in narrative and "therefore" in narrative:
                    return narrative, response.input_tokens, response.output_tokens

            except Exception as e:
                log_agent_warning(self.name, f"attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise

        raise ValueError("failed to generate enhanced narrative after 3 attempts")
    
    def _save_content(self, content: Dict, filename: str, content_dir: Path):
        with open(content_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
    
    def _save_raw_text(self, raw_text: str, content_dir: Path):
        with open(content_dir / "raw.md", 'w', encoding='utf-8') as f:
            f.write(raw_text)
    
    def _extract_assets(self, result, name: str, assets_dir: Path) -> Tuple[Dict, Dict]:
        log_agent_info(self.name, "extracting assets")

        if isinstance(result, MinerUExtraction):
            return self._extract_mineru_assets(result, assets_dir)
        
        document, rendered, marker_images = result
        
        caption_map = self._extract_captions(document)
        
        figures = {}
        tables = {}
        image_count = 0
        table_count = 0
        
        for img_name, pil_image in marker_images.items():
            caption_info = caption_map.get(img_name, {'captions': [], 'block_type': 'Unknown'})
            
            if 'table' in img_name.lower() or 'Table' in img_name or caption_info.get('block_type') == 'Table':
                table_count += 1
                path = assets_dir / f"table-{table_count}.png"
                pil_image.save(path, "PNG")
                
                tables[str(table_count)] = {
                    'caption': caption_info['captions'][0] if caption_info['captions'] else f"Table {table_count}",
                    'path': str(path),
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'aspect': pil_image.width / pil_image.height if pil_image.height > 0 else 1,
                }
            else:
                image_count += 1
                path = assets_dir / f"figure-{image_count}.png"
                pil_image.save(path, "PNG")
                
                figures[str(image_count)] = {
                    'caption': caption_info['captions'][0] if caption_info['captions'] else f"Figure {image_count}",
                    'path': str(path),
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'aspect': pil_image.width / pil_image.height if pil_image.height > 0 else 1,
                }
        
        with open(assets_dir / "figures.json", 'w', encoding='utf-8') as f:
            json.dump(figures, f, indent=2)
        with open(assets_dir / "tables.json", 'w', encoding='utf-8') as f:
            json.dump(tables, f, indent=2)
        with open(assets_dir / "fig_tab_caption_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)
        
        return figures, tables

    def _extract_mineru_assets(self, extraction: MinerUExtraction, assets_dir: Path) -> Tuple[Dict, Dict]:
        figures: Dict[str, Dict[str, Any]] = {}
        tables: Dict[str, Dict[str, Any]] = {}
        caption_map: Dict[str, Dict[str, Any]] = {}
        image_count = 0
        table_count = 0
        pdf_document = None
        if extraction.pdf_path and Path(extraction.pdf_path).exists():
            try:
                pdf_document = fitz.open(str(extraction.pdf_path))
            except Exception as exc:
                log_agent_warning(self.name, f"could not open source PDF for high-resolution asset rendering: {exc}")

        try:
            for item in extraction.content_items:
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"image", "chart", "figure"}:
                    image_count += 1
                    asset = self._copy_mineru_asset(
                        extraction,
                        item,
                        assets_dir / f"figure-{image_count}.png",
                        fallback_caption=f"Figure {image_count}",
                        caption_keys=("image_caption", "caption", "img_caption"),
                        pdf_document=pdf_document,
                    )
                    if asset:
                        figures[str(image_count)] = asset
                        caption_map[f"figure-{image_count}.png"] = self._mineru_caption_entry(item, asset["caption"])
                    else:
                        image_count -= 1
                elif item_type == "table":
                    table_count += 1
                    asset = self._copy_mineru_asset(
                        extraction,
                        item,
                        assets_dir / f"table-{table_count}.png",
                        fallback_caption=f"Table {table_count}",
                        caption_keys=("table_caption", "caption"),
                        pdf_document=pdf_document,
                    )
                    if asset:
                        tables[str(table_count)] = asset
                        caption_map[f"table-{table_count}.png"] = self._mineru_caption_entry(item, asset["caption"])
                    else:
                        table_count -= 1
        finally:
            if pdf_document is not None:
                pdf_document.close()

        with open(assets_dir / "figures.json", 'w', encoding='utf-8') as f:
            json.dump(figures, f, indent=2)
        with open(assets_dir / "tables.json", 'w', encoding='utf-8') as f:
            json.dump(tables, f, indent=2)
        with open(assets_dir / "fig_tab_caption_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)

        extraction.report.update({
            "figure_count": len(figures),
            "table_count": len(tables),
            "pdf_bbox_render_count": sum(
                1
                for asset in [*figures.values(), *tables.values()]
                if asset.get("extraction_method") == "pdf_bbox_render"
            ),
        })
        self._write_mineru_report(extraction.extract_dir.parent, extraction.report)

        return figures, tables

    def _copy_mineru_asset(
        self,
        extraction: MinerUExtraction,
        item: Dict[str, Any],
        target_path: Path,
        *,
        fallback_caption: str,
        caption_keys: Tuple[str, ...],
        pdf_document=None,
    ) -> Dict[str, Any] | None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_size = self._render_mineru_bbox_asset(pdf_document, item, target_path)
        if rendered_size is not None:
            width, height = rendered_size
            return {
                "caption": self._mineru_caption(item, caption_keys, fallback_caption),
                "path": str(target_path),
                "width": width,
                "height": height,
                "aspect": width / height if height > 0 else 1,
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "extraction_method": "pdf_bbox_render",
            }

        source = self._resolve_mineru_asset_path(extraction, item)
        if source is None or not source.exists():
            log_agent_warning(self.name, f"MinerU asset missing img_path: {item.get('img_path') or item.get('image_path')}")
            return None

        try:
            with Image.open(source) as image:
                image.convert("RGB").save(target_path, "PNG")
                width, height = image.size
        except Exception:
            shutil.copyfile(source, target_path)
            with Image.open(target_path) as image:
                width, height = image.size

        return {
            "caption": self._mineru_caption(item, caption_keys, fallback_caption),
            "path": str(target_path),
            "width": width,
            "height": height,
            "aspect": width / height if height > 0 else 1,
            "page_idx": item.get("page_idx"),
            "bbox": item.get("bbox"),
            "extraction_method": "mineru_image_copy",
        }

    def _render_mineru_bbox_asset(
        self,
        pdf_document,
        item: Dict[str, Any],
        target_path: Path,
    ) -> Tuple[int, int] | None:
        if pdf_document is None:
            return None
        bbox = item.get("bbox")
        page_idx = item.get("page_idx")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            page_idx = int(page_idx)
            values = [float(value) for value in bbox]
            if page_idx < 0 or page_idx >= len(pdf_document):
                return None
            if values[2] <= values[0] or values[3] <= values[1]:
                return None

            page = pdf_document[page_idx]
            page_rect = page.rect
            # MinerU content-list bboxes use a normalized 0..1000 page space.
            clip = fitz.Rect(
                page_rect.x0 + values[0] / 1000.0 * page_rect.width,
                page_rect.y0 + values[1] / 1000.0 * page_rect.height,
                page_rect.x0 + values[2] / 1000.0 * page_rect.width,
                page_rect.y0 + values[3] / 1000.0 * page_rect.height,
            ) & page_rect
            if clip.width <= 1 or clip.height <= 1:
                return None

            target_long_edge = 2200.0
            minimum_scale = 300.0 / 72.0
            scale = max(minimum_scale, target_long_edge / max(clip.width, clip.height))
            scale = min(scale, 18.0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            pixmap.save(str(target_path))
            return int(pixmap.width), int(pixmap.height)
        except Exception as exc:
            log_agent_warning(self.name, f"source-PDF asset rendering failed; using MinerU image copy: {exc}")
            return None

    def _resolve_mineru_asset_path(self, extraction: MinerUExtraction, item: Dict[str, Any]) -> Path | None:
        raw_path = item.get("img_path") or item.get("image_path") or item.get("path")
        if not raw_path:
            return None
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            return candidate
        candidates = [extraction.extract_dir / candidate]
        if extraction.content_list_path is not None:
            candidates.append(extraction.content_list_path.parent / candidate)
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _mineru_caption(self, item: Dict[str, Any], keys: Tuple[str, ...], fallback: str) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, list):
                text = " ".join(str(part).strip() for part in value if str(part).strip())
            else:
                text = str(value or "").strip()
            if text:
                return text
        return fallback

    def _mineru_caption_entry(self, item: Dict[str, Any], caption: str) -> Dict[str, Any]:
        return {
            "block_id": str(item.get("id") or item.get("block_id") or ""),
            "block_type": str(item.get("type") or ""),
            "captions": [caption] if caption else [],
            "page": item.get("page_idx"),
            "bbox": item.get("bbox"),
        }

    def _build_visual_registry(self, figures: Dict, tables: Dict) -> Dict[str, Dict[str, Any]]:
        """Build the canonical visual asset registry for downstream agents."""
        visual_assets: Dict[str, Dict[str, Any]] = {}

        for fig_id, fig_data in figures.items():
            asset_id = f"figure_{fig_id}"
            visual_assets[asset_id] = {
                "asset_id": asset_id,
                "asset_type": "figure",
                "source_path": fig_data.get("path"),
                "resolved_path": None,
                "caption": fig_data.get("caption", ""),
                "aspect": fig_data.get("aspect", 1.0),
                "provenance": "paper_extracted",
            }

        for table_id, table_data in tables.items():
            asset_id = f"table_{table_id}"
            visual_assets[asset_id] = {
                "asset_id": asset_id,
                "asset_type": "table",
                "source_path": table_data.get("path"),
                "resolved_path": None,
                "caption": table_data.get("caption", ""),
                "aspect": table_data.get("aspect", 1.0),
                "provenance": "paper_extracted",
            }

        return visual_assets

    def _extract_captions(self, document):
        caption_map = {}
        
        for page in document.pages:
            for block_id in page.structure:
                block = page.get_block(block_id)
                
                if block.block_type in [BlockTypes.FigureGroup, BlockTypes.TableGroup, BlockTypes.PictureGroup]:
                    child_blocks = block.structure_blocks(page)
                    figure_or_table = None
                    captions = []
                    
                    for child in child_blocks:
                        child_block = page.get_block(child)
                        if child_block.block_type in [BlockTypes.Figure, BlockTypes.Table, BlockTypes.Picture]:
                            figure_or_table = child_block
                        elif child_block.block_type in [BlockTypes.Caption, BlockTypes.Footnote]:
                            captions.append(child_block.raw_text(document))
                    
                    if figure_or_table:
                        image_filename = f"{figure_or_table.id.to_path()}.jpeg"
                        caption_map[image_filename] = {
                            'block_id': str(figure_or_table.id),
                            'block_type': str(figure_or_table.block_type),
                            'captions': captions,
                            'page': page.page_id
                        }
                
                elif block.block_type in [BlockTypes.Figure, BlockTypes.Table, BlockTypes.Picture]:
                    image_filename = f"{block.id.to_path()}.jpeg"
                    if image_filename not in caption_map:
                        nearby_captions = self._find_nearby_captions(page, block, document)
                        caption_map[image_filename] = {
                            'block_id': str(block.id),
                            'block_type': str(block.block_type),
                            'captions': nearby_captions,
                            'page': page.page_id
                        }
        
        return caption_map

    def _find_nearby_captions(self, page, target_block, document):
        captions = []
        
        # Check all blocks on the page for captions
        for block_id in page.structure:
            block = page.get_block(block_id)
            if block.block_type in [BlockTypes.Caption, BlockTypes.Text]:
                caption_text = block.raw_text(document)
                # Look for figure/table keywords and check if it's nearby
                if any(keyword in caption_text for keyword in ['Figure', 'Table', 'Fig.']):
                    captions.append(caption_text)
        
        # If no captions found, try previous/next blocks
        if not captions:
            for block in [page.get_prev_block(target_block), page.get_next_block(target_block)]:
                if block and block.block_type in [BlockTypes.Caption, BlockTypes.Text]:
                    caption_text = block.raw_text(document)
                    if any(keyword in caption_text for keyword in ['Figure', 'Table', 'Fig.']):
                        captions.append(caption_text)
        
        return captions

    def _cleanup_unused_assets(self, output_dir: Path, name: str, images: Dict, tables: Dict):
        valid_paths = set()
        for img_data in images.values():
            valid_paths.add(Path(img_data['path']).name)
        for table_data in tables.values():
            valid_paths.add(Path(table_data['path']).name)
        
        for png_file in output_dir.glob(f"{name}-*.png"):
            if png_file.name not in valid_paths:
                png_file.unlink()

    def _extract_title_authors(self, text: str, config, state) -> Tuple[str, str]:
        log_agent_info(self.name, "extracting title and authors with llm")
        agent = LangGraphAgent("expert academic paper parser", config, state, "parser")
        
        for attempt in range(3):
            try:
                prompt = Template(self.title_authors_prompt).render(markdown_document=text)
                agent.reset()
                response = agent.step(prompt)
                
                result = extract_json(response.content)

                if "title" in result and "authors" in result:
                    title = normalize_title_for_poster(result["title"].strip())
                    authors = normalize_text_for_poster(result["authors"].strip())
                    
                    # validate format
                    if title and authors:
                        return title, authors
                        
            except Exception as e:
                log_agent_warning(self.name, f"title/authors extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    return "Untitled", "Authors not found"
        
        return "Untitled", "Authors not found"
    
    
    def _classify_visual_assets(self, figures: Dict, tables: Dict, raw_text: str, config, state) -> Tuple[Dict, int, int]:
        # combine all visuals for classification
        all_visuals = []
        for fig_id, fig_data in figures.items():
            all_visuals.append({
                "id": f"figure_{fig_id}",
                "type": "figure", 
                "caption": fig_data.get("caption", ""),
                "aspect_ratio": fig_data.get("aspect", 1.0)
            })
        
        for tab_id, tab_data in tables.items():
            all_visuals.append({
                "id": f"table_{tab_id}",
                "type": "table",
                "caption": tab_data.get("caption", ""),
                "aspect_ratio": tab_data.get("aspect", 1.0)
            })
        
        if not all_visuals:
            return {"key_visual": None, "problem_illustration": [], "method_workflow": [], "main_results": [], "comparative_results": [], "supporting": []}, 0, 0
            
        log_agent_info(self.name, f"classifying {len(all_visuals)} visual assets")
        agent = LangGraphAgent("expert poster designer", config, state, "parser")
        
        for attempt in range(3):
            try:
                prompt = Template(self.visual_classification_prompt).render(
                    visuals_list=json.dumps(all_visuals, indent=2)
                )
                
                agent.reset()
                response = agent.step(prompt)
                classification = extract_json(response.content)
                
                # validate classification
                required_keys = ["key_visual", "problem_illustration", "method_workflow", "main_results", "comparative_results", "supporting"]
                if all(key in classification for key in required_keys):
                    return classification, response.input_tokens, response.output_tokens
                    
            except Exception as e:
                log_agent_warning(self.name, f"visual classification attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    # fallback classification
                    return self._fallback_visual_classification(all_visuals), 0, 0
        
        return self._fallback_visual_classification(all_visuals), 0, 0
    
    def _fallback_visual_classification(self, visuals):
        # simple rule-based fallback
        classification = {
            "key_visual": None,
            "problem_illustration": [],
            "method_workflow": [],
            "main_results": [],
            "comparative_results": [],
            "supporting": [],
        }
        
        for visual in visuals:
            caption = visual.get("caption", "").lower()
            if "result" in caption or "performance" in caption or "comparison" in caption:
                classification["main_results"].append(visual["id"])
            elif "method" in caption or "architecture" in caption or "framework" in caption:
                classification["method_workflow"].append(visual["id"])
            elif "problem" in caption or "challenge" in caption or "motivation" in caption:
                classification["problem_illustration"].append(visual["id"])
            else:
                classification["supporting"].append(visual["id"])
        
        # select key visual from main results or method visuals
        if classification["main_results"]:
            classification["key_visual"] = classification["main_results"][0]
        elif classification["method_workflow"]:
            classification["key_visual"] = classification["method_workflow"][0]
        elif classification["supporting"]:
            classification["key_visual"] = classification["supporting"][0]
        
        return classification

    def _extract_structured_sections(self, raw_text: str, config, state) -> Dict:
        log_agent_info(self.name, "extracting structured sections from paper")
        agent = LangGraphAgent("expert paper section extractor", config, state, "parser")
        
        for attempt in range(3):
            try:
                prompt = Template(self.section_extraction_prompt).render(raw_text=raw_text)
                agent.reset()
                response = agent.step(prompt)
                
                structured_sections = extract_json(response.content)
                
                if self._validate_structured_sections(structured_sections):
                    log_agent_success(self.name, f"extracted {len(structured_sections.get('paper_sections', []))} structured sections")
                    return structured_sections
                else:
                    log_agent_warning(self.name, f"attempt {attempt + 1}: invalid structured sections")
                    
            except Exception as e:
                log_agent_warning(self.name, f"section extraction attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise ValueError("failed to extract structured sections after multiple attempts")

        raise ValueError("failed to extract valid structured sections after multiple attempts")
    
    def _validate_structured_sections(self, structured_sections: Dict) -> bool:
        """validate structured sections format"""
        if "paper_sections" not in structured_sections:
            log_agent_warning(self.name, "validation error: missing 'paper_sections'")
            return False
        
        sections = structured_sections["paper_sections"]
        if not isinstance(sections, list) or len(sections) < 3:
            log_agent_warning(self.name, f"validation error: need at least 3 sections, got {len(sections)}")
            return False
        
        # validate each section
        for i, section in enumerate(sections):
            required_fields = ["section_name", "section_type", "content"]
            for field in required_fields:
                if field not in section:
                    log_agent_warning(self.name, f"validation error: section {i} missing '{field}'")
                    return False
        
        return True


def parser_node(state: PosterState) -> PosterState:
    return Parser()(state) 
