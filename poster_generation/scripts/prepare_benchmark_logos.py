#!/usr/bin/env python3
"""Prepare verified local affiliation logos before benchmark poster generation.

The script extracts the primary institution from each PDF header in small LLM
batches, resolves a vector or sufficiently large raster logo, and writes the
accepted image to ``<paper_dir>/affiliation_logo.png``. It never uses favicons
and never enlarges a low-resolution raster image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests
from PIL import Image, ImageChops

from src.state.poster_state import ModelConfig
from utils.langgraph_utils import LangGraphAgent, extract_json


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUBSETS = ("aaai2026", "cvpr2025", "neurips2025", "p2peval", "pairs")
DEFAULT_SUBSETS = ("aaai2026", "cvpr2025")
INVENTORY_PATH = REPO_ROOT / "Benchmark" / "affiliation_logo_inventory.json"
CSV_PATH = REPO_ROOT / "Benchmark" / "affiliation_logo_inventory.csv"
GLOBAL_LOGO_DIR = REPO_ROOT / "assets" / "institution_logos"
GLOBAL_MANIFEST_PATH = GLOBAL_LOGO_DIR / "manifest.json"
MIN_RASTER_LONG_EDGE = 300
MIN_RASTER_SHORT_EDGE = 72
MAX_NORMALIZED_LONG_EDGE = 2400
GENERIC_EMAIL_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "outlook.com",
    "hotmail.com",
    "qq.com",
    "163.com",
    "126.com",
    "foxmail.com",
    "icloud.com",
    "proton.me",
}
INSTITUTION_ALIASES = {
    "bdsi anu": "Australian National University",
    "hkust": "Hong Kong University of Science and Technology",
    "iis c bangalore": "Indian Institute of Science",
    "iisc bangalore": "Indian Institute of Science",
    "institute for ai industry research air tsinghua university": "Tsinghua University",
    "institute for ai industry research tsinghua university": "Tsinghua University",
    "institute of automation chinese academy of sciences": "Chinese Academy of Sciences",
    "institute of computing technology chinese academy of sciences": "Chinese Academy of Sciences",
    "institute of information engineering chinese academy of sciences": "Chinese Academy of Sciences",
    "institute of software chinese academy of sciences": "Chinese Academy of Sciences",
    "hangzhou institute of medicine chinese academy of sciences": "Chinese Academy of Sciences",
    "kaist": "Korea Advanced Institute of Science and Technology",
    "kaist ai": "Korea Advanced Institute of Science and Technology",
    "postech": "Pohang University of Science and Technology",
    "qatar computing research institute hbku": "Hamad Bin Khalifa University",
    "state key laboratory of virtual reality technology and systems": "Beihang University",
    "the key laboratory of brain machine intelligence technology ministry of education": "Nanjing University of Aeronautics and Astronautics",
    "key laboratory of brain machine intelligence technology ministry of education": "Nanjing University of Aeronautics and Astronautics",
    "technical university munich tum": "Technical University of Munich",
    "technical university munich": "Technical University of Munich",
    "tubingen ai center": "University of Tubingen",
    "university federico ii of naples": "University of Naples Federico II",
    "university of queensland": "University of Queensland",
    "xjtu": "Xi'an Jiaotong University",
}
CURATED_LOGO_URLS = {
    "ben gurion university of negev": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Ben-Gurion_University_of_the_Negev_logo.svg",
    ),
    "brookhaven national laboratory": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Brookhaven_National_Laboratory_logo_2021.svg",
    ),
    "bytedance intelligent creation": (
        "curated_wikimedia",
        "https://upload.wikimedia.org/wikipedia/commons/0/07/ByteDance_logo_English.svg",
    ),
    "great bay university": (
        "curated_official",
        "https://www.gbu.edu.cn/Uploads/Picture/2026/05/20/s6a0d4d67ef57a_compress.png",
    ),
    "hamad bin khalifa university": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/HBKU_Logo,_2016.jpg",
    ),
    "hong kong university of science and technology": (
        "curated_official",
        "https://hkust.edu.hk/sites/default/files/2024-03/HKUST_logo_1.svg",
    ),
    "indian institute of science": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/IISc_logo(2).svg",
    ),
    "imperial college london": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Imperial_logo.svg",
    ),
    "macao polytechnic university": (
        "curated_wikimedia",
        "https://upload.wikimedia.org/wikipedia/commons/e/ee/Macao_Polytechnic_University_logo.svg",
    ),
    "meshcapade": ("curated_official", "https://meshcapade.com/images/logo.svg"),
    "nara institute of science and technology": (
        "curated_official",
        "https://www.naist.jp/mt-static/support/theme_static/naist_en_main/images/logo-full.png",
    ),
    "naver labs": ("curated_official", "https://www.naverlabs.com/img/svg/logo.svg"),
    "national university of singapore": (
        "curated_wikipedia",
        "https://upload.wikimedia.org/wikipedia/en/9/9b/NationalUniversityofSingapore.svg",
    ),
    "reichman university": (
        "curated_wikipedia",
        "https://upload.wikimedia.org/wikipedia/en/a/a8/Reichman_University.svg",
    ),
    "purdue university": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/Purdue_University_system_logo.svg",
    ),
    "seoul national university": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/Seoul_national_university_emblem.svg",
    ),
    "shanghai artificial intelligence laboratory": (
        "curated_official",
        "https://www.shlab.org.cn/static/asset/img/share-logo.png",
    ),
    "shanghai innovation institute": (
        "curated_official",
        "https://www.sii.edu.cn/_upload/tpl/00/09/9/template9/images/logo.svg",
    ),
    "shanghaitech university": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/ShanghaiTech_University_Wordmark.svg",
    ),
    "south china normal university": (
        "curated_urongda",
        "https://cdn.urongda.com/images/normal/medium/south-china-normal-university-logo-1024px.png",
    ),
    "stanford university": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/Stanford_wordmark_(2012).svg",
    ),
    "stony brook university": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/Stony_Brook_U_logo_horizontal.svg",
    ),
    "chinese university of hong kong": (
        "curated_official",
        "https://www.cuhk.edu.hk/english/images/cuhk_logo_2x.png?20221027",
    ),
    "toyota technological institute japan": (
        "curated_official",
        "https://www.toyota-ti.ac.jp/img/logo.svg",
    ),
    "university of chinese academy of sciences": (
        "curated_urongda",
        "https://cdn.urongda.com/images/normal/medium/university-of-chinese-academy-of-sciences-logo-1024px.png",
    ),
    "university of florida": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_Florida_logo.svg",
    ),
    "university of illinois urbana champaign": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_Illinois_at_Urbana-Champaign_Wordmark.svg",
    ),
    "university of iowa": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_Iowa_seal.svg",
    ),
    "university of macau": (
        "curated_wikimedia",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/University_of_Macau_logo.svg",
    ),
    "university of oulu": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_Oulu_logo.jpg",
    ),
    "university of queensland": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/UQlogo.svg",
    ),
    "university of southampton": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_Southampton_logo.svg",
    ),
    "university of surrey": (
        "curated_wikimedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/Uni_of_Surrey_master_logo.png",
    ),
    "university of tubingen": (
        "curated_wikipedia",
        "https://en.wikipedia.org/wiki/Special:Redirect/file/University_of_T%C3%BCbingen_logo.svg",
    ),
    "university of washington": (
        "curated_wikipedia",
        "https://upload.wikimedia.org/wikipedia/en/5/58/University_of_Washington_seal.svg",
    ),
    "yonsei university": (
        "curated_wikipedia",
        "https://upload.wikimedia.org/wikipedia/en/6/6e/Yonsei_university_logo_en.svg",
    ),
}

CURATED_LOGO_URLS.update(
    {
        name: (
            "curated_urongda",
            f"https://cdn.urongda.com/images/normal/medium/{name.replace(' ', '-')}-logo-1024px.png",
        )
        for name in (
            "central south university",
            "china university of geosciences",
            "dalian university of technology",
            "east china normal university",
            "harbin institute of technology",
            "hohai university",
            "nanjing university",
            "national university of defense technology",
            "shandong university",
            "southern university of science and technology",
            "wuhan university of technology",
        )
    }
)

CURATED_SVG_COLOR_OVERRIDES = {
    "naver labs": ("white", "#03c75a"),
    "toyota technological institute japan": ("#fff", "#24295f"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_name(value: str) -> str:
    value = html.unescape(str(value or ""))
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(token for token in value.split() if token not in {"the"})


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_name(value)).strip("-") or "institution"


def canonical_institution(value: str) -> str:
    normalized = normalize_name(value)
    return INSTITUTION_ALIASES.get(normalized, re.sub(r"\s+", " ", str(value or "")).strip())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def first_page_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=45,
        check=False,
    )
    text = result.stdout.replace("\x00", "")
    abstract_match = re.search(r"(?im)^\s*(?:abstract|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t)\s*$", text)
    if abstract_match:
        text = text[: abstract_match.start()]
    return text[:6500].strip()


def email_domains(text: str) -> list[str]:
    domains = []
    seen = set()
    for domain in re.findall(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text):
        normalized = domain.lower().strip(".,;:()[]{}")
        if normalized not in GENERIC_EMAIL_DOMAINS and normalized not in seen:
            domains.append(normalized)
            seen.add(normalized)
    return domains


def discover_papers(benchmark_root: Path, subsets: list[str]) -> list[Path]:
    papers: list[Path] = []
    for subset in subsets:
        papers.extend(sorted((benchmark_root / subset).glob("*/paper.pdf"), key=lambda item: item.parent.name.lower()))
    return [path.resolve() for path in papers]


def load_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "papers": {},
            "extraction_usage": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_inventory(path: Path, inventory: dict[str, Any], benchmark_root: Path) -> None:
    inventory["updated_at"] = now_iso()
    papers = inventory.setdefault("papers", {})
    status_counts: dict[str, int] = {}
    for record in papers.values():
        status = str(record.get("logo_status") or "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    inventory["summary"] = {
        "paper_count": len(papers),
        "institution_extracted": sum(bool(record.get("institution")) for record in papers.values()),
        "local_logo_ready": sum(record.get("logo_status") == "ready" for record in papers.values()),
        "not_applicable": sum(record.get("logo_status") == "not_applicable" for record in papers.values()),
        "preflight_complete": sum(
            record.get("logo_status") in {"ready", "not_applicable"} for record in papers.values()
        ),
        "status_counts": status_counts,
    }
    atomic_json(path, inventory)

    rows = []
    for relative_path, record in sorted(papers.items()):
        rows.append(
            {
                "paper": relative_path,
                "title": record.get("title", ""),
                "institution": record.get("institution", ""),
                "logo_institution": record.get("logo_institution", ""),
                "domain": record.get("domain", ""),
                "confidence": record.get("confidence", ""),
                "logo_status": record.get("logo_status", "pending"),
                "logo_source_kind": record.get("logo_source_kind", ""),
                "logo_source_url": record.get("logo_source_url", ""),
                "local_logo_path": record.get("local_logo_path", ""),
                "native_width": (record.get("native_size") or ["", ""])[0],
                "native_height": (record.get("native_size") or ["", ""])[1],
                "warning": record.get("warning", ""),
            }
        )
    csv_path = path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["paper"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def initialize_records(inventory: dict[str, Any], papers: list[Path], benchmark_root: Path) -> None:
    records = inventory.setdefault("papers", {})
    for pdf_path in papers:
        relative = str(pdf_path.relative_to(benchmark_root))
        record = records.setdefault(relative, {})
        record.setdefault("title", pdf_path.parent.name)
        record.setdefault("paper_path", str(pdf_path))
        local_logo = pdf_path.parent / "affiliation_logo.png"
        if local_logo.exists() and local_logo.stat().st_size > 0:
            size = image_size(local_logo)
            record.update(
                {
                    "logo_status": "ready",
                    "logo_source_kind": record.get("logo_source_kind") or "existing_local",
                    "local_logo_path": str(local_logo.resolve()),
                    "native_size": list(size) if size else record.get("native_size", []),
                }
            )


def extract_institutions(
    inventory: dict[str, Any],
    benchmark_root: Path,
    model_name: str,
    batch_size: int,
    inventory_path: Path,
) -> None:
    pending: list[tuple[str, dict[str, Any], str, list[str]]] = []
    for relative, record in inventory["papers"].items():
        if record.get("institution"):
            continue
        pdf_path = benchmark_root / relative
        header = first_page_text(pdf_path)
        record["header_sha1"] = hashlib.sha1(header.encode("utf-8", errors="ignore")).hexdigest()
        record["email_domains"] = email_domains(header)
        pending.append((relative, record, header, record["email_domains"]))

    if not pending:
        return

    config = ModelConfig(model_name=model_name, provider="openai", temperature=1.0, max_tokens=4096)
    agent = LangGraphAgent(
        "You extract exact author affiliations from academic paper headers. Use only the supplied text.",
        config,
        None,
        "logo_inventory",
    )
    usage = inventory.setdefault("extraction_usage", {"input_tokens": 0, "output_tokens": 0, "calls": 0})

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        entries = []
        for index, (relative, record, header, domains) in enumerate(batch, start=1):
            entries.append(
                f"\n=== PAPER {index} ===\n"
                f"id: {relative}\n"
                f"directory_title: {record['title']}\n"
                f"observed_email_domains: {json.dumps(domains)}\n"
                f"header:\n{header}\n"
            )
        prompt = """Extract the primary institution for every paper below.

Rules:
1. Primary institution means the affiliation number attached to the FIRST listed author. If numbering is unclear, use the first full institution listed beneath the authors.
2. Return the canonical parent university/company/research institute name, not a department, lab, city, country, or email address.
3. Preserve the official English name visible in the header. Expand obvious wrapped lines, but do not invent an institution.
4. domain must be an observed email domain associated with that institution when possible; otherwise use an empty string. Do not guess domains.
5. confidence is high, medium, or low. Set low when the first-author mapping is genuinely ambiguous.
6. Return every id exactly once.

Return JSON only:
{"items":[{"id":"...","institution":"...","domain":"...","confidence":"high","evidence":"short exact header fragment"}]}
""" + "".join(entries)

        response = agent.step(prompt)
        parsed = extract_json(response.content)
        items = parsed.get("items") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"institution extraction batch {offset // batch_size + 1} returned invalid JSON")
        by_id = {str(item.get("id")): item for item in items if isinstance(item, dict) and item.get("id")}
        missing = [relative for relative, *_ in batch if relative not in by_id]
        if missing:
            raise RuntimeError(f"institution extraction omitted {len(missing)} papers: {missing[:3]}")

        for relative, record, _header, observed_domains in batch:
            item = by_id[relative]
            institution = re.sub(r"\s+", " ", str(item.get("institution") or "")).strip(" ,.;")
            domain = str(item.get("domain") or "").lower().strip(" ,.;")
            if domain and domain not in observed_domains:
                domain = ""
            record.update(
                {
                    "institution": institution,
                    "domain": domain,
                    "confidence": str(item.get("confidence") or "low").lower(),
                    "institution_evidence": str(item.get("evidence") or "")[:500],
                    "institution_extraction": f"llm_header:{model_name}",
                }
            )
            if not institution:
                record["warning"] = "primary institution could not be extracted"
        usage["input_tokens"] = int(usage.get("input_tokens", 0)) + int(response.input_tokens or 0)
        usage["output_tokens"] = int(usage.get("output_tokens", 0)) + int(response.output_tokens or 0)
        usage["calls"] = int(usage.get("calls", 0)) + 1
        save_inventory(inventory_path, inventory, benchmark_root)
        print(
            f"[{now_iso()}] extracted institutions {min(offset + len(batch), len(pending))}/{len(pending)} "
            f"input={response.input_tokens} output={response.output_tokens}",
            flush=True,
        )


@dataclass
class LogoCandidate:
    source_kind: str
    source_url: str
    website_url: str = ""


class LogoResolver:
    def __init__(self, cache_dir: Path, request_delay: float = 0.25):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PosterMELD/1.0 (https://github.com/Jackey0903/PosterMELD)",
                "Accept-Language": "en-US,en;q=0.8",
            }
        )
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not GLOBAL_MANIFEST_PATH.exists():
            return {}
        try:
            data = json.loads(GLOBAL_MANIFEST_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _request(self, url: str, **kwargs: Any) -> requests.Response | None:
        for attempt in range(3):
            time.sleep(self.request_delay)
            try:
                response = self.session.get(url, timeout=25, allow_redirects=True, **kwargs)
            except requests.exceptions.SSLError:
                try:
                    response = self.session.get(url, timeout=25, allow_redirects=True, verify=False, **kwargs)
                except requests.RequestException:
                    response = None
            except requests.RequestException:
                response = None
            if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt < 2:
                retry_after = 0.0
                if response is not None:
                    try:
                        retry_after = min(float(response.headers.get("retry-after", 0) or 0), 8.0)
                    except ValueError:
                        retry_after = 0.0
                time.sleep(max(retry_after, 1.0 + attempt * 1.5))
        return response

    def existing_cache(self, institution: str) -> tuple[Path, dict[str, Any]] | None:
        target_key = normalize_name(institution)
        for filename, metadata in self.manifest.items():
            if normalize_name(metadata.get("institution", "")) != target_key:
                continue
            path = self.cache_dir / filename
            if path.exists() and validate_raster(path):
                return path, metadata
        slug = slugify(institution)
        for path in sorted(self.cache_dir.glob(f"{slug}.*")):
            if path.suffix.lower() == ".png" and validate_raster(path):
                return path, {"institution": institution, "source_url": "", "source_kind": "global_cache"}
        compact_target = normalize_name(institution).replace(" ", "")
        for path in sorted(self.cache_dir.glob("*.png")):
            if normalize_name(path.stem).replace(" ", "") == compact_target and validate_raster(path):
                return path, {"institution": institution, "source_url": "", "source_kind": "global_cache"}
        return None

    def resolve(self, institution: str, domain: str = "") -> tuple[Path, dict[str, Any]] | None:
        curated = CURATED_LOGO_URLS.get(normalize_name(institution))
        if curated:
            source_kind, source_url = curated
            accepted = self._download_candidate(institution, LogoCandidate(source_kind, source_url))
            if accepted:
                return accepted

        cached = self.existing_cache(institution)
        if cached:
            path, metadata = cached
            return path, {
                "source_kind": metadata.get("source_kind") or "global_cache",
                "source_url": metadata.get("source_url", ""),
                "native_size": list(image_size(path) or (0, 0)),
            }

        candidates, website = self._wikidata_candidates(institution)
        for candidate in candidates:
            accepted = self._download_candidate(institution, candidate)
            if accepted:
                return accepted

        for candidate in self._wikipedia_candidates(institution):
            accepted = self._download_candidate(institution, candidate)
            if accepted:
                return accepted

        official_sites = []
        if website:
            official_sites.append(website)
        if domain:
            official_sites.extend([f"https://www.{domain}/", f"https://{domain}/"])
        for site in dict.fromkeys(official_sites):
            for candidate in self._official_site_candidates(site):
                accepted = self._download_candidate(institution, candidate)
                if accepted:
                    return accepted

        urongda = LogoCandidate(
            "urongda_1024",
            f"https://cdn.urongda.com/images/normal/medium/{slugify(institution)}-logo-1024px.png",
        )
        accepted = self._download_candidate(institution, urongda)
        if accepted:
            return accepted
        return None

    def _wikidata_candidates(self, institution: str) -> tuple[list[LogoCandidate], str]:
        response = self._request(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": institution,
                "language": "en",
                "format": "json",
                "limit": 6,
                "type": "item",
            },
        )
        if response is None or response.status_code != 200:
            return [], ""
        try:
            results = response.json().get("search", [])
        except Exception:
            return [], ""
        target = normalize_name(institution)
        ranked = []
        for item in results:
            label = normalize_name(item.get("label", ""))
            aliases = [normalize_name(alias) for alias in item.get("aliases", [])]
            scores = [SequenceMatcher(None, target, label).ratio()]
            scores.extend(SequenceMatcher(None, target, alias).ratio() for alias in aliases)
            ranked.append((max(scores or [0]), item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked or ranked[0][0] < 0.64:
            return [], ""
        entity_id = ranked[0][1].get("id")
        entity_response = self._request(f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json")
        if entity_response is None or entity_response.status_code != 200:
            return [], ""
        try:
            claims = entity_response.json()["entities"][entity_id].get("claims", {})
        except Exception:
            return [], ""

        website = ""
        for claim in claims.get("P856", []):
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(value, str) and value.startswith("http"):
                website = value
                break
        filenames: list[str] = []
        for property_id in ("P154", "P94"):
            for claim in claims.get(property_id, []):
                value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(value, str) and value not in filenames:
                    filenames.append(value)
        candidates = []
        for filename in filenames:
            info = self._mediawiki_image_info("https://commons.wikimedia.org/w/api.php", filename)
            if info:
                candidates.append(LogoCandidate("wikidata_commons", info, website))
        return candidates, website

    def _wikipedia_candidates(self, institution: str) -> list[LogoCandidate]:
        api_url = "https://en.wikipedia.org/w/api.php"
        response = self._request(
            api_url,
            params={"action": "query", "list": "search", "srsearch": institution, "srlimit": 3, "format": "json"},
        )
        if response is None or response.status_code != 200:
            return []
        try:
            pages = response.json().get("query", {}).get("search", [])
        except Exception:
            return []
        target = normalize_name(institution)
        ranked_pages = sorted(
            pages,
            key=lambda item: SequenceMatcher(None, target, normalize_name(item.get("title", ""))).ratio(),
            reverse=True,
        )
        candidates: list[LogoCandidate] = []
        for page in ranked_pages[:2]:
            if SequenceMatcher(None, target, normalize_name(page.get("title", ""))).ratio() < 0.58:
                continue
            parsed = self._request(
                api_url,
                params={"action": "parse", "pageid": page.get("pageid"), "prop": "images", "format": "json"},
            )
            if parsed is None or parsed.status_code != 200:
                continue
            try:
                images = parsed.json().get("parse", {}).get("images", [])
            except Exception:
                continue
            scored = []
            for filename in images:
                lowered = filename.lower()
                if any(
                    token in lowered
                    for token in (
                        "commons-logo",
                        "wikisource-logo",
                        "wikidata-logo",
                        "wiktionary-logo",
                        "wikipedia-logo",
                    )
                ):
                    continue
                if not any(token in lowered for token in ("logo", "seal", "crest", "emblem", "wordmark")):
                    continue
                score = sum(token in lowered for token in ("logo", "seal", "crest", "emblem", "wordmark"))
                score += sum(token in normalize_name(filename) for token in target.split() if len(token) > 3)
                scored.append((score, filename))
            for _, filename in sorted(scored, reverse=True)[:4]:
                url = self._mediawiki_image_info(api_url, filename)
                if url:
                    candidates.append(LogoCandidate("wikipedia_logo", url))

            page_image = self._request(
                api_url,
                params={
                    "action": "query",
                    "pageids": page.get("pageid"),
                    "prop": "pageimages",
                    "piprop": "original|thumbnail|name",
                    "pithumbsize": 1800,
                    "format": "json",
                },
            )
            if page_image is not None and page_image.status_code == 200:
                try:
                    page_info = next(iter(page_image.json().get("query", {}).get("pages", {}).values()))
                    filename = str(page_info.get("pageimage") or "")
                    source = str(
                        (page_info.get("original") or {}).get("source")
                        or (page_info.get("thumbnail") or {}).get("source")
                        or ""
                    )
                except Exception:
                    filename, source = "", ""
                if source and any(
                    token in filename.lower()
                    for token in ("logo", "seal", "crest", "emblem", "arms", "shield", "wordmark", "signature")
                ):
                    candidates.append(LogoCandidate("wikipedia_pageimage", source))
        return candidates

    def _mediawiki_image_info(self, api_url: str, filename: str) -> str:
        response = self._request(
            api_url,
            params={
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": 1800,
                "titles": f"File:{filename}",
            },
        )
        if response is None or response.status_code != 200:
            return ""
        try:
            page = next(iter(response.json().get("query", {}).get("pages", {}).values()))
            info = (page.get("imageinfo") or [{}])[0]
        except Exception:
            return ""
        mime = str(info.get("mime") or "")
        if mime == "image/svg+xml":
            return str(info.get("url") or "")
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if max(width, height) < MIN_RASTER_LONG_EDGE or min(width, height) < MIN_RASTER_SHORT_EDGE:
            return ""
        return str(info.get("thumburl") or info.get("url") or "")

    def _official_site_candidates(self, website: str) -> list[LogoCandidate]:
        response = self._request(website)
        if response is None or response.status_code != 200 or not response.text:
            return []
        tags = re.findall(r"<img\b[^>]*>", response.text, flags=re.IGNORECASE)
        scored: list[tuple[int, str]] = []
        for tag in tags:
            src_match = re.search(r"(?:src|data-src)\s*=\s*[\"']([^\"']+)[\"']", tag, flags=re.IGNORECASE)
            if not src_match:
                continue
            src = html.unescape(src_match.group(1).strip())
            descriptor = tag.lower()
            score = 0
            for token, weight in (("logo", 12), ("brand", 7), ("crest", 8), ("emblem", 8), ("university", 3)):
                if token in descriptor:
                    score += weight
            if any(token in descriptor for token in ("footer", "partner", "sponsor", "favicon", "icon-")):
                score -= 8
            if src.lower().endswith(".svg"):
                score += 5
            if score >= 7:
                scored.append((score, urljoin(str(response.url), src)))
        return [LogoCandidate("official_site_header", url, website) for _, url in sorted(scored, reverse=True)[:6]]

    def _download_candidate(self, institution: str, candidate: LogoCandidate) -> tuple[Path, dict[str, Any]] | None:
        if not candidate.source_url:
            return None
        response = self._request(candidate.source_url)
        if response is None or response.status_code != 200 or len(response.content) < 500:
            return None
        content_type = str(response.headers.get("content-type") or "").lower()
        suffix = Path(urlparse(str(response.url)).path).suffix.lower()
        is_svg = "svg" in content_type or suffix == ".svg" or response.content.lstrip().startswith(b"<svg")
        slug = slugify(institution)
        with tempfile.TemporaryDirectory(prefix="paper2poster-logo-") as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / ("source.svg" if is_svg else f"source{suffix or '.img'}")
            source_path.write_bytes(response.content)
            output_path = temp_root / "logo.png"
            if is_svg:
                color_override = CURATED_SVG_COLOR_OVERRIDES.get(normalize_name(institution))
                if color_override:
                    source_color, target_color = color_override
                    svg_text = source_path.read_text(encoding="utf-8", errors="replace")
                    source_path.write_text(
                        re.sub(re.escape(source_color), target_color, svg_text, flags=re.IGNORECASE),
                        encoding="utf-8",
                    )
                command = shutil.which("rsvg-convert")
                if not command:
                    return None
                converted = subprocess.run(
                    [command, "-w", "1800", str(source_path), "-o", str(output_path)],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if converted.returncode != 0:
                    return None
            else:
                if not normalize_raster_without_upscale(source_path, output_path):
                    return None
            if not trim_raster_canvas(output_path):
                return None
            if not validate_raster(output_path):
                return None
            target = self.cache_dir / f"{slug}.png"
            shutil.copy2(output_path, target)
            native = image_size(target) or (0, 0)
            return target, {
                "source_kind": candidate.source_kind,
                "source_url": candidate.source_url,
                "website_url": candidate.website_url,
                "native_size": list(native),
            }


def normalize_raster_without_upscale(source: Path, destination: Path) -> bool:
    try:
        with Image.open(source) as image:
            image.load()
            image = image.convert("RGBA")
            native_width, native_height = image.size
            if max(native_width, native_height) < MIN_RASTER_LONG_EDGE:
                return False
            if min(native_width, native_height) < MIN_RASTER_SHORT_EDGE:
                return False
            if max(image.size) > MAX_NORMALIZED_LONG_EDGE:
                image.thumbnail((MAX_NORMALIZED_LONG_EDGE, MAX_NORMALIZED_LONG_EDGE), Image.Resampling.LANCZOS)
            image.save(destination, "PNG")
        return True
    except Exception:
        return False


def trim_raster_canvas(path: Path) -> bool:
    """Remove transparent or white outer padding without enlarging the logo."""
    try:
        with Image.open(path) as image:
            image.load()
            rgba = image.convert("RGBA")
            composite = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite.alpha_composite(rgba)
            white = Image.new("RGB", rgba.size, (255, 255, 255))
            mask = ImageChops.difference(composite.convert("RGB"), white).convert("L").point(
                lambda value: 255 if value > 10 else 0
            )
            bbox = mask.getbbox()
            if bbox is None:
                return False
            width, height = rgba.size
            padding = max(2, round(max(width, height) * 0.01))
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            bottom = min(height, bbox[3] + padding)
            if (left, top, right, bottom) != (0, 0, width, height):
                rgba.crop((left, top, right, bottom)).save(path, "PNG")
        return True
    except Exception:
        return False


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def validate_raster(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if max(width, height) < MIN_RASTER_LONG_EDGE or min(width, height) < MIN_RASTER_SHORT_EDGE:
                return False
            rgba = image.convert("RGBA")
            composite = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite.alpha_composite(rgba)
            white = Image.new("RGB", rgba.size, (255, 255, 255))
            mask = ImageChops.difference(composite.convert("RGB"), white).convert("L").point(
                lambda value: 255 if value > 10 else 0
            )
            bbox = mask.getbbox()
            if bbox is None:
                return False
            visible_pixels = mask.histogram()[255]
            return visible_pixels / max(width * height, 1) >= 0.001
    except Exception:
        return False


def resolve_logos(
    inventory: dict[str, Any],
    benchmark_root: Path,
    inventory_path: Path,
    request_delay: float,
    workers: int,
) -> None:
    records = inventory["papers"]
    pending_groups: dict[str, dict[str, Any]] = {}
    for relative, record in sorted(records.items()):
        paper_dir = (benchmark_root / relative).parent
        local_path = paper_dir / "affiliation_logo.png"
        institution = str(record.get("institution") or "").strip()
        logo_institution = canonical_institution(institution) if institution else ""
        record["logo_institution"] = logo_institution
        force_curated = normalize_name(logo_institution) in CURATED_LOGO_URLS
        if local_path.exists():
            trim_raster_canvas(local_path)
        if local_path.exists() and validate_raster(local_path) and not force_curated:
            record.update(
                {
                    "logo_status": "ready",
                    "local_logo_path": str(local_path.resolve()),
                    "native_size": list(image_size(local_path) or (0, 0)),
                }
            )
            continue
        if not institution:
            record.update({"logo_status": "unresolved_institution", "warning": "no primary institution"})
            continue
        if normalize_name(logo_institution) == "independent researcher":
            marker = paper_dir / "no_affiliation_logo"
            marker.write_text("No institutional affiliation is listed for the primary author.\n", encoding="utf-8")
            record.update(
                {
                    "logo_status": "not_applicable",
                    "local_logo_path": "",
                    "warning": "primary author is explicitly listed as an independent researcher",
                }
            )
            continue
        cache_key = normalize_name(logo_institution)
        group = pending_groups.setdefault(
            cache_key,
            {
                "institution": logo_institution,
                "domain": str(record.get("domain") or ""),
                "records": [],
            },
        )
        if not group["domain"] and record.get("domain"):
            group["domain"] = str(record["domain"])
        group["records"].append((relative, record, local_path))

    def resolve_group(cache_key: str, group: dict[str, Any]) -> tuple[str, tuple[Path, dict[str, Any]] | None]:
        resolver = LogoResolver(GLOBAL_LOGO_DIR, request_delay=request_delay)
        return cache_key, resolver.resolve(group["institution"], group["domain"])

    total_groups = len(pending_groups)
    completed_groups = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="logo") as executor:
        futures = {
            executor.submit(resolve_group, cache_key, group): cache_key
            for cache_key, group in pending_groups.items()
        }
        for future in as_completed(futures):
            cache_key = futures[future]
            group = pending_groups[cache_key]
            try:
                _, resolved = future.result()
            except Exception as exc:
                resolved = None
                failure_warning = f"logo resolver error: {str(exc)[:300]}"
            else:
                failure_warning = "no vector or sufficiently large raster logo found; favicon disabled"

            for relative, record, local_path in group["records"]:
                if not resolved:
                    record.update({"logo_status": "unresolved_logo", "warning": failure_warning})
                    continue
                cached_path, metadata = resolved
                shutil.copy2(cached_path, local_path)
                record.update(
                    {
                        "logo_status": "ready",
                        "logo_source_kind": metadata.get("source_kind", ""),
                        "logo_source_url": metadata.get("source_url", ""),
                        "logo_website_url": metadata.get("website_url", ""),
                        "local_logo_path": str(local_path.resolve()),
                        "native_size": list(image_size(local_path) or (0, 0)),
                        "warning": "",
                    }
                )
            completed_groups += 1
            save_inventory(inventory_path, inventory, benchmark_root)
            status = "ready" if resolved else "unresolved_logo"
            print(
                f"[{now_iso()}] institution logo {completed_groups}/{total_groups} {status} :: {group['institution']} "
                f"(papers={len(group['records'])})",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local high-resolution affiliation logos for benchmark papers.")
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "Benchmark")
    parser.add_argument("--subsets", nargs="+", choices=SUPPORTED_SUBSETS, default=list(DEFAULT_SUBSETS))
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--model", default=os.getenv("PAPER2POSTER_TEXT_MODEL") or "gpt-5.4")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    inventory_path = args.inventory.resolve()
    papers = discover_papers(benchmark_root, args.subsets)
    if args.max_papers > 0:
        papers = papers[: args.max_papers]
    inventory = load_inventory(inventory_path)
    initialize_records(inventory, papers, benchmark_root)
    save_inventory(inventory_path, inventory, benchmark_root)
    if not args.skip_extraction:
        extract_institutions(inventory, benchmark_root, args.model, args.batch_size, inventory_path)
    if not args.extract_only:
        resolve_logos(inventory, benchmark_root, inventory_path, args.request_delay, args.workers)
    save_inventory(inventory_path, inventory, benchmark_root)
    print(json.dumps(inventory.get("summary", {}), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
