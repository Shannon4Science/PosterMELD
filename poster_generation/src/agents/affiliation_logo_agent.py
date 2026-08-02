"""Affiliation logo discovery and caching."""

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

from src.config.poster_config import load_config
from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_error, log_agent_info, log_agent_success, log_agent_warning


class AffiliationLogoAgent:
    """Resolve paper affiliations into local logo assets.

    The agent deliberately runs before layout construction. Layout elements should
    receive already-local image paths, while the renderer only paints those paths.
    """

    # Built-in institution -> web domain seeds. Western services (Clearbit
    # autocomplete, Google/DuckDuckGo favicons, Wikimedia) have poor coverage of
    # Chinese/Asian universities, so seeding the real domain lets the site-icon
    # resolver fetch the logo directly from the institution's own site. Config
    # `known_domains` overrides these.
    _DEFAULT_KNOWN_DOMAINS: Dict[str, str] = {
        "Tsinghua University": "tsinghua.edu.cn",
        "Peking University": "pku.edu.cn",
        "Shanghai Jiao Tong University": "sjtu.edu.cn",
        "Fudan University": "fudan.edu.cn",
        "Zhejiang University": "zju.edu.cn",
        "University of Science and Technology of China": "ustc.edu.cn",
        "Nanjing University": "nju.edu.cn",
        "East China Normal University": "ecnu.edu.cn",
        "Beihang University": "buaa.edu.cn",
        "Beijing Institute of Technology": "bit.edu.cn",
        "Harbin Institute of Technology": "hit.edu.cn",
        "Wuhan University": "whu.edu.cn",
        "Huazhong University of Science and Technology": "hust.edu.cn",
        "Sun Yat-sen University": "sysu.edu.cn",
        "Xi'an Jiaotong University": "xjtu.edu.cn",
        "Tongji University": "tongji.edu.cn",
        "Nankai University": "nankai.edu.cn",
        "Tianjin University": "tju.edu.cn",
        "Sichuan University": "scu.edu.cn",
        "Xiamen University": "xmu.edu.cn",
        "Southeast University": "seu.edu.cn",
        "Beijing University of Posts and Telecommunications": "bupt.edu.cn",
        "Renmin University of China": "ruc.edu.cn",
        "Central South University": "csu.edu.cn",
        "Shandong University": "sdu.edu.cn",
        "Beijing Normal University": "bnu.edu.cn",
        "Dalian University of Technology": "dlut.edu.cn",
        "South China University of Technology": "scut.edu.cn",
        "University of Electronic Science and Technology of China": "uestc.edu.cn",
        "Northwestern Polytechnical University": "nwpu.edu.cn",
        "Chongqing University": "cqu.edu.cn",
        "Jilin University": "jlu.edu.cn",
        "Hunan University": "hnu.edu.cn",
        "University of Chinese Academy of Sciences": "ucas.ac.cn",
        "Chinese Academy of Sciences": "cas.cn",
        "The Hong Kong University of Science and Technology": "ust.hk",
        "Hong Kong University of Science and Technology": "ust.hk",
        "The University of Hong Kong": "hku.hk",
        "The Chinese University of Hong Kong": "cuhk.edu.hk",
        "The Hong Kong Polytechnic University": "polyu.edu.hk",
        "City University of Hong Kong": "cityu.edu.hk",
        "Nanyang Technological University": "ntu.edu.sg",
        "National University of Singapore": "nus.edu.sg",
    }

    def __init__(self):
        self.name = "affiliation_logo_agent"
        self.config = load_config().get("affiliation_logos", {})
        self.timeout = self.config.get("request_timeout_seconds", 20)
        self.max_logos = self.config.get("max_logos", 4)
        self.clearbit_base_url = self.config.get("clearbit_base_url", "https://logo.clearbit.com").rstrip("/")
        self.known_domains = {**self._DEFAULT_KNOWN_DOMAINS, **self.config.get("known_domains", {})}
        self.official_logo_urls = self.config.get("official_logo_urls", {})
        self.known_commons_files = self.config.get("known_commons_files", {})
        self.local_dirs = self.config.get("local_dirs", ["affiliation_logos", "logos"])
        self.min_logo_long_edge = int(self.config.get("min_logo_long_edge", 320))
        self.normalized_max_size = tuple(self.config.get("normalized_max_size", [1800, 720]))
        # populated by _try_openalex_institutions; maps institution name → Wikidata QID URL
        self._openalex_wikidata_cache: dict[str, str] = {}

    def __call__(self, state: PosterState) -> PosterState:
        if not state.get("enable_affiliation_logos", False):
            state["affiliation_logos"] = []
            self._save_outputs(state, [], state.get("affiliations") or [])
            return state

        try:
            affiliations = self._get_affiliations(state)
            output_dir = Path(state["output_dir"])
            logo_dir = output_dir / "assets" / "affiliation_logos"
            logo_dir.mkdir(parents=True, exist_ok=True)

            # 'single' places one institution logo (default); 'multi' places up to the
            # configured maximum (currently 3), based on how many actually resolve.
            mode = str(state.get("affiliation_logo_mode") or "single").strip().lower()
            effective_max = 1 if mode == "single" else self.max_logos

            logos: List[Dict[str, Any]] = []
            seen_logo_keys = set()
            for affiliation in affiliations:
                if len(logos) >= effective_max:
                    break
                logo_key = self._canonical_logo_key(affiliation)
                if logo_key in seen_logo_keys:
                    continue
                entry = self._resolve_logo(affiliation, logo_dir)
                if entry:
                    logos.append(entry)
                    seen_logo_keys.add(logo_key)

            state["affiliation_logos"] = logos
            if logos and not (state.get("aff_logo_path") and Path(state["aff_logo_path"]).exists()):
                state["aff_logo_path"] = logos[0]["logo_path"]
            state["current_agent"] = self.name
            self._save_outputs(state, logos, affiliations)
            log_agent_success(self.name, f"resolved {len(logos)} affiliation logos")
        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")

        return state

    def _get_affiliations(self, state: PosterState) -> List[str]:
        # Prefer authoritative institution names from OpenAlex (by DOI, else by title).
        # The parser's raw affiliation strings are often garbled ("Technology Beijing
        # Institute"), which breaks every downstream logo lookup; OpenAlex returns
        # canonical names ("Beijing Institute of Technology").
        openalex_insts: List[str] = []
        doi = state.get("doi")
        if doi:
            openalex_insts = self._try_openalex_institutions(doi)
            source = f"DOI {doi}"
        if not openalex_insts:
            title = self._paper_title(state)
            if title:
                openalex_insts = self._try_openalex_by_title(title)
                source = "title search"
        if openalex_insts:
            log_agent_info(self.name, f"OpenAlex returned {len(openalex_insts)} institutions ({source})")
            seen = {n.lower() for n in openalex_insts}
            for name in (state.get("affiliations") or []):
                if name.lower() not in seen:
                    openalex_insts.append(name)
                    seen.add(name.lower())
            return openalex_insts[:6]

        affiliations = state.get("affiliations") or []
        if not affiliations:
            narrative = state.get("narrative_content") or {}
            affiliations = narrative.get("meta", {}).get("affiliations", [])

        deduped: List[str] = []
        seen = set()
        for name in affiliations:
            normalized = self._normalize_name(str(name))
            key = normalized.lower()
            if normalized and key not in seen:
                deduped.append(normalized)
                seen.add(key)
        return deduped

    def _try_openalex_institutions(self, doi: str) -> List[str]:
        """Query OpenAlex for authoritative institution names for a DOI."""
        try:
            url = f"https://api.openalex.org/works/doi:{doi}"
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "PosterMELD/1.0"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            seen: set[str] = set()
            names: List[str] = []
            for authorship in data.get("authorships", []):
                for inst in authorship.get("institutions", []):
                    raw_name = inst.get("display_name", "").strip()
                    wikidata_id = inst.get("ids", {}).get("wikidata", "")
                    if not raw_name:
                        continue
                    key = raw_name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    # Store wikidata ID alongside name so _resolve_logo can use it
                    self._openalex_wikidata_cache[raw_name] = wikidata_id
                    names.append(raw_name)
            return names
        except Exception:
            return []

    def _paper_title(self, state: PosterState) -> Optional[str]:
        title = state.get("title") or state.get("paper_title")
        if not title:
            meta = (state.get("narrative_content") or {}).get("meta") or {}
            title = meta.get("poster_title") or meta.get("title")
        title = str(title or "").strip()
        return title or None

    def _try_openalex_by_title(self, title: str) -> List[str]:
        """Look up authoritative institution names by paper title (works for arXiv
        papers that have no DOI)."""
        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params={"filter": f"title.search:{title}", "per-page": 1},
                timeout=self.timeout,
                headers={"User-Agent": "PosterMELD/1.0 (mailto:noreply@example.com)"},
            )
            results = resp.json().get("results", []) if resp.status_code == 200 else []
        except Exception as exc:
            log_agent_warning(self.name, f"OpenAlex title lookup failed: {exc}")
            return []
        if not results:
            return []
        names: List[str] = []
        seen: set[str] = set()
        for authorship in results[0].get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name")
                if not name or name.lower() in seen:
                    continue
                names.append(name)
                seen.add(name.lower())
                ids = inst.get("ids") if isinstance(inst.get("ids"), dict) else {}
                wikidata = (ids or {}).get("wikidata") or inst.get("wikidata")
                if wikidata:
                    self._openalex_wikidata_cache[name] = wikidata
        return names

    def _lookup_domain_via_autocomplete(self, institution: str) -> Optional[str]:
        """Resolve an institution name to its real web domain via Clearbit's free
        autocomplete API — far more reliable than guessing a domain from initials."""
        cache = getattr(self, "_domain_autocomplete_cache", None)
        if cache is None:
            cache = {}
            self._domain_autocomplete_cache = cache
        key = institution.lower().strip()
        if key in cache:
            return cache[key]
        domain: Optional[str] = None
        try:
            resp = requests.get(
                "https://autocomplete.clearbit.com/v1/companies/suggest",
                params={"query": institution},
                timeout=self.timeout,
                headers={"User-Agent": "PosterMELD/1.0"},
            )
            hits = resp.json() if resp.status_code == 200 else []
            academic = [h for h in hits if any(t in (h.get("domain") or "") for t in (".edu", ".ac."))]
            chosen = academic[0] if academic else (hits[0] if hits else None)
            if chosen:
                domain = chosen.get("domain")
        except Exception:
            domain = None
        cache[key] = domain
        return domain

    def _http_get(self, url: str) -> Optional[requests.Response]:
        """GET with a browser UA; retry once without TLS verification, because some
        university sites (notably Chinese .edu.cn) present incomplete cert chains."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        try:
            return requests.get(url, timeout=self.timeout, headers=headers, verify=True, allow_redirects=True)
        except requests.exceptions.SSLError:
            pass
        except Exception:
            return None
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        try:
            return requests.get(url, timeout=self.timeout, headers=headers, verify=False, allow_redirects=True)
        except Exception:
            return None

    def _best_icon_url(self, html: str, base_url: str) -> Optional[str]:
        """Pick the best icon URL declared in a page's <link rel=...icon...> tags,
        preferring apple-touch-icon and the largest declared size; skip SVG/mask icons
        (Pillow cannot rasterize them)."""
        from urllib.parse import urljoin

        best_url: Optional[str] = None
        best_score = -(10 ** 9)
        for tag in re.findall(r"<link\b[^>]*>", html, re.I):
            rel = re.search(r"rel\s*=\s*[\"']([^\"']+)[\"']", tag, re.I)
            href = re.search(r"href\s*=\s*[\"']([^\"']+)[\"']", tag, re.I)
            if not rel or not href:
                continue
            rel_val = rel.group(1).lower()
            if "icon" not in rel_val:
                continue
            url = urljoin(base_url, href.group(1).strip())
            path = url.lower().split("?")[0]
            size_match = re.search(r"sizes\s*=\s*[\"']?(\d+)", tag, re.I)
            score = int(size_match.group(1)) if size_match else 0
            if "apple-touch" in rel_val:
                score += 200  # apple-touch icons are usually >=180px, good base quality
            if "mask-icon" in rel_val or path.endswith(".svg"):
                score -= 5000  # monochrome/vector, unusable as a color logo
            if score > best_score:
                best_score, best_url = score, url
        return best_url

    def _download_site_icon_logo(self, domain: str, output_path: Path) -> Optional[str]:
        """Fetch the institution's own homepage and download the icon it declares via
        <link rel="icon"/"apple-touch-icon">. Far more reliable than third-party
        favicon services for sites (e.g. Chinese .edu.cn) that Google/DuckDuckGo cannot
        crawl, and often higher-resolution than a bare /favicon.ico."""
        if not domain:
            return None
        from io import BytesIO
        from urllib.parse import urljoin

        html = None
        base_url = None
        for candidate in (f"https://www.{domain}/", f"https://{domain}/"):
            resp = self._http_get(candidate)
            if resp is not None and resp.status_code == 200 and resp.text:
                html, base_url = resp.text, str(resp.url)
                break
        if not html:
            return None

        icon_url = self._best_icon_url(html, base_url) or urljoin(base_url, "/favicon.ico")
        resp = self._http_get(icon_url)
        if resp is None or resp.status_code != 200 or len(resp.content) < 400:
            return None
        try:
            with Image.open(BytesIO(resp.content)) as img:
                img = img.convert("RGBA")
                long_edge = max(img.size)
                target = max(self.min_logo_long_edge + 192, 512)
                if long_edge < target:
                    scale = target / max(long_edge, 1)
                    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
                img.save(output_path, "PNG")
            if self._normalize_image_file(output_path):
                return str(output_path)
            output_path.unlink(missing_ok=True)
        except Exception:
            output_path.unlink(missing_ok=True)
        return None

    def _download_favicon_logo(self, domain: str, output_path: Path) -> Optional[str]:
        """Last-resort, license-safe logo: the site's high-res favicon. Lower quality
        than a real logo but reliable when no logo asset can be found."""
        if not domain:
            return None
        from io import BytesIO
        for url in (
            f"https://www.google.com/s2/favicons?domain={domain}&sz=256",
            f"https://icons.duckduckgo.com/ip3/{domain}.ico",
        ):
            try:
                resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0 PosterMELD/1.0"})
            except Exception:
                continue
            if resp.status_code != 200 or "image" not in resp.headers.get("content-type", "") or len(resp.content) < 1500:
                continue
            try:
                with Image.open(BytesIO(resp.content)) as img:
                    img = img.convert("RGBA")
                    long_edge = max(img.size)
                    target = max(self.min_logo_long_edge + 192, 512)
                    if long_edge < target:
                        scale = target / max(long_edge, 1)
                        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
                    img.save(output_path, "PNG")
                if self._normalize_image_file(output_path):
                    return str(output_path)
                output_path.unlink(missing_ok=True)
            except Exception:
                output_path.unlink(missing_ok=True)
                continue
        return None

    def _resolve_logo(self, institution: str, logo_dir: Path) -> Optional[Dict[str, Any]]:
        institution = self._canonical_institution_name(institution)
        domain = self._resolve_domain(institution)
        slug = self._slugify(institution)
        output_path = logo_dir / f"{slug}.png"

        local_logo = self._find_local_logo(institution, logo_dir.parent.parent)
        if local_logo:
            cached = self._copy_logo_to_cache(local_logo, output_path)
            if cached:
                return self._make_logo_entry(institution, domain, cached, "local_asset", "resolved")

        official_logo = self._download_official_logo(institution, output_path)
        if official_logo:
            return self._make_logo_entry(institution, domain, official_logo, "official_site", "resolved")

        known_commons_logo = self._download_known_commons_logo(institution, output_path)
        if known_commons_logo:
            return self._make_logo_entry(institution, domain, known_commons_logo, "wikimedia_commons_known", "resolved")

        wikidata_logo = self._download_wikidata_logo(institution, output_path)
        if wikidata_logo:
            return self._make_logo_entry(institution, domain, wikidata_logo, "wikimedia_commons", "resolved")

        if domain:
            downloaded = self._download_clearbit_logo(domain, output_path)
            if downloaded:
                return self._make_logo_entry(institution, domain, downloaded, "clearbit", "resolved")
            site_icon = self._download_site_icon_logo(domain, output_path)
            if site_icon:
                return self._make_logo_entry(institution, domain, site_icon, "site_icon", "resolved")
            favicon = self._download_favicon_logo(domain, output_path)
            if favicon:
                return self._make_logo_entry(institution, domain, favicon, "favicon", "resolved")
            log_agent_warning(self.name, f"logo download failed for {institution} ({domain})")

        if self.config.get("include_placeholders", True):
            placeholder = self._create_placeholder_logo(institution, output_path)
            return self._make_logo_entry(institution, domain, placeholder, "placeholder", "placeholder")

        return None

    def _resolve_domain(self, institution: str) -> Optional[str]:
        institution = self._canonical_institution_name(institution)
        if institution in self.known_domains:
            return self.known_domains[institution]

        lowered = institution.lower()
        for known_name, domain in self.known_domains.items():
            if lowered == known_name.lower() or lowered in known_name.lower() or known_name.lower() in lowered:
                return domain

        autocompleted = self._lookup_domain_via_autocomplete(institution)
        if autocompleted:
            return autocompleted

        return self._guess_domain(institution)

    def _canonical_logo_key(self, institution: str) -> str:
        local_logo = self._find_local_logo(institution, None)
        if local_logo:
            return f"local:{local_logo.resolve()}"
        commons_file = self._resolve_known_commons_file(institution)
        if commons_file:
            return f"commons:{commons_file.lower()}"
        domain = self._resolve_domain(institution)
        if domain:
            return f"domain:{domain.lower()}"
        return f"name:{institution.lower()}"

    def _guess_domain(self, institution: str) -> Optional[str]:
        name = institution.lower()
        aliases = {
            "washington university in st. louis": "wustl.edu",
            "washington university": "wustl.edu",
            "george mason university": "gmu.edu",
            "tsinghua university": "tsinghua.edu.cn",
            "beijing university of posts and telecommunications": "bupt.edu.cn",
            "the chinese university of hong kong": "cuhk.edu.hk",
            "chinese university of hong kong": "cuhk.edu.hk",
            "university of illinois chicago": "uic.edu",
            "university of illinois at chicago": "uic.edu",
            "hong kong university of science and technology guangzhou": "hkust-gz.edu.cn",
            "hong kong university of science and technology": "hkust.edu.hk",
        }
        for key, domain in aliases.items():
            if key in name:
                return domain

        compact = re.sub(r"[^a-z0-9 ]", "", name)
        compact = re.sub(r"\b(the|of|at|in|and|school|college|department|division)\b", "", compact)
        words = [word for word in compact.split() if word]
        if len(words) >= 2 and any(token in name for token in ("university", "institute", "college")):
            return "".join(word[0] for word in words[:4]) + ".edu"
        return None

    def _find_local_logo(self, institution: str, output_dir: Optional[Path]) -> Optional[Path]:
        candidates: List[Path] = []
        if output_dir:
            pdf_root = Path(output_dir).parent.parent / "data" / Path(output_dir).name
            candidates.extend(self._candidate_local_dirs(pdf_root))

        slug = self._slugify(institution)
        name_tokens = set(slug.split("-"))
        for directory in candidates:
            if not directory.exists():
                continue
            files = [
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
            ]
            for path in files:
                stem = self._slugify(path.stem)
                if stem == slug or slug in stem or stem in slug:
                    return path
            for path in files:
                stem_tokens = set(self._slugify(path.stem).split("-"))
                if len(name_tokens & stem_tokens) >= 2:
                    return path
        return None

    def _candidate_local_dirs(self, paper_dir: Path) -> List[Path]:
        dirs = [paper_dir]
        dirs.extend(paper_dir / dirname for dirname in self.local_dirs)
        return dirs

    def _copy_logo_to_cache(self, source_path: Path, output_path: Path) -> str:
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
        if source_path.suffix.lower() == ".svg":
            if not self._convert_svg_to_png(source_path, output_path):
                return ""
        else:
            with Image.open(source_path) as img:
                img.convert("RGBA").save(output_path, "PNG")
        if not self._normalize_image_file(output_path):
            return ""
        return str(output_path)

    def _download_clearbit_logo(self, domain: str, output_path: Path) -> Optional[str]:
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        url = f"{self.clearbit_base_url}/{domain}"
        try:
            response = requests.get(url, timeout=self.timeout, headers={"User-Agent": "PosterMELD/1.0"})
            response.raise_for_status()
            if "image" not in response.headers.get("content-type", ""):
                return None
            content_type = response.headers.get("content-type", "")
            if "svg" in content_type:
                svg_path = output_path.with_suffix(".svg")
                svg_path.write_bytes(response.content)
                if not self._convert_svg_to_png(svg_path, output_path):
                    svg_path.unlink(missing_ok=True)
                    return None
                svg_path.unlink(missing_ok=True)
            else:
                output_path.write_bytes(response.content)
            if not self._normalize_image_file(output_path):
                output_path.unlink(missing_ok=True)
                return None
            return str(output_path)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    def _download_official_logo(self, institution: str, output_path: Path) -> Optional[str]:
        urls = self._resolve_official_logo_urls(institution)
        for url in urls:
            downloaded = self._download_url_logo(url, output_path)
            if downloaded:
                return downloaded
        return None

    def _resolve_official_logo_urls(self, institution: str) -> List[str]:
        if institution in self.official_logo_urls:
            return list(self.official_logo_urls[institution])
        lowered = institution.lower()
        for known_name, urls in self.official_logo_urls.items():
            known_lowered = known_name.lower()
            if lowered == known_lowered or lowered in known_lowered or known_lowered in lowered:
                return list(urls)
        return []

    def _download_url_logo(self, url: str, output_path: Path) -> Optional[str]:
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
        try:
            try:
                response = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0 PosterMELD/1.0"})
            except requests.exceptions.SSLError:
                response = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0 PosterMELD/1.0"}, verify=False)
            response.raise_for_status()
            if "image" not in response.headers.get("content-type", ""):
                return None
            content_type = response.headers.get("content-type", "")
            if "svg" in content_type or url.lower().split("?", 1)[0].endswith(".svg"):
                svg_path = output_path.with_suffix(".svg")
                svg_path.write_bytes(response.content)
                if not self._convert_svg_to_png(svg_path, output_path):
                    svg_path.unlink(missing_ok=True)
                    return None
                svg_path.unlink(missing_ok=True)
            else:
                output_path.write_bytes(response.content)
            if not self._normalize_image_file(output_path):
                output_path.unlink(missing_ok=True)
                return None
            return str(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)
            return None

    def _download_known_commons_logo(self, institution: str, output_path: Path) -> Optional[str]:
        filename = self._resolve_known_commons_file(institution)
        if not filename:
            return None
        return self._download_commons_file(filename, output_path)

    def _resolve_known_commons_file(self, institution: str) -> Optional[str]:
        if institution in self.known_commons_files:
            return self.known_commons_files[institution]
        lowered = institution.lower()
        for known_name, filename in self.known_commons_files.items():
            known_lowered = known_name.lower()
            if lowered == known_lowered or lowered in known_lowered or known_lowered in lowered:
                return filename
        return None

    def _download_commons_file(self, filename: str, output_path: Path) -> Optional[str]:
        try:
            file_url = self._get_commons_thumbnail_url(filename) or f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width=1800"
            response = requests.get(file_url, timeout=self.timeout, headers={"User-Agent": "PosterMELD/1.0"})
            response.raise_for_status()
            if "image" not in response.headers.get("content-type", ""):
                return None
            content_type = response.headers.get("content-type", "")
            if "svg" in content_type:
                svg_path = output_path.with_suffix(".svg")
                svg_path.write_bytes(response.content)
                if not self._convert_svg_to_png(svg_path, output_path):
                    svg_path.unlink(missing_ok=True)
                    return None
                svg_path.unlink(missing_ok=True)
            else:
                output_path.write_bytes(response.content)
            if not self._normalize_image_file(output_path):
                output_path.unlink(missing_ok=True)
                return None
            return str(output_path)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    def _get_commons_thumbnail_url(self, filename: str) -> Optional[str]:
        """Resolve a high-resolution raster thumbnail URL for PNG/SVG Commons files."""
        params = {
            "action": "query",
            "format": "json",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": 1800,
        }
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": "PosterMELD/1.0"},
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            return info.get("thumburl") or info.get("url")
        return None

    def _download_wikidata_logo(self, institution: str, output_path: Path) -> Optional[str]:
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        try:
            # Prefer the QID from OpenAlex (exact entity, no fuzzy search)
            cached_qid_url = self._openalex_wikidata_cache.get(institution, "")
            if cached_qid_url:
                # QID URL looks like "https://www.wikidata.org/entity/Q49117"
                entity_id = cached_qid_url.rstrip("/").rsplit("/", 1)[-1]
            else:
                entity_id = self._search_wikidata_entity(institution)
            if not entity_id:
                return None
            filename = self._get_wikidata_image_filename(entity_id)
            if not filename:
                return None
            return self._download_commons_file(filename, output_path)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None

    def _search_wikidata_entity(self, institution: str) -> Optional[str]:
        params = {
            "action": "wbsearchentities",
            "search": institution,
            "language": "en",
            "format": "json",
            "limit": 3,
        }
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": "PosterMELD/1.0"},
        )
        response.raise_for_status()
        for item in response.json().get("search", []):
            label = item.get("label", "").lower()
            description = item.get("description", "").lower()
            if any(token in f"{label} {description}" for token in ("university", "school", "college", "institution")):
                return item.get("id")
        search = response.json().get("search", [])
        return search[0].get("id") if search else None

    def _get_wikidata_image_filename(self, entity_id: str) -> Optional[str]:
        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "claims",
            "format": "json",
        }
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": "PosterMELD/1.0"},
        )
        response.raise_for_status()
        claims = response.json().get("entities", {}).get(entity_id, {}).get("claims", {})
        for property_id in ("P154", "P94"):
            for claim in claims.get(property_id, []):
                value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(value, str):
                    return value
        return None

    def _normalize_image_file(self, path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img = img.convert("RGBA")
                img = self._trim_logo_whitespace(img)
                if not self._is_usable_logo_image(img) and self._looks_like_white_transparent_logo(img):
                    img = self._recolor_visible_pixels(img, self.config.get("monochrome_logo_color", "#1E3A8A"))
                    img = self._trim_logo_whitespace(img)
                if not self._is_usable_logo_image(img):
                    return False
                img.thumbnail(self.normalized_max_size, Image.LANCZOS)
                canvas = Image.new("RGBA", img.size, (255, 255, 255, 0))
                canvas.alpha_composite(img)
                canvas.save(path, "PNG")
            return True
        except Exception:
            return False

    def _is_usable_logo_image(self, img: Image.Image) -> bool:
        width, height = img.size
        if max(width, height) < self.min_logo_long_edge:
            return False
        if min(width, height) < 32:
            return False
        pixels = img.load()
        visible = 0
        non_white = 0
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a > 12:
                    visible += 1
                    if not (r > 246 and g > 246 and b > 246):
                        non_white += 1
        return visible > 0 and non_white / max(visible, 1) > 0.01

    def _looks_like_white_transparent_logo(self, img: Image.Image) -> bool:
        pixels = img.load()
        width, height = img.size
        visible = 0
        near_white = 0
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a > 12:
                    visible += 1
                    if r > 235 and g > 235 and b > 235:
                        near_white += 1
        return visible > 0 and near_white / visible > 0.96

    def _recolor_visible_pixels(self, img: Image.Image, hex_color: str) -> Image.Image:
        hex_color = str(hex_color).lstrip("#")
        try:
            target = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            target = (30, 58, 138)
        recolored = Image.new("RGBA", img.size, (255, 255, 255, 0))
        src = img.load()
        dst = recolored.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                r, g, b, a = src[x, y]
                if a > 12:
                    dst[x, y] = (*target, a)
        return recolored

    def _convert_svg_to_png(self, source_path: Path, output_path: Path) -> bool:
        try:
            import cairosvg  # type: ignore

            cairosvg.svg2png(url=str(source_path), write_to=str(output_path), output_width=1800)
            return True
        except Exception:
            convert = shutil.which("rsvg-convert")
            if not convert:
                return False
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                import subprocess

                subprocess.run([convert, "-w", "1800", "-o", str(tmp_path), str(source_path)], check=True)
                tmp_path.replace(output_path)
                return True
            except Exception:
                tmp_path.unlink(missing_ok=True)
                return False

    def _trim_logo_whitespace(self, img: Image.Image) -> Image.Image:
        pixels = img.load()
        width, height = img.size
        xs: List[int] = []
        ys: List[int] = []
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                is_visible = a > 12
                is_non_white = not (r > 244 and g > 244 and b > 244)
                if is_visible and is_non_white:
                    xs.append(x)
                    ys.append(y)
        if not xs or not ys:
            return img
        pad_x = max(4, int((max(xs) - min(xs) + 1) * 0.05))
        pad_y = max(4, int((max(ys) - min(ys) + 1) * 0.08))
        box = (
            max(min(xs) - pad_x, 0),
            max(min(ys) - pad_y, 0),
            min(max(xs) + pad_x + 1, width),
            min(max(ys) + pad_y + 1, height),
        )
        return img.crop(box)

    def _create_placeholder_logo(self, institution: str, output_path: Path) -> str:
        initials = self._initials(institution)
        width, height = 640, 260
        bg = (238, 242, 248, 255)
        accent = (38, 74, 120, 255)
        image = Image.new("RGBA", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((14, 14, width - 14, height - 14), radius=34, outline=accent, width=6)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 96)
            small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        text_box = draw.textbbox((0, 0), initials, font=font)
        draw.text(
            ((width - (text_box[2] - text_box[0])) / 2, 50),
            initials,
            fill=accent,
            font=font,
        )
        label = self._short_label(institution)
        label_box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            ((width - (label_box[2] - label_box[0])) / 2, 174),
            label,
            fill=(54, 63, 75, 255),
            font=small_font,
        )
        image.save(output_path, "PNG")
        return str(output_path)

    def _make_logo_entry(
        self,
        institution: str,
        domain: Optional[str],
        logo_path: str,
        source: str,
        status: str,
    ) -> Dict[str, Any]:
        aspect = 1.0
        try:
            with Image.open(logo_path) as img:
                aspect = img.size[0] / max(img.size[1], 1)
        except Exception:
            pass
        return {
            "institution": institution,
            "domain": domain,
            "logo_path": logo_path,
            "source": source,
            "status": status,
            "aspect": aspect,
        }

    def _save_outputs(self, state: PosterState, logos: List[Dict[str, Any]], affiliations: List[str]) -> None:
        content_dir = Path(state["output_dir"]) / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / "affiliations.json").write_text(
            json.dumps(affiliations, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (content_dir / "affiliation_logos.json").write_text(
            json.dumps(logos, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _normalize_name(self, name: str) -> str:
        return self._canonical_institution_name(re.sub(r"\s+", " ", name).strip(" ,.;"))

    def _canonical_institution_name(self, name: str) -> str:
        normalized = re.sub(r"\s+", " ", str(name)).strip(" ,.;")
        normalized = normalized.replace("Hongkong", "Hong Kong")
        aliases = {
            "The Chinese University of Hong Kong": "The Chinese University of Hong Kong",
            "The Chinese University of Hong Kong University": "The Chinese University of Hong Kong",
            "Hong Kong University of Science and Technology (Guangzhou)": "Hong Kong University of Science and Technology (Guangzhou)",
            "Hong Kong University of Science and Technology Guangzhou": "Hong Kong University of Science and Technology (Guangzhou)",
            "University of Illinois at Chicago": "University of Illinois Chicago",
        }
        lowered = normalized.lower()
        for alias, canonical in aliases.items():
            if lowered == alias.lower():
                return canonical
        return normalized

    def _slugify(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return slug or "affiliation-logo"

    def _initials(self, name: str) -> str:
        stop = {"the", "of", "at", "in", "and", "for", "school", "college", "department", "division"}
        words = [word for word in re.findall(r"[A-Za-z]+", name) if word.lower() not in stop]
        return "".join(word[0].upper() for word in words[:4]) or "AFF"

    def _short_label(self, name: str) -> str:
        label = re.sub(r"\s+", " ", name).strip()
        return label if len(label) <= 34 else label[:31].rstrip() + "..."


def affiliation_logo_agent_node(state: PosterState) -> Dict[str, Any]:
    result = AffiliationLogoAgent()(state)
    return {
        **state,
        "affiliation_logos": result.get("affiliation_logos", []),
        "current_agent": result.get("current_agent"),
        "errors": result.get("errors", []),
    }
