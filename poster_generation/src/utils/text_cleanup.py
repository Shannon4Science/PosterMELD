"""Text cleanup helpers shared by layout and rendering agents."""

import re


COMMON_OCR_FIXES = {
    "Effcient": "Efficient",
    "effcient": "efficient",
    "Effciency": "Efficiency",
    "effciency": "efficiency",
}

TITLE_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "via",
    "vs",
    "with",
}

DANGLING_TERMINAL_WORDS = (
    "and|or|but|with|in|of|to|for|by|as|at|from|than|while|where|when|after|despite|"
    "that|which|through|into|over|under|within|via|on|only|also|past|a|an|the|this|"
    "these|those|their|its|using|including|exploiting|selecting|relying|letting|local|stale|"
    "may|can|could|would|should|will|must|is|are|was|were|be|been|being|"
    "typically|generally|often|roughly|approximately|consistently|significantly|"
    "limited|especially|particularly|notably|further"
    "|outperforming|improving|exceeding|achieving|injecting|creating|reducing|enabling|enables|enabled|called"
)


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake without touching clean text."""
    if not isinstance(text, str) or not text:
        return text

    text = (
        text.replace("â¢", "•")
        .replace("â¦", "◦")
        .replace("â", "-")
        .replace("â", "-")
    )

    if any(marker in text for marker in ("â", "Â", "Ã", "Î", "î", "ï¿½")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if repaired.count("\ufffd") <= text.count("\ufffd"):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return text


def strip_latex_and_markup(text: str) -> str:
    """Strip unrendered LaTeX math and HTML sub/sup markup from poster body text.

    Generated text sometimes contains raw LaTeX (``$x_i$``, ``$$ \\mathcal{L} = ...$$``)
    and ``<sub>``/``<sup>`` tags that PowerPoint renders verbatim. Besides looking wrong,
    these fragments (often truncated mid-equation) add several extra lines and push a
    block past its panel. Plainise them: unwrap sub/sup, drop math delimiters and LaTeX
    commands/braces, and tighten subscript underscores.
    """
    if not isinstance(text, str) or "$" not in text and "\\" not in text and "<su" not in text.lower():
        return text
    # <sub>i</sub> / <sup>2</sup> -> i / 2, then drop any stray sub/sup tags
    text = re.sub(r"<\s*sub\s*>(.*?)<\s*/\s*sub\s*>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?\s*(?:sub|sup)\s*>", "", text, flags=re.IGNORECASE)
    # math delimiters ($$ display, $ inline), LaTeX commands (\mathcal, \frac, ...), braces
    text = text.replace("$$", " ").replace("$", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", " ", text)
    # tighten "x _ i" subscript spacing to "x_i" and collapse whitespace
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def normalize_text_for_poster(text: str) -> str:
    """Normalize generated poster text before it reaches PowerPoint."""
    if not isinstance(text, str) or not text:
        return text

    text = repair_mojibake(text)
    text = strip_latex_and_markup(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("‐", "-")
    text = text.replace("‑", "-")
    text = text.replace("‒", "-")
    text = text.replace("−", "-")
    text = text.replace("Î»", "lambda ")
    text = text.replace("î»", "lambda ")

    for wrong, right in COMMON_OCR_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)

    normalized_lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*[•●]\s*", "• ", line.strip())
        line = re.sub(r"^\s*[◦▪▫]\s*", "◦ ", line)
        line = _strip_poster_artifact_noise(line)
        if not line:
            continue
        line = _repair_leading_bold_label(line)
        line = repair_truncated_sentence_end(line)
        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def normalize_title_for_poster(title: str) -> str:
    """Repair OCR in poster titles while preserving conventional title casing."""
    if not isinstance(title, str) or not title:
        return title

    text = repair_mojibake(title)
    text = (
        text.replace("\u00a0", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("−", "-")
    )
    for wrong, right in COMMON_OCR_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    words = text.split()
    normalized = []
    for index, word in enumerate(words):
        stripped = word.strip()
        match = re.match(r"^([(\"'“‘]*)(.*?)([)\"'”’,:;.!?]*)$", stripped)
        if not match:
            normalized.append(stripped)
            continue
        prefix, core, suffix = match.groups()
        if 0 < index < len(words) - 1 and core.lower() in TITLE_SMALL_WORDS:
            core = core.lower()
        normalized.append(f"{prefix}{core}{suffix}")
    return " ".join(normalized)


def repair_possessive_title_apostrophe(title: str) -> str:
    """Repair possessive apostrophes that model cleanup split into a lone S."""
    def replacement(match: re.Match[str]) -> str:
        word = match.group(1)
        if word.isupper():
            return match.group(0)
        return f"{word}'s "

    return re.sub(r"\b([A-Za-z][A-Za-z]+)\s+[sS]\s+(?=[A-Za-z])", replacement, str(title or ""))


def _strip_poster_artifact_noise(line: str) -> str:
    """Remove OCR, markdown, and metadata artifacts that should never appear on posters."""
    if not isinstance(line, str) or not line:
        return line

    original = line
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", line)
    line = re.sub(r"\[[^\]]{0,80}\]\([^)]*\.(?:png|jpe?g|pdf|svg)[^)]*\)", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\b[\w./\\-]*_page_\d+_[A-Za-z]+_\d+\.(?:png|jpe?g)\b", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\b(?:[\w.-]+/)+[\w.-]+\.(?:png|jpe?g|pdf|svg)\b", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\b[\w.-]+\.(?:png|jpe?g|svg)\b", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\s*\[[0-9,\s-]+\]", "", line)

    if re.fullmatch(
        r"\s*(?:the\s+)?(?:results|values|numbers|details|comparison|performance|ablation|evaluation)\s+"
        r"(?:are|is|were|was)\s+(?:shown|provided|presented|reported|summarized|listed|given)\s+in\s+"
        r"(?:table|tables|figure|figures|fig\.?|figs\.?)\s*\d+(?:\s*(?:and|,|-|to)\s*\d+)*\.?\s*",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.match(r"^\s*(?:fig(?:ure)?|table)\s*\d+[\.:]", line, flags=re.IGNORECASE):
        return ""
    line = re.sub(r"\b(?:fig(?:ure)?|table)\s*\d+[\.:]\s*[^.|;]*\.?", "", line, flags=re.IGNORECASE)
    if (
        re.search(r"\b(?:algorithm|appendix|supplement|supplementary)\b", line, flags=re.IGNORECASE)
        and re.search(r"\b(?:detailed|complete|provided|presentation|details?|see|refer|defer)\b", line, flags=re.IGNORECASE)
    ):
        return ""
    if re.match(r"^\s*we\s+also\s+provide\s+results?\s+of\b", line, flags=re.IGNORECASE):
        return ""
    if re.match(
        r"^\s*(?:next|then|finally|subsequently),?\s+we\s+"
        r"(?:describe|present|discuss|report|evaluate|detail)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.match(
        r"^\s*(?:in\s+this\s+section|in\s+the\s+next\s+section|below),?\s+we\s+"
        r"(?:describe|present|discuss|report|evaluate|detail)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.search(
        r"\bwe\s+compare\s+(?:the\s+proposed\s+approach|our\s+approach|the\s+method|our\s+method)?"
        r".{0,80}\b(?:to|against)\s+(?:the\s+)?following\s+baselines\b",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.search(r"\bintroduced\s+in\s+the\s+hierarchical\.\s*$", line, flags=re.IGNORECASE):
        return ""

    line = re.sub(
        r"\s+that\s+(?:jointly\s+)?handles\.\s*$",
        ".",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\s*[:;]\s*[A-Za-z][A-Za-z0-9_]*(?:\s*\([^)]+\))?\s*=\s*"
        r"[A-Za-z][A-Za-z0-9_]*(?:\s*[+\-*/]\s*[A-Za-z0-9_]+)*\.?\s*$",
        ".",
        line,
    )

    line = re.sub(
        r"\b(?:as\s+)?(?:shown|provided|presented|reported|summarized|listed|given)\s+in\s+"
        r"(?:table|tables|figure|figures|fig\.?|figs\.?)\s*\d+(?:\s*(?:and|,|-|to)\s*\d+)*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\b(?:see|cf\.?|from)\s+(?:table|tables|figure|figures|fig\.?|figs\.?)\s*\d+"
        r"(?:\s*(?:and|,|-|to)\s*\d+)*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\b(?:table|tables|figure|figures|fig\.?|figs\.?)\s*\d+(?:\s*(?:and|,|-|to)\s*\d+)*[\.:]?",
        "",
        line,
        flags=re.IGNORECASE,
    )

    if line.count("|") >= 2:
        prefix = line.split("|", 1)[0].strip()
        line = prefix if len(prefix.split()) >= 4 else ""

    line = re.sub(r"\s+\bimportance\s+(?:high|medium|low)\b.*$", "", line, flags=re.IGNORECASE)
    line = re.sub(
        r"\b(?:contains_figures|contains_tables|section_name|section_type|visual_assets|source_sections)\b\s*[:=]?\s*",
        " ",
        line,
        flags=re.IGNORECASE,
    )
    if _looks_like_metadata_only(line):
        return ""
    if _looks_like_bibliography(line):
        return ""
    if re.search(
        r"\b(?:USA|United States)\b.*\b(?:University|School|College|Department|Institute)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if re.search(
        r"\b(?:Brown School at Washington University|Washington University in St\.?\s*Louis|George Mason University)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return ""

    line = re.sub(r"\b(?:results|values|details|comparison|performance)\s+are\s+(?:in|shown|presented|provided|reported)\s*\.?", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\b(?:the\s+)?(?:table|figure)\s+(?:shows|reports|presents|summarizes)\b.*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s{2,}", " ", line).strip()
    line = re.sub(r"\s+([,.;:])", r"\1", line)
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"\s+[,;:]\s*$", ".", line).strip()

    if re.fullmatch(
        r"(?:the\s+)?(?:results|values|details|comparison|performance)\s+(?:are|is|were|was)\.?",
        line,
        flags=re.IGNORECASE,
    ):
        return ""
    if not line or re.fullmatch(r"[\W_]+", line):
        return ""
    if original.count("|") >= 2 and len(line.split()) < 4:
        return ""
    return line


def _looks_like_metadata_only(line: str) -> bool:
    if not line:
        return True
    lowered = line.lower()
    metadata_tokens = (
        "importance",
        "contains_figures",
        "contains_tables",
        "section_name",
        "section_type",
        "source_sections",
    )
    if any(token in lowered for token in metadata_tokens):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z-]+", line)
    metadata_labels = {"content", "foundation", "main", "support"}
    return bool(words) and len(words) <= 4 and all(word.lower() in metadata_labels for word in words)


def _looks_like_bibliography(line: str) -> bool:
    if not line:
        return False
    has_year = bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line))
    has_venue = bool(
        re.search(
            r"\b(?:journal|proceedings|conference|transactions|arxiv|doi|isbn|acm|ieee|neurips|icml|iclr)\b",
            line,
            flags=re.IGNORECASE,
        )
    )
    has_many_names = len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.", line)) >= 2
    return has_year and (has_venue or has_many_names)


def fit_complete_sentence_prefix(text: str, max_chars: int) -> str:
    """Fit only whole sentences; never turn a character slice into a sentence."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if sentence.strip()
    ]
    if len(sentences) <= 1:
        return text

    fitted: list[str] = []
    used = 0
    for sentence in sentences:
        projected = used + len(sentence) + (1 if fitted else 0)
        if projected > max_chars:
            break
        fitted.append(sentence)
        used = projected
    return " ".join(fitted) if fitted else sentences[0]


def repair_truncated_sentence_end(line: str) -> str:
    """Remove obvious dangling endings introduced by capacity-based truncation."""
    if not isinstance(line, str) or not line:
        return line
    line = re.sub(r"[,;:]\s*\.$", ".", line.strip())
    previous = None
    while previous != line:
        previous = line
        if line.count("(") > line.count(")"):
            open_paren = line.rfind("(")
            if open_paren >= 0 and len(line) - open_paren <= 160:
                line = line[:open_paren].rstrip(" ,;:") + "."
        line = re.sub(
            r"\s+(?:while|where|when|because|although|whereas)\s+[^.;:]{1,120}"
            r"\s+(?:may|can|could|would|should|will|must|is|are|was|were|be|been|being|with\s+[A-Z])\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\s+(?:while|where|when|because|although|whereas)\s+[A-Za-z]{1,12}\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r",\s+even\s+though\s+[^.;:]{1,120}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r"[,;:]\s+[^.;:]{1,120}\s+(?:may|can|could|would|should|will|must|is|are|was|were|be|been|being|with\s+[A-Z])\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+as\s+(?:a|an|the)\s+[A-Za-z-]{2,28}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+as\s+new\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r"\s+(?:creates?|created|creating|causes?|caused|causing|forms?|formed|forming|poses?|posed|posing|injects?|injected|injecting|includes?|included|including)\s+(?:a|an|the)?\s*[A-Za-z-]{0,28}\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+to\s+find\s+and\s+support\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+and\s+(?:reducing|increasing|improving|decreasing)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+and\s+(?:reduce|increase|improve|decrease|support)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+with\s+(?:especially|particularly|notably)\s+[A-Za-z-]{2,28}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r",\s+where\s+[^.;:]{1,160}\s+and\s+(?:statistically|computationally|operationally|empirically)\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"\s+to\s+(?:update|learn|scale|adapt|improve|reach|select|query|discover|estimate|predict)\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(?<!properties)(?<!parcels)\s+to\s+visit\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+for\s+large\s+urban\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+across\s+cost\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+within\s+a\s+[A-Za-z-]*(?:searc|regio|neighbo|budg)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+by\s+average\s+(?:number|numbe)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r"[;:]\s+(?:performance|results?|evaluation|analysis|target|targets?|policy|method)\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"[,;:]\s+and\s+the\s+average\s+number\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"[,;:]\s+(?:measuring|testing|reporting)\s+[^.;:]{1,72}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+and\s+lower\s+eviction\s+pr\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+to\s+handle\s+tens\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s*[-–]\s*multimodal\s+parcel\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r"\s+(?:large-area|small-area|city-scale|region-level|parcel-level|query-cost|travel-aware|budget-constrained|non-hierarchical|within-region)\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+for\s+thousands\s+of\s+[A-Za-z-]{1,24}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"[,]?\s+and\s+the\s+non-hierarchi\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+despite\s+(?:limited|scarce|restricted)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+under\s+(?:tight|limited|strict)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+with\s+(?:a|an|the)\s+[A-Za-z-]{0,16}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r",?\s+aiming\s+to\s+close\s+the\s+gap(?:\s+between)?\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+of\s+surface-anchored\s+volumetric\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+plus\s+lightweight\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+under\s+the\s+same\s+parameter\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r",\s+which\s+uses\s+a\s+3D\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+and\s+a\s+share\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+to\s+large\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+with\s+(?:either|any|the|a|an)\s+[A-Za-z-]*(?:unif|uniform|vi)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(?:by|with|under|to|for|and|over|via)\s+[A-Za-z-]*(?:princ|approxima|substant|tight|co|wit|mul|stronges|unif|vi)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(?:and|or|for|with|under|to|by|of)\s+[A-Za-z-]*(?:cos|thousan)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+by\s+first\s+[A-Za-z-]+ing(?:\s+[A-Za-z-]+){0,2}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+by\s+[A-Za-z-]+ing\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(?:with|using|via|by|for|to)\s+[A-Z]\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(?:[A-Za-z]|fo|fou|ou|ar|re|wi|pr|parc|non|larg|hierarchi|non-hierarchi|cos|evic|prob|unif|unifor|withi|cha|dis|se|lo|ri|mo|res|vis|analys|thousan|princ|approxima|substant|tight|co|wit|mul|stronges|vi)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(rf"\s+({DANGLING_TERMINAL_WORDS})\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(
            r"\s+and\s+(?:also|then|therefore|local|stale|limited|new|more|less|other)\.$",
            ".",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\s+instead\s+of\s+[^.;:]{1,64}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+using\s+[A-Za-z-]+(?:\s+[A-Za-z-]+){0,2}\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r",\s+(travel|local|geospatial|stale|limited)\.$", ".", line, flags=re.IGNORECASE)
        line = re.sub(r"[,;:]\s*\.$", ".", line.strip())
        line = re.sub(r"\.{2,}$", ".", line)
    return line


def _repair_leading_bold_label(line: str) -> str:
    """Turn malformed '**Label: rest' into '**Label:** rest'."""
    bullet_match = re.match(r"^([•◦]\s+)?\*\*([^*\n:]{2,48}):\s+(.*)$", line)
    if not bullet_match:
        trailing_match = re.match(r"^([•◦]\s+)?([^*\n:]{2,48}):\*\*\s+(.*)$", line)
        if not trailing_match:
            return line
        bullet = trailing_match.group(1) or ""
        label = trailing_match.group(2).strip()
        rest = trailing_match.group(3).strip()
        return f"{bullet}**{label}:** {rest}"
    bullet = bullet_match.group(1) or ""
    label = bullet_match.group(2).strip()
    rest = bullet_match.group(3).strip()
    return f"{bullet}**{label}:** {rest}"
