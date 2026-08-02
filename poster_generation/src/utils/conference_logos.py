"""Conference logo resolution: name string → local PNG path."""

import re
from pathlib import Path
from typing import Optional

# Canonical slug → list of name aliases (lowercase, year-stripped)
_ALIASES: dict[str, list[str]] = {
    "acs": [
        "acs", "acs publications", "american chemical society",
        "journal of chemical information and modeling",
        "j. chem. inf. model",
    ],
    "cvpr":    ["cvpr", "computer vision and pattern recognition"],
    "iccv":    ["iccv", "international conference on computer vision"],
    "eccv":    ["eccv", "european conference on computer vision"],
    "neurips": [
        "neurips", "nips", "neural information processing systems",
        "datasets and benchmarks track",
        "ml4ad", "machine learning for autonomous driving",
        "machine learning and the physical sciences", "ml4physical",
        "table representation learning", "ts4h", "time series for health",
        "icbinb", "tsrml", "trustworthy ml", "trustworthy machine learning",
        "machine learning for structural biology", "mlsb",
        "ml reproducibility challenge", "reproducibility challenge",
    ],
    "icml":    ["icml", "international conference on machine learning"],
    "iclr":    [
        "iclr", "international conference on learning representations",
        "gem workshop", "gem", "climate change ai", "ccai",
    ],
    "aaai":    ["aaai", "association for the advancement of artificial intelligence"],
    "acl":     ["acl", "annual meeting of the association for computational linguistics"],
    "findings_acl": ["findings of acl", "findings-acl", "findings acl"],
    "coling":  ["coling", "international conference on computational linguistics"],
    "colm":    ["colm", "conference on language modeling"],
    "jmlr":    ["jmlr", "journal of machine learning research"],
    "arxiv":   ["arxiv", "corr"],
    "f1000research": ["f1000research", "f1000 research"],
    "biorxiv": ["biorxiv", "bioRxiv"],
    "medrxiv": ["medrxiv", "medRxiv"],
    "bmc": [
        "bmc", "bmc bioinformatics", "conflict and health", "biology direct", "microbiome",
        "journal of translational medicine", "human genomics",
        "journal of biomedical semantics",
    ],
    "bioinformatics": ["bioinformatics"],
    "bmj": ["bmj", "bmj open"],
    "elsevier": ["elsevier", "current biology"],
    "frontiers": ["frontiers", "frontiers in"],
    "genome_biology": ["genome biology"],
    "genome_research": ["genome research"],
    "mdpi": ["mdpi", "international journal of molecular sciences", "diagnostics"],
    "molecular_cellular_proteomics": [
        "molecular & cellular proteomics",
        "molecular and cellular proteomics",
        "mcponline", "mcp",
    ],
    "mary_ann_liebert": ["mary ann liebert", "journal of computational biology"],
    "nucleic_acids_research": ["nucleic acids research"],
    "oup": [
        "oup", "oxford university press", "cerebral cortex",
        "cercor",
    ],
    "plos": ["plos", "plos one", "plos computational biology", "plos genetics", "plos biology"],
    "public_health_action": [
        "public health action",
        "international union against tuberculosis and lung disease",
        "the union",
    ],
    "springer_nature": [
        "springer", "springer nature", "springerlink",
        "scientific reports", "nature communications", "parasitol res",
        "parasitology research", "psychopharmacology", "npj",
    ],
    "wiley": ["wiley", "ecology and evolution", "conservation letters"],
    "emnlp":   ["emnlp", "empirical methods in natural language processing"],
    "naacl":   ["naacl", "north american chapter of the association"],
    "ijcai":   ["ijcai", "international joint conference on artificial intelligence"],
    "kdd":     ["kdd", "knowledge discovery and data mining"],
    "www":     ["www", "world wide web", "the web conference"],
    "sigir":   ["sigir", "research and development in information retrieval"],
    "mm":      ["acmmm", "acm mm", "acm multimedia"],
    "siggraph":["siggraph"],
    "wacv":    ["wacv", "winter conference on applications of computer vision"],
    "miccai":  ["miccai", "medical image computing and computer assisted intervention"],
}

_LOGO_DIR = Path(__file__).parent.parent.parent / "assets" / "conference_logos"


def _strip_year(name: str) -> str:
    return re.sub(r"\b(19|20)\d{2}\b", "", name).strip()


def _year(name: str) -> Optional[str]:
    match = re.search(r"\b((?:19|20)\d{2})\b", name)
    return match.group(1) if match else None


def _slug(name: str) -> Optional[str]:
    """Map a free-form conference name to a canonical slug, or None."""
    normalized = _strip_year(name).lower()
    # direct slug match first
    for slug in _ALIASES:
        if slug == normalized:
            return slug
    # alias substring match
    for slug, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in normalized or normalized in alias:
                return slug
    return None


def resolve_conference_logo(conference_name: str) -> Optional[str]:
    """Return absolute path to conference logo PNG, or None if not found."""
    if not conference_name:
        return None
    slug = _slug(conference_name)
    if slug is None:
        return None
    year = _year(conference_name)
    if year:
        year_candidate = _LOGO_DIR / f"{slug}_{year}.png"
        if year_candidate.exists():
            return str(year_candidate)
    candidate = _LOGO_DIR / f"{slug}.png"
    return str(candidate) if candidate.exists() else None


def list_supported_conferences() -> list[str]:
    return sorted(_ALIASES.keys())
