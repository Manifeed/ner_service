from __future__ import annotations

from app.schemas.ner_schema import ArticleTheme


COMMON_LABELS = frozenset({"PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "PRODUCT"})

THEME_LABELS: dict[ArticleTheme, set[str]] = {
    "politics": {"GOVERNMENT_BODY", "LAW", "POLITICAL_PARTY"},
    "economy": {"COMPANY", "MARKET_INDEX", "CURRENCY", "FINANCIAL_INSTRUMENT"},
    "business": {"COMPANY", "MARKET_INDEX", "CURRENCY", "FINANCIAL_INSTRUMENT"},
    "finance": {"COMPANY", "MARKET_INDEX", "CURRENCY", "FINANCIAL_INSTRUMENT"},
    "sports": {"TEAM", "COMPETITION"},
    "technology": {"SOFTWARE", "HARDWARE", "TECHNOLOGY"},
    "science": {"RESEARCH_FIELD"},
    "health": {"RESEARCH_FIELD", "DISEASE", "MEDICAL_TREATMENT"},
    "society": set(),
    "news": set(),
    "culture": set(),
    "environment": set(),
    "world": set(),
    "justice": {"LAW"},
    "education": set(),
    "other": set(),
}


def resolve_ner_labels(themes: list[ArticleTheme]) -> list[str]:
    labels = set(COMMON_LABELS)
    for theme in themes:
        labels.update(THEME_LABELS.get(theme, set()))
    return sorted(labels)
