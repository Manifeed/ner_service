from __future__ import annotations

COMMON_LABELS = frozenset({
	"PERSON",
	"ORGANIZATION",
	"COMPANY",
	"LOCATION",
	"DATE",
	"EVENT",
	"PRODUCT",
	"TECHNOLOGY",
	"CURRENCY",
})

NER_LABELS: tuple[str, ...] = tuple(sorted(COMMON_LABELS))
