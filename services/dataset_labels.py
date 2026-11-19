"""User-facing dataset names while preserving stable internal identifiers."""

_DATASET_LABELS = {
    "Gasperini": "Enhancer screen — high MOI (Gasperini)",
    "Replogle": "Genome-wide screen — low MOI (Replogle)",
    "Custom": "Custom",
}


def dataset_label(dataset: str) -> str:
    return _DATASET_LABELS.get(dataset, dataset)
