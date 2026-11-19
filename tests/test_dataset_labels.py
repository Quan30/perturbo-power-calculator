from services.dataset_labels import dataset_label


def test_builtin_dataset_labels_explain_screen_design():
    assert dataset_label("Gasperini") == "Enhancer screen — high MOI (Gasperini)"
    assert dataset_label("Replogle") == "Genome-wide screen — low MOI (Replogle)"


def test_custom_dataset_label_is_unchanged():
    assert dataset_label("Custom") == "Custom"


def test_analysis3_formats_dataset_names_without_changing_values():
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "pages" / "analysis3_screen_power.py"
    text = page.read_text(encoding="utf-8")
    assert "format_func=dataset_label" in text
