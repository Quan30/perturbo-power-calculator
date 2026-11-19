import pandas as pd
import pytest
from utils_analysis import collectri


def test_regulons_keep_complexes_separate_deduplicate_and_exclude_self():
    raw = pd.DataFrame({
        "source_genesymbol": ["MYC", "MYC", "MYC", "MYC", "FOS_JUN", None],
        "target_genesymbol": ["TERT", "TERT", "MYC", " ", "JUN", "BAD"],
    })
    programs = collectri.prepare_collectri(raw)
    assert programs.to_dict("records") == [
        {"tf": "FOS_JUN", "target_gene": "JUN"},
        {"tf": "MYC", "target_gene": "TERT"},
    ]
    assert collectri.collectri_tfs(programs) == ["FOS_JUN", "MYC"]
    assert collectri.collectri_targets(programs, "MYC") == ["TERT"]
    assert collectri.collectri_targets(programs, "UNKNOWN") == []


def test_raw_schema_is_required():
    with pytest.raises(ValueError, match="source_genesymbol"):
        collectri.prepare_collectri(pd.DataFrame({"source": ["P01106"]}))
