from io import BytesIO

import pytest

from services.uploads import save_uploaded_dataset


class ExampleUpload(BytesIO):
    def __init__(self, name: str, payload: bytes):
        super().__init__(payload)
        self.name = name
        self.size = len(payload)
        self.file_id = "example-id"


@pytest.mark.parametrize("suffix", ["h5ad", "h5mu"])
def test_custom_upload_is_saved_atomically_with_safe_name(tmp_path, suffix):
    uploaded = ExampleUpload(f"../custom.{suffix}", b"example-data")
    saved = save_uploaded_dataset(uploaded, tmp_path, allowed_suffixes={".h5ad", ".h5mu"}, prefix="custom")

    assert saved.parent == tmp_path
    assert saved.suffix == f".{suffix}"
    assert saved.read_bytes() == b"example-data"
    assert not list(tmp_path.glob("*.part"))


def test_custom_upload_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported upload type"):
        save_uploaded_dataset(
            ExampleUpload("counts.csv", b"x"),
            tmp_path,
            allowed_suffixes={".h5ad", ".h5mu"},
            prefix="custom",
        )
