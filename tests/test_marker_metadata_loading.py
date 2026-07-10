from pathlib import Path


def test_marker_alias_present_in_run_markers():
    text = Path('src/pipelines/experimental/sr_pipeline/run_markers.py').read_text()
    assert '--group-definitions-csv' in text

import pytest

@pytest.mark.skip(reason='Requires experimental dataset files not present in CI.')
def test_marker_metadata_file_exists_placeholder():
    pass
