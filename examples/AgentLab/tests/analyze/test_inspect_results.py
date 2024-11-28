from pathlib import Path
import shutil
import tempfile

import pandas as pd
from agentlab.analyze.inspect_results import get_study_summary


def test_get_study_summary():

    with tempfile.TemporaryDirectory() as tmp_dir:
        study_dir = Path(tmp_dir) / "test_study"

        study_dir_original = Path(__file__).parent.parent / "data" / "test_study"

        # recursively copy the study to the temp dir using shutil
        shutil.copytree(study_dir_original, study_dir)

        sentinel = {}

        summary = get_study_summary(study_dir, sentinel=sentinel)
        assert isinstance(summary, pd.DataFrame)
        assert sentinel["from_cache"] == False

        summary = get_study_summary(study_dir, sentinel=sentinel)
        assert isinstance(summary, pd.DataFrame)
        assert sentinel["from_cache"] == True

        summary = get_study_summary(study_dir, ignore_cache=True, sentinel=sentinel)
        assert isinstance(summary, pd.DataFrame)
        assert sentinel["from_cache"] == False


if __name__ == "__main__":
    test_get_study_summary()
