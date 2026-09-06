# -*- coding: utf-8 -*-
"""The space explorer must survive missing data, end to end.

0.1.5 made the value functions NaN-native and 0.1.8 finishes the job, because
the ANALYSIS stage still assumed complete data:

    all_spaces = {'original': X}

put the raw matrix beside the behavioural spaces, MinMaxScaler carried a NaN
into the per-column bounds, and rng.uniform(nan, nan) raised

    OverflowError: Range exceeds valid bounds

before a single figure was drawn. The behavioural spaces themselves computed
perfectly first, which is what made it look like a plotting problem.

Unit-testing the pieces would not have caught this: the crash lives in the
seam between a NaN-native engine and a complete-data analysis. So these run the
whole script, which is also the only way to cover a file that is delivered as a
`%run -i` script rather than an importable module.
"""
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

SPACES = ("variance", "skewness", "kurtosis", "entropy")


def get_script_path(name):
    """The explorer script, whether we are testing the repo or an install.

    In the repo the scripts sit beside the single-file module at the root; in an
    installed package they live under shapley_behaviors/scripts. Tests run from
    the repo import the ROOT module rather than the installed package, so the
    packaged helper is not always importable.
    """
    try:
        from shapley_behaviors.scripts import get_script_path as packaged
        return Path(packaged(name))
    except Exception:
        root = Path(__file__).resolve().parent.parent / name
        if root.exists():
            return root
        raise


def _dataset(tmp_path, missing_fraction, n=24, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d) * np.array([1.0, 3.0, 0.2, 1.0, 5.0])
    if missing_fraction:
        mask = rng.rand(n, d) < missing_fraction
        # never blank a whole column: that is a separate degenerate case
        mask[0, :] = False
        X[mask] = np.nan
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(d)])
    df.insert(0, "sid", [f"s{i:02d}" for i in range(n)])
    df["group"] = rng.choice(["a", "b"], n)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _run(tmp_path, csv, **extra):
    cfg = dict(
        SEED=1, N_PERMUTATIONS=4, N_JOBS=1, DATASET_NAME="T",
        DATA_FILE=str(csv), ID_COLUMN="sid", DROP_COLUMNS=[],
        LABEL_COLUMNS=["group"], OUTPUT_DIR=str(tmp_path / "out"),
        SELECTED_FEATURES=None, CREATE_OUTLIER_PROFILES=False,
        MAX_OUTLIERS_PER_SPACE=1, SHOW_PLOTS=False,
    )
    cfg.update(extra)
    return runpy.run_path(str(get_script_path("behavioral_space_explorer.py")),
                          init_globals=cfg)


def test_runs_to_completion_with_missing_values(tmp_path, capsys):
    """The regression: this raised OverflowError before 0.1.8."""
    csv = _dataset(tmp_path, missing_fraction=0.25)
    _run(tmp_path, csv)
    out = capsys.readouterr().out
    assert "missing values" in out
    assert "left out of the space comparison" in out
    # and it says why, rather than dropping the space silently
    assert "impute" in out


def test_behavioural_spaces_are_finite_despite_gaps(tmp_path):
    """A missing entry contributes a zero marginal, so Phi stays finite."""
    csv = _dataset(tmp_path, missing_fraction=0.25)
    _run(tmp_path, csv)
    spaces = np.load(tmp_path / "out" / "T_behavioral_spaces.npy",
                     allow_pickle=True).item()
    assert set(SPACES) <= set(spaces)
    for name, phi in spaces.items():
        assert np.isfinite(phi).all(), f"{name} space has non-finite entries"


def test_complete_data_still_includes_the_original_space(tmp_path, capsys):
    """The guard must not fire when there is nothing missing."""
    csv = _dataset(tmp_path, missing_fraction=0.0)
    _run(tmp_path, csv)
    out = capsys.readouterr().out
    assert "left out of the space comparison" not in out
    assert "ORIGINAL SPACE" in out.upper()


def test_hopkins_is_computed_for_a_space_with_gaps():
    """Hopkins reduces over observed values, so a gapped space still scores.

    Checked directly rather than through the script, because the script only
    ever hands it complete behavioural spaces.
    """
    src = get_script_path("behavioral_space_explorer.py").read_text(encoding="utf-8")
    ns = {}
    # from _nearest, which hopkins_statistic depends on, not from hopkins itself
    start = src.index("def _nearest(")
    end = src.index("def hopkins_statistic_with_pvalue(")
    exec("import numpy as np\n"
         "from sklearn.preprocessing import MinMaxScaler\n" + src[start:end], ns)
    rng = np.random.RandomState(0)
    X = np.r_[rng.randn(40, 4), rng.randn(40, 4) + 6.0]
    X[rng.rand(*X.shape) < 0.2] = np.nan
    H = ns["hopkins_statistic"](X, random_state=0)
    assert np.isfinite(H), "Hopkins returned a non-finite value on gapped data"
    assert 0.0 <= H <= 1.0


def test_hopkins_still_separates_clustered_from_uniform():
    """The masked distance must not flatten the statistic it exists to measure."""
    src = get_script_path("behavioral_space_explorer.py").read_text(encoding="utf-8")
    ns = {}
    # from _nearest, which hopkins_statistic depends on, not from hopkins itself
    start = src.index("def _nearest(")
    end = src.index("def hopkins_statistic_with_pvalue(")
    exec("import numpy as np\n"
         "from sklearn.preprocessing import MinMaxScaler\n" + src[start:end], ns)
    rng = np.random.RandomState(3)
    tight = np.r_[rng.randn(60, 3) * 0.05, rng.randn(60, 3) * 0.05 + 8.0]
    spread = rng.uniform(0, 1, size=(120, 3))
    h_tight = ns["hopkins_statistic"](tight, random_state=1)
    h_spread = ns["hopkins_statistic"](spread, random_state=1)
    assert h_tight > h_spread, (h_tight, h_spread)
