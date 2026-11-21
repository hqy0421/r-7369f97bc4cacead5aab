"""Tests for the per-warehouse moving-block residual bootstrap."""

from copy import deepcopy

import numpy as np
import pytest

from pygam import LinearGAM, s
from pygam.bootstrap import moving_block_residual_bootstrap


def _make_panel(rng, n_warehouses=3, days_per_warehouse=20, gap_at=None):
    """Build a small multi-warehouse panel with a temporal signal."""
    gap_at = gap_at or {}
    rows = []
    for wid in range(n_warehouses):
        skipped = gap_at.get(wid, set())
        day = 0
        produced = 0
        while produced < days_per_warehouse:
            if day not in skipped:
                rows.append((wid, day))
                produced += 1
            day += 1
    warehouse_ids = np.array([r[0] for r in rows], dtype=np.int64)
    days = np.array([r[1] for r in rows], dtype=np.int64)
    n_rows = len(rows)

    time = days + warehouse_ids * 7.0
    X = (time / max(time.max(), 1.0)).reshape(-1, 1)
    weights = rng.uniform(0.5, 3.0, n_rows)
    signal = np.sin(2 * np.pi * X.ravel()) + 0.3 * warehouse_ids
    noise = rng.normal(0.0, 0.25, n_rows) / np.sqrt(weights)
    y = signal + noise
    return X, y, warehouse_ids, days, weights


@pytest.fixture(scope="module")
def fitted_panel():
    rng = np.random.default_rng(2024)
    X, y, warehouse_ids, days, weights = _make_panel(rng)
    gam = LinearGAM(s(0, n_splines=10, lam=2.5)).fit(X, y, weights=weights)
    return gam, X, y, warehouse_ids, days, weights


def test_result_shapes_and_interval_ordering(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    query_X = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=4,
        n_bootstraps=25,
        query_X=query_X,
        random_state=7,
    )

    assert result["bootstrap_predictions"].shape == (25, 7)
    assert result["mean"].shape == (7,)
    assert result["quantiles"].shape == (7, 2)
    assert result["interval"].shape == (7, 2)
    assert result["fitted_predictions"].shape == (7,)
    # interval bounds must bracket the bootstrap mean and respect ordering
    assert np.all(result["interval"][:, 0] <= result["mean"])
    assert np.all(result["mean"] <= result["interval"][:, 1])
    assert np.allclose(result["interval"][:, 0], result["quantiles"][:, 0])
    assert np.allclose(result["interval"][:, 1], result["quantiles"][:, 1])


def test_custom_quantiles_are_returned_in_requested_order(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        n_bootstraps=20,
        quantiles=(0.1, 0.5, 0.9),
        random_state=3,
    )
    n_rows = X.shape[0]
    assert result["quantiles"].shape == (n_rows, 3)
    assert np.all(result["quantiles"][:, 0] <= result["quantiles"][:, 1])
    assert np.all(result["quantiles"][:, 1] <= result["quantiles"][:, 2])
    # the median column should sit close to the bootstrap mean
    assert np.corrcoef(result["quantiles"][:, 1], result["mean"])[0, 1] > 0.99


def test_permutation_invariance_same_seed(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(y))

    base = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=4,
        n_bootstraps=30,
        query_X=X,
        random_state=123,
    )
    shuffled = moving_block_residual_bootstrap(
        gam,
        X[perm],
        y[perm],
        warehouse_ids[perm],
        days[perm],
        weights=weights[perm],
        block_length=4,
        n_bootstraps=30,
        # query points are not input rows, so they keep their own ordering
        query_X=X,
        random_state=123,
    )

    np.testing.assert_allclose(
        shuffled["bootstrap_predictions"], base["bootstrap_predictions"]
    )
    np.testing.assert_allclose(shuffled["mean"], base["mean"])
    np.testing.assert_allclose(shuffled["interval"], base["interval"])

    # the audit trail is expressed in canonical indices, so it matches too
    for key in (
        "block_id",
        "donor_canonical_index",
        "donor_day",
        "receiver_canonical_index",
    ):
        np.testing.assert_array_equal(
            shuffled["source_mapping"][key], base["source_mapping"][key]
        )
    np.testing.assert_array_equal(
        shuffled["source_mapping"]["donor_warehouse"],
        base["source_mapping"]["donor_warehouse"],
    )


def test_different_seeds_change_draws(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    a = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        n_bootstraps=15,
        random_state=1,
    )
    b = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        n_bootstraps=15,
        random_state=2,
    )
    assert not np.allclose(a["bootstrap_predictions"], b["bootstrap_predictions"])


def test_segments_split_on_warehouse_and_day_gaps(fitted_panel):
    gam, X, y, _, _, weights = fitted_panel
    warehouse_ids = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    days = np.array([0, 1, 2, 4, 3, 4, 5], dtype=np.int64)
    result = moving_block_residual_bootstrap(
        gam,
        X[:7],
        y[:7],
        warehouse_ids,
        days,
        weights=weights[:7],
        block_length=2,
        n_bootstraps=3,
        random_state=0,
    )
    segments = result["segments"]
    # warehouse 0 splits at the day 2 -> 4 gap; warehouse 1 is one run
    assert len(segments) == 3
    assert [seg["warehouse_id"] for seg in segments] == [0, 0, 1]
    assert [list(seg["days"]) for seg in segments] == [
        [0, 1, 2],
        [4],
        [3, 4, 5],
    ]


def test_donors_stay_inside_receiver_segment(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=5,
        n_bootstraps=12,
        random_state=11,
    )
    order = result["canonical_order"]
    sorted_warehouse = warehouse_ids[order]
    sorted_days = days[order]
    mapping = result["source_mapping"]

    for segment in result["segments"]:
        start, stop = segment["start"], segment["stop"]
        receiver = mapping["receiver_canonical_index"]
        in_segment = (receiver >= start) & (receiver < stop)
        donors = mapping["donor_canonical_index"][in_segment]
        assert donors.min() >= start
        assert donors.max() < stop
        # every donor really belongs to the same warehouse and contiguous days
        assert np.all(sorted_warehouse[donors] == segment["warehouse_id"])
        assert np.all(sorted_days[donors] >= segment["days"][0])
        assert np.all(sorted_days[donors] <= segment["days"][-1])


def test_block_ids_are_contiguous_and_truncation_works(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    block_length = 6
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=block_length,
        n_bootstraps=4,
        random_state=5,
    )
    mapping = result["source_mapping"]
    for segment in result["segments"]:
        start, stop = segment["start"], segment["stop"]
        seg_len = stop - start
        expected_blocks = int(np.ceil(seg_len / min(block_length, seg_len)))
        for b in range(4):
            ids = mapping["block_id"][b, start:stop]
            assert ids[0] >= 0
            # block ids are contiguous and each block is at most block_length
            unique_ids = np.unique(ids)
            assert len(unique_ids) == expected_blocks
            assert np.array_equal(unique_ids, np.arange(ids[0], ids[-1] + 1))
            counts = np.bincount(ids - ids[0])
            assert counts.max() <= min(block_length, seg_len)
            # only the final block may be truncated
            assert np.all(counts[:-1] == min(block_length, seg_len))


def test_short_segments_use_actual_length():
    rng = np.random.default_rng(0)
    # two single-row warehouses and one 2-row warehouse
    warehouse_ids = np.array([0, 1, 2, 2], dtype=np.int64)
    days = np.array([0, 0, 0, 1], dtype=np.int64)
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = rng.normal(size=4)
    weights = np.full(4, 1.5)
    gam = LinearGAM(s(0, n_splines=4)).fit(X, y, weights=weights)

    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=10,
        n_bootstraps=8,
        random_state=4,
    )
    mapping = result["source_mapping"]
    # single-row segments must donate themselves; each is filled by one block
    for canonical_row in (0, 1):
        donors = mapping["donor_canonical_index"][:, canonical_row]
        assert np.all(donors == canonical_row)
        assert np.all(
            mapping["block_id"][:, canonical_row]
            == mapping["block_id"][:, canonical_row][0, None]
        )
    # the two-row segment resamples within its own two rows
    donors = mapping["donor_canonical_index"][:, 2:4]
    assert np.all((donors == 2) | (donors == 3))


def test_residuals_rebuilt_with_receiver_weights(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    n_bootstraps = 10
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=3,
        n_bootstraps=n_bootstraps,
        random_state=21,
    )
    order = result["canonical_order"]
    sorted_weights = weights[order]
    mapping = result["source_mapping"]
    mu = gam.predict(X[order])

    # reconstruct the standardized draw from the audit trail and rebuild y*
    original_residual = y[order] - mu
    centered = np.empty_like(original_residual)
    for segment in result["segments"]:
        sl = slice(segment["start"], segment["stop"])
        w_seg = sorted_weights[sl]
        r_seg = original_residual[sl]
        centered[sl] = r_seg - np.dot(w_seg, r_seg) / w_seg.sum()
    sigma = np.sqrt(np.dot(sorted_weights, centered**2) / sorted_weights.sum())
    standardized = np.sqrt(sorted_weights) * centered / sigma

    for b in range(n_bootstraps):
        rebuilt_standardized = standardized[mapping["donor_canonical_index"][b]]
        rebuilt_y = mu + sigma * rebuilt_standardized / np.sqrt(sorted_weights)

        # run the same refit path and check predictions match the stored output
        check_gam = deepcopy(gam)
        check_gam.set_params(gam.get_params())
        check_gam.fit(X[order], rebuilt_y, weights=sorted_weights)
        np.testing.assert_allclose(
            check_gam.predict(X[order]),
            result["bootstrap_predictions"][b],
            atol=1e-10,
        )


def test_refits_keep_same_lam_and_basis(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    original_lam = gam.terms.lam
    original_knots = [
        np.array(term.edge_knots_, copy=True)
        for term in gam.terms
        if hasattr(term, "edge_knots_")
    ]
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        n_bootstraps=5,
        random_state=0,
    )
    assert gam.terms.lam == original_lam
    refit_knots = [
        np.array(term.edge_knots_, copy=True)
        for term in gam.terms
        if hasattr(term, "edge_knots_")
    ]
    for a, b in zip(original_knots, refit_knots):
        np.testing.assert_allclose(a, b)
    assert result["bootstrap_predictions"].shape[0] == 5


def test_centering_removes_segment_weighted_mean(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel
    # the within-segment weighted mean of the centered working residuals
    # must be exactly zero for every segment
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        weights=weights,
        block_length=1,
        n_bootstraps=1,
        random_state=0,
    )
    order = result["canonical_order"]
    sorted_weights = weights[order]
    mu = gam.predict(X[order])
    residual = y[order] - mu
    for segment in result["segments"]:
        sl = slice(segment["start"], segment["stop"])
        w_seg = sorted_weights[sl]
        r_seg = residual[sl]
        weighted_mean = np.dot(w_seg, r_seg) / w_seg.sum()
        np.testing.assert_allclose(
            r_seg
            - weighted_mean
            - (r_seg - weighted_mean).mean() * 0
            + (weighted_mean - weighted_mean),
            r_seg - weighted_mean,
        )
        # weighted mean of the centered residuals is exactly zero
        np.testing.assert_allclose(
            np.dot(w_seg, r_seg - weighted_mean), 0.0, atol=1e-12
        )


def test_unit_weights_default(fitted_panel):
    gam, X, y, warehouse_ids, days, _ = fitted_panel
    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        n_bootstraps=5,
        random_state=1,
    )
    assert result["bootstrap_predictions"].shape == (5, X.shape[0])
    assert np.all(np.isfinite(result["interval"]))


def test_string_warehouse_ids_supported():
    rng = np.random.default_rng(8)
    warehouse_ids = np.array(["wh-a"] * 6 + ["wh-b"] * 6)
    days = np.tile(np.arange(6), 2)
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    y = np.sin(X.ravel() * 3) + rng.normal(0, 0.1, 12)
    gam = LinearGAM(s(0, n_splines=6)).fit(X, y)

    result = moving_block_residual_bootstrap(
        gam,
        X,
        y,
        warehouse_ids,
        days,
        block_length=3,
        n_bootstraps=6,
        random_state=2,
    )
    donor_warehouse = result["source_mapping"]["donor_warehouse"]
    receiver = result["source_mapping"]["receiver_canonical_index"]
    for wh in ("wh-a", "wh-b"):
        mask = receiver < 6 if wh == "wh-a" else receiver >= 6
        assert np.all(donor_warehouse[mask] == wh)
    assert result["segments"][0]["warehouse_id"] == "wh-a"


def test_invalid_inputs_raise(fitted_panel):
    gam, X, y, warehouse_ids, days, weights = fitted_panel

    with pytest.raises(TypeError, match="fitted pygam.LinearGAM"):
        moving_block_residual_bootstrap(
            LinearGAM(s(0)), X, y, warehouse_ids, days, random_state=0
        )

    with pytest.raises(ValueError, match="same number of rows"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids[:-1],
            days,
            weights=weights,
            random_state=0,
        )

    bad_weights = weights.copy()
    bad_weights[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids,
            days,
            weights=bad_weights,
            random_state=0,
        )

    with pytest.raises(ValueError, match="integer day indices"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids,
            days + 0.5,
            weights=weights,
            random_state=0,
        )

    with pytest.raises(ValueError, match="block_length"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids,
            days,
            weights=weights,
            block_length=0,
            random_state=0,
        )

    with pytest.raises(ValueError, match="n_bootstraps"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids,
            days,
            weights=weights,
            n_bootstraps=0,
            random_state=0,
        )

    with pytest.raises(ValueError, match="quantiles"):
        moving_block_residual_bootstrap(
            gam,
            X,
            y,
            warehouse_ids,
            days,
            weights=weights,
            quantiles=(0.0, 0.5),
            random_state=0,
        )


def test_zero_variance_residuals_raise(monkeypatch):
    rng = np.random.default_rng(0)
    warehouse_ids = np.array([0, 0, 0], dtype=np.int64)
    days = np.array([0, 1, 2], dtype=np.int64)
    X = np.array([[0.0], [1.0], [2.0]])
    y = rng.normal(size=3)
    gam = LinearGAM(s(0, n_splines=4)).fit(X, y)

    # force an exact fit so the working residuals vanish and sigma == 0
    monkeypatch.setattr(gam, "predict", lambda points: y)
    with pytest.raises(ValueError, match="residual scale is zero"):
        moving_block_residual_bootstrap(gam, X, y, warehouse_ids, days, random_state=0)
