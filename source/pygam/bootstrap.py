"""Per-warehouse moving-block residual bootstrap for fitted LinearGAMs.

Residuals of a daily fulfillment-time model often stay on the same side of
zero for several consecutive days within a warehouse, and independent row
resampling would draw from a distribution that is too narrow.  This module
implements a *moving-block residual bootstrap* that keeps the temporal
dependence inside each warehouse intact.

Pipeline
--------
Rows carry a warehouse id, an integer day index and a fit weight.  Within each
warehouse, rows are split into maximal contiguous runs of adjacent days
(``day[t + 1] == day[t] + 1``); blocks never cross warehouses or gaps.

For precision weights ``w`` (the convention pygam uses in its weighted
least-squares fit), the working residual ``r = y - mu`` is converted to a
weighted Pearson residual

    z = sqrt(w) * (r - r_bar_seg) / sigma,

where ``r_bar_seg`` is the weighted mean of the residuals inside the segment
and ``sigma`` is the pooled weighted residual scale
``sqrt(sum(w * (r - r_bar)^2) / sum(w))`` (equal to the fitted LinearGAM's
estimated scale).  Centering is done per segment, pooling of the scale is
global.

Blocks of length ``block_length`` are drawn *with replacement* from the
standardized residuals of the same segment until the segment is filled; the
final block is truncated.  Segments shorter than ``block_length`` use their
actual length, which makes every single-row segment an ordinary (within
segment) residual draw.  Each drawn residual is then restored on the scale of
the receiving row with its own weight,

    r* = sigma * z* / sqrt(w_receiver),

and added back to the original fitted mean: ``y* = mu + r*``.  A fresh
LinearGAM with the identical spline basis and smoothing parameters is
refit to ``(X, y*, weights=w)`` for every bootstrap replicate.

All randomness is consumed after sorting rows into a canonical
``(warehouse_id, day)`` order, so shuffling the input rows does not change
the result when the same seed is used.  Every drawn block is reported with the
warehouse/day it came from, which makes the resampling fully auditable.
"""

from copy import deepcopy

import numpy as np

from pygam.pygam import LinearGAM

__all__ = ["moving_block_residual_bootstrap"]


def _canonical_order(warehouse_ids, days):
    """Return indices that sort rows into canonical (warehouse, day) order.

    Sorting on a structured array gives a deterministic total order for any
    warehouse-id dtype (strings, integers, ...) and keeps days ordered inside
    each warehouse.
    """
    key = np.rec.fromarrays(
        [np.asarray(warehouse_ids), np.asarray(days)], names="warehouse,day"
    )
    return np.argsort(key, kind="stable")


def _build_segments(warehouse_ids, days):
    """Split canonical rows into contiguous within-warehouse day runs.

    Parameters
    ----------
    warehouse_ids : np.ndarray of shape (n_samples,)
        Warehouse identifiers in canonical order.
    days : np.ndarray of shape (n_samples,)
        Integer day indices in canonical order.

    Returns
    -------
    segments : list of dict
        Each dictionary describes one segment with keys ``warehouse_id``,
        ``start``, ``stop`` and ``days`` (positions are indices into the
        canonical arrays).
    """
    segments = []
    n_rows = len(days)
    if n_rows == 0:
        return segments

    start = 0
    for pos in range(1, n_rows):
        new_warehouse = warehouse_ids[pos] != warehouse_ids[pos - 1]
        gap = days[pos] != days[pos - 1] + 1
        if new_warehouse or gap:
            segments.append(
                {
                    "warehouse_id": warehouse_ids[pos - 1],
                    "start": start,
                    "stop": pos,
                    "days": days[start:pos],
                }
            )
            start = pos
    segments.append(
        {
            "warehouse_id": warehouse_ids[n_rows - 1],
            "start": start,
            "stop": n_rows,
            "days": days[start:n_rows],
        }
    )
    return segments


def moving_block_residual_bootstrap(
    gam,
    X,
    y,
    warehouse_ids,
    days,
    weights=None,
    block_length=5,
    n_bootstraps=100,
    query_X=None,
    quantiles=(0.025, 0.975),
    random_state=None,
):
    """Per-warehouse moving-block residual bootstrap for a fitted LinearGAM.

    Parameters
    ----------
    gam : LinearGAM
        An already fitted ``LinearGAM``.  Its spline basis (knots, term
        specification) and smoothing parameters ``lam`` are reused unchanged
        for every bootstrap refit.
    X : array-like of shape (n_samples, m_features)
        Training features used to fit ``gam``.
    y : array-like of shape (n_samples,)
        Observed responses used to fit ``gam``.
    warehouse_ids : array-like of shape (n_samples,)
        Warehouse identifier of each row.
    days : array-like of shape (n_samples,)
        Integer day index of each row.  Adjacent integer values define
        contiguous segments within a warehouse.
    weights : array-like of shape (n_samples,) or None, optional
        Fit weights of each row (precision weights).  If ``None``, unit
        weights are used.
    block_length : int, optional
        Nominal moving-block length.  Segments shorter than this use their
        actual length; the last block filling a segment may be truncated.
    n_bootstraps : int, optional
        Number of bootstrap replicates (and refits) to run.
    query_X : array-like of shape (n_query, m_features) or None, optional
        New query points at which to collect predictions.  If ``None``,
        predictions are collected at the training rows ``X``.
    quantiles : tuple of floats, optional
        Quantiles of the bootstrap prediction distribution to report for
        every query point.  Defaults to ``(0.025, 0.975)``, i.e. a 95%
        interval.
    random_state : int, numpy.random.Generator or None, optional
        Seed or generator controlling all resampling.  Results are identical
        under input-row permutation for the same value.

    Returns
    -------
    result : dict
        - ``mean`` : np.ndarray of shape (n_query,), bootstrap mean
          prediction at each query point.
        - ``quantiles`` : np.ndarray of shape (n_query, len(quantiles)),
          bootstrap quantile predictions, one column per requested quantile.
        - ``interval`` : np.ndarray of shape (n_query, 2), lower and upper
          prediction interval (first and last requested quantile).
        - ``bootstrap_predictions`` : np.ndarray of shape
          (n_bootstraps, n_query), every replicate's prediction.
        - ``fitted_predictions`` : np.ndarray of shape (n_query,),
          predictions of the original fitted model at the query points.
        - ``segments`` : list of dict, the contiguous segments found in the
          data; each entry contains ``warehouse_id``, ``start``, ``stop`` and
          ``days``, with positions indexing the canonical row order.
        - ``canonical_order`` : np.ndarray of shape (n_samples,), indices that
          sort the input rows into canonical ``(warehouse_id, day)`` order.
        - ``source_mapping`` : dict with the audit trail of every drawn block:

          - ``receiver_canonical_index`` : np.ndarray of shape
            (n_bootstraps, n_samples), canonical index of each receiving row.
          - ``block_id`` : np.ndarray of shape (n_bootstraps, n_samples),
            id of the block filling each receiving position (ids reset to 0
            per bootstrap replicate).
          - ``donor_canonical_index`` : np.ndarray of shape
            (n_bootstraps, n_samples), canonical index of the donor row whose
            standardized residual was placed at each receiving row.
          - ``donor_warehouse`` : object np.ndarray of shape
            (n_bootstraps, n_samples), warehouse id of each donor.
          - ``donor_day`` : np.ndarray of shape (n_bootstraps, n_samples),
            integer day of each donor.

    Raises
    ------
    TypeError
        If ``gam`` is not a fitted LinearGAM.
    ValueError
        If inputs have inconsistent lengths, weights are non-positive, days
        are not integer-valued, ``block_length`` / ``n_bootstraps`` /
        ``quantiles`` are invalid, or the residual scale is zero.
    """
    if not isinstance(gam, LinearGAM) or not gam._is_fitted:
        raise TypeError(
            "gam must be an already fitted pygam.LinearGAM instance; "
            f"got {type(gam).__name__}"
        )

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    warehouse_ids = np.asarray(warehouse_ids)
    days = np.asarray(days)
    n_rows = len(y)

    if weights is None:
        weights = np.ones(n_rows, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).ravel()

    if not (X.shape[0] == len(weights) == len(warehouse_ids) == len(days) == n_rows):
        raise ValueError(
            "X, y, warehouse_ids, days and weights must all have the same "
            f"number of rows; got {X.shape[0]}, {n_rows}, "
            f"{len(warehouse_ids)}, {len(days)} and {len(weights)}"
        )
    if np.any(weights <= 0):
        raise ValueError("all fit weights must be strictly positive")
    if not np.issubdtype(days.dtype, np.integer):
        if np.all(days == np.round(days)):
            days = days.astype(np.int64)
        else:
            raise ValueError("days must contain integer day indices")
    block_length = int(block_length)
    if block_length < 1:
        raise ValueError("block_length must be a positive integer")
    n_bootstraps = int(n_bootstraps)
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be a positive integer")

    quantiles = np.asarray(quantiles, dtype=float).ravel()
    if quantiles.size < 1 or np.any(quantiles <= 0) or np.any(quantiles >= 1):
        raise ValueError("quantiles must all lie strictly inside (0, 1)")

    rng = np.random.default_rng(random_state)

    if query_X is None:
        query_X = X
    else:
        query_X = np.asarray(query_X, dtype=float)
        if query_X.ndim == 1:
            query_X = query_X[:, None]

    # canonical order: all RNG consumption below is independent of input order
    order = _canonical_order(warehouse_ids, days)
    warehouse_sorted = warehouse_ids[order]
    days_sorted = days[order]
    weights_sorted = weights[order]
    X_sorted = X[order]
    mu_sorted = gam.predict(X_sorted)
    y_sorted = y[order]
    residuals = y_sorted - mu_sorted

    segments = _build_segments(warehouse_sorted, days_sorted)

    # weighted, per-segment centering of the working residuals
    centered = np.empty(n_rows, dtype=float)
    for segment in segments:
        sl = slice(segment["start"], segment["stop"])
        seg_weights = weights_sorted[sl]
        seg_residuals = residuals[sl]
        weighted_mean = np.dot(seg_weights, seg_residuals) / seg_weights.sum()
        centered[sl] = seg_residuals - weighted_mean

    # pooled weighted Pearson scale, matching the LinearGAM scale estimate
    sigma = np.sqrt(np.dot(weights_sorted, centered**2) / weights_sorted.sum())
    if not np.isfinite(sigma) or sigma == 0.0:
        raise ValueError(
            "residual scale is zero (or non-finite); the bootstrap needs "
            "residual variation to resample"
        )
    standardized = np.sqrt(weights_sorted) * centered / sigma

    # audit-trail arrays, indexed in canonical row order
    receiver_index = np.tile(np.arange(n_rows), n_bootstraps).reshape(
        n_bootstraps, n_rows
    )
    block_ids = np.empty((n_bootstraps, n_rows), dtype=np.int64)
    donor_index = np.empty((n_bootstraps, n_rows), dtype=np.int64)
    donor_day = np.empty((n_bootstraps, n_rows), dtype=days_sorted.dtype)
    donor_warehouse = np.empty((n_bootstraps, n_rows), dtype=warehouse_sorted.dtype)
    standardized_draws = np.empty((n_bootstraps, n_rows), dtype=float)

    for b in range(n_bootstraps):
        next_block_id = 0
        for segment in segments:
            start, stop = segment["start"], segment["stop"]
            seg_len = stop - start
            draw_len = min(block_length, seg_len)
            n_starts = int(np.ceil(seg_len / draw_len))
            # with-replacement block starts within this same segment
            starts = rng.integers(0, seg_len - draw_len + 1, size=n_starts)

            fill_pos = 0
            for local_start in starts:
                take = min(draw_len, seg_len - fill_pos)
                dst = start + fill_pos
                src = start + local_start
                donor_positions = np.arange(src, src + take)
                receiver_positions = np.arange(dst, dst + take)

                block_ids[b, receiver_positions] = next_block_id
                donor_index[b, receiver_positions] = donor_positions
                donor_day[b, receiver_positions] = days_sorted[donor_positions]
                donor_warehouse[b, receiver_positions] = warehouse_sorted[
                    donor_positions
                ]
                standardized_draws[b, receiver_positions] = standardized[
                    donor_positions
                ]
                fill_pos += take
                next_block_id += 1

    # restore the residual scale with the *receiving* row's weight and
    # rebuild bootstrap responses on top of the original fitted mean
    bootstrap_y_canonical = mu_sorted[None, :] + (
        sigma * standardized_draws / np.sqrt(weights_sorted)[None, :]
    )

    n_query = query_X.shape[0]
    bootstrap_predictions = np.empty((n_bootstraps, n_query), dtype=float)
    original_lam = deepcopy(gam.terms.lam)
    for b in range(n_bootstraps):
        bootstrap_gam = deepcopy(gam)
        bootstrap_gam.set_params(gam.get_params())
        bootstrap_gam.fit(X_sorted, bootstrap_y_canonical[b], weights=weights_sorted)
        if bootstrap_gam.terms.lam != original_lam:
            raise RuntimeError(
                "bootstrap refit changed the smoothing parameter; the same "
                "lam as the original fit must be preserved"
            )
        bootstrap_predictions[b] = bootstrap_gam.predict(query_X)

    mean_prediction = bootstrap_predictions.mean(axis=0)
    quantile_predictions = np.quantile(bootstrap_predictions, quantiles, axis=0).T
    interval = np.column_stack(
        [quantile_predictions[:, 0], quantile_predictions[:, -1]]
    )

    return {
        "mean": mean_prediction,
        "quantiles": quantile_predictions,
        "interval": interval,
        "bootstrap_predictions": bootstrap_predictions,
        "fitted_predictions": gam.predict(query_X),
        "segments": segments,
        "canonical_order": order,
        "source_mapping": {
            "receiver_canonical_index": receiver_index,
            "block_id": block_ids,
            "donor_canonical_index": donor_index,
            "donor_warehouse": donor_warehouse,
            "donor_day": donor_day,
        },
    }
