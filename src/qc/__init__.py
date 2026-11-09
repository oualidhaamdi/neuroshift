"""
Quality Control Package
"""

import math


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None, so json.dump emits valid JSON.

    Python's default json.dump writes the literal `NaN` (not in the JSON spec),
    which strict parsers (jq, browsers, many web tools) reject. We map NaN/Inf
    to None on write; the inverse rehydration is done by restore_nan_from_json.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def restore_nan_from_json(obj):
    """Recursively replace None with float('nan') inside numeric metric fields.

    Used after json.load to undo sanitize_for_json so downstream consumers
    that rely on np.isnan(...) keep working unchanged.

    Conservative scope: only converts None -> nan when the key matches a known
    numeric metric pattern. Bare None values outside these fields are left
    alone (e.g. legitimate "not applicable" entries).
    """
    NUMERIC_PREFIXES = ("epe_", "ang_", "pct_gt90_", "err_")
    NUMERIC_KEYS = {
        "mean", "median", "p95", "max", "min",
        "mean_central", "weighted_mean",
        "epe_central", "ang_pct_gt90",
        "gt_peak_mm", "pct_fold",
    }

    def _recurse(node, parent_key=None):
        if isinstance(node, dict):
            return {k: _recurse(v, parent_key=k) for k, v in node.items()}
        if isinstance(node, list):
            return [_recurse(v, parent_key=parent_key) for v in node]
        if node is None and parent_key is not None:
            if parent_key in NUMERIC_KEYS or any(parent_key.startswith(p) for p in NUMERIC_PREFIXES):
                return float("nan")
        return node

    return _recurse(obj)
