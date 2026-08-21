import numpy as np, hashlib, json, os
import proto_dist_ml.rng as _rng

class DegenerateInitError(RuntimeError):
    pass

# --- configuration (set from the notebook before running) --------------------
# Number of ALTERNATIVE seeds tried after the canonical one when the canonical
# init collapses (Option A: rescue seed-unlucky collapses without changing the
# init method). Set to 0 to restore the strict single-seed behaviour.
MAX_INIT_RETRIES = 8
# If set to a path, every rescue (canonical seed failed, a retry seed succeeded)
# is appended as one JSON line -> your reference of where the normal init failed.
RESCUE_LOG_PATH = None


def _seed_for(base_seed, label, view_tag):
    key = f"{int(base_seed)}|{view_tag}|{label}".encode()
    h = int(hashlib.sha256(key).hexdigest(), 16) % (2**31 - 1)
    return int(np.random.SeedSequence([int(base_seed), h]).generate_state(1)[0])


def _record_rescue(rec):
    path = RESCUE_LOG_PATH
    if not path:
        return
    try:
        # POSIX append of a short line is atomic enough for a diagnostic log,
        # also across the loky worker processes.
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _proto_candidates(D_l, k, seed):
    """One RNG init attempt under a given seed. Returns the closest-data-point
    indices and whether they are k distinct points."""
    np.random.seed(seed)
    rng_l = _rng.RNG(k)
    rng_l.fit(D_l, is_squared=True)
    Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
    closest = np.argmin(Dp_l, axis=1)
    return closest, (np.unique(closest).size >= k)


def init_class_prototypes(overall, inClass, k, *, base_seed=42, label=None, view_tag="global"):
    """Hard-RNG prototype init (squared sub-dissimilarity -> RNG -> nearest data
    point per prototype), deterministic per (seed, label, view).

    Option A: the canonical seed is tried FIRST (so successful inits are
    bit-identical to the strict version). Only if it collapses do we try up to
    MAX_INIT_RETRIES alternative deterministic seeds; the first that yields k
    distinct prototypes wins, and the rescue is logged. If every seed collapses,
    the collapse is fundamental and DegenerateInitError is raised.
    """
    inClass = np.asarray(inClass, dtype=int)
    k = int(k)
    if inClass.size < k:
        raise DegenerateInitError(f"class {label}: {inClass.size} points < K={k}")
    D_l = np.square(overall[np.ix_(inClass, inClass)])

    seeds = [_seed_for(base_seed, label, view_tag)]
    for a in range(1, int(MAX_INIT_RETRIES) + 1):
        seeds.append(_seed_for(base_seed, label, f"{view_tag}#retry{a}"))

    last_unique = None
    for attempt, s in enumerate(seeds):
        closest, ok = _proto_candidates(D_l, k, s)
        if ok:
            if attempt > 0:
                _record_rescue({
                    "label": (int(label) if label is not None else None),
                    "view_tag": view_tag, "k": k, "base_seed": int(base_seed),
                    "canonical_failed": True, "rescued_attempt": attempt,
                    "n_class_points": int(inClass.size),
                })
            return inClass[closest]
        last_unique = int(np.unique(closest).size)

    raise DegenerateInitError(
        f"class {label}: prototype collapse ({last_unique} < {k}) on view "
        f"'{view_tag}' after {len(seeds)} seeds — fundamental (not seed-luck)."
    )


def read_rescue_log(path):
    """Load the rescue log into a DataFrame: one row per (config) where the
    canonical init failed but a retry seed succeeded."""
    import pandas as pd
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["label", "view_tag", "k", "base_seed",
                                     "canonical_failed", "rescued_attempt", "n_class_points"])
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(rows)