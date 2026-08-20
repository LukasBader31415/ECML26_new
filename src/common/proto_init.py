import numpy as np, hashlib
import proto_dist_ml.rng as _rng

class DegenerateInitError(RuntimeError):
    pass

def _seed_for(base_seed, label, view_tag):
    key = f"{int(base_seed)}|{view_tag}|{label}".encode()
    h = int(hashlib.sha256(key).hexdigest(), 16) % (2**31 - 1)
    return int(np.random.SeedSequence([int(base_seed), h]).generate_state(1)[0])

def init_class_prototypes(overall, inClass, k, *, base_seed=42, label=None, view_tag="global"):
    """Hard-RNG prototype init, reproducing the original M3GLVQ.py extraction
    (squared sub-dissimilarity -> RNG -> nearest data point per prototype),
    with deterministic per-(seed,label,view) seeding and a no-fallback collapse check."""
    inClass = np.asarray(inClass, dtype=int)
    k = int(k)
    if inClass.size < k:
        raise DegenerateInitError(f"class {label}: {inClass.size} points < K={k}")
    D_l = np.square(overall[np.ix_(inClass, inClass)])
    np.random.seed(_seed_for(base_seed, label, view_tag))
    rng_l = _rng.RNG(k)
    rng_l.fit(D_l, is_squared=True)
    Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
    closest = np.argmin(Dp_l, axis=1)
    if np.unique(closest).size < k:
        raise DegenerateInitError(f"class {label}: prototype collapse ({np.unique(closest).size} < {k})")
    return inClass[closest]
