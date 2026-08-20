"""
ECML26 pipeline — clean, repo-agnostic rebuild.

Layout mirrors the paper's block structure (Structure -> Single-View -> Multi-View
-> Pointwise -> Paper tables -> Figure) but runs on the Stage-8 parallel/SQLite
motor and folds in the four improvements: tie-safe kNN purity, repeated-OOF
correctness, corrected models + label arm, parallel search + fold cache.

Phase 1 delivered here: models/, data.py, align.py, engine.py.
Still to come: structural.py (tie-safe purity + margin), search.py (job builders +
staged grid), repeated_oof.py, tables.py, figures.py, linking.py, notebook.
"""
from . import data, engine, models, search, repeated_oof, weight_profiles

__all__ = ["data", "engine", "models", "search", "repeated_oof", "weight_profiles"]
