"""
Reusable stage-based checkpointing for ML training pipelines.

Design
------
Each checkpoint records *which pipeline stage* just completed and persists
the artefacts produced so far.  On resume the manifest is read and all
artefact files present in the checkpoint directory are loaded back
using format-specific (no raw pickle) serializers.

Supported artefact types
────────────────────────
  • ``graph``      → joblib  (NetworkX graphs, compressed)
  • ``edges``      → .npz    (lists of (src, dst) tuples as numpy arrays)
  • ``keyed_vectors`` → gensim native .kv  (memory-mappable)
  • ``sklearn``    → joblib  (sklearn pipelines / estimators, compressed)
  • ``json``       → JSON    (dicts, metrics — human-readable, safe)

Usage
-----
    from checkpoint import CheckpointManager

    ckpt = CheckpointManager(
        stages=["load", "split", "train", "eval", "done"],
        checkpoint_dir="artifacts/checkpoints",
        registry={
            "G":          "graph",
            "train_pos":  "edges",
            "wv":         "keyed_vectors",
            "classifier": "sklearn",
            "metrics":    "json",
        },
    )

    # Save after a stage completes
    ckpt.save("load", G=G)

    # Resume
    stage, data = ckpt.load()
    if ckpt.past_stage(stage, "load"):
        G = data["G"]
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Artefact type handlers ─────────────────────────────────────────────────────
# Each handler is a (extension, save_fn, load_fn) triple.


def _save_graph(obj: Any, path: str) -> None:
    joblib.dump(obj, path, compress=3)


def _load_graph(path: str) -> Any:
    return joblib.load(path)


def _save_edges(obj: Any, path: str) -> None:
    """Save a list of (src, dst) tuples as a compressed numpy array."""
    np.savez_compressed(path, edges=np.array(obj, dtype=np.int64))


def _load_edges(path: str) -> list:
    """Load edge list back as a list of tuples."""
    arr = np.load(path)["edges"]
    return [tuple(row) for row in arr]


def _save_keyed_vectors(obj: Any, path: str) -> None:
    obj.save(path)


def _load_keyed_vectors(path: str) -> Any:
    from gensim.models import KeyedVectors
    return KeyedVectors.load(path)


def _save_sklearn(obj: Any, path: str) -> None:
    joblib.dump(obj, path, compress=3)


def _load_sklearn(path: str) -> Any:
    return joblib.load(path)


def _save_json(obj: Any, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


# type name → (extension, saver, loader)
_TYPE_HANDLERS = {
    "graph":         (".joblib", _save_graph,         _load_graph),
    "edges":         (".npz",    _save_edges,         _load_edges),
    "keyed_vectors": (".kv",     _save_keyed_vectors, _load_keyed_vectors),
    "sklearn":       (".joblib", _save_sklearn,       _load_sklearn),
    "json":          (".json",   _save_json,          _load_json),
}


class CheckpointManager:
    """
    Stage-based checkpoint manager.

    Parameters
    ----------
    stages : list[str]
        Ordered list of stage names that define the pipeline sequence.
    checkpoint_dir : str
        Directory where checkpoint files are stored.
    registry : dict[str, str]
        Mapping of artefact name → type name (one of the keys in
        ``_TYPE_HANDLERS``).  Every artefact passed to ``save()`` must
        appear here so the correct serializer is used.
    """

    def __init__(
        self,
        stages: List[str],
        checkpoint_dir: str,
        registry: Dict[str, str],
    ) -> None:
        self.stages = stages
        self.checkpoint_dir = checkpoint_dir
        self._manifest_path = os.path.join(checkpoint_dir, "manifest.json")

        # Resolve registry entries to (ext, saver, loader) triples
        self._registry: Dict[str, Tuple] = {}
        for key, type_name in registry.items():
            if type_name not in _TYPE_HANDLERS:
                raise ValueError(
                    f"Unknown artefact type '{type_name}' for key '{key}'. "
                    f"Valid types: {list(_TYPE_HANDLERS.keys())}"
                )
            self._registry[key] = _TYPE_HANDLERS[type_name]

    # ── helpers ────────────────────────────────────────────────────────────────

    def _ckpt_path(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, name)

    # ── save / load ────────────────────────────────────────────────────────────

    def save(self, stage: str, **artefacts: Any) -> None:
        """
        Persist *artefacts* and record *stage* as the last completed step.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for key, obj in artefacts.items():
            if key not in self._registry:
                raise KeyError(
                    f"Artefact '{key}' not found in checkpoint registry. "
                    f"Registered keys: {list(self._registry.keys())}"
                )
            ext, saver, _ = self._registry[key]
            path = self._ckpt_path(f"{key}{ext}")
            saver(obj, path)
            logger.info(f"  checkpoint  ✓  {key} → '{path}'")

        manifest = {"stage": stage}
        with open(self._manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info(f"  checkpoint  stage='{stage}' recorded.")

    def load(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns ``(stage, data)`` where *stage* is the last completed stage
        (or ``None``) and *data* maps artefact names to loaded objects.
        """
        if not os.path.isfile(self._manifest_path):
            return None, {}

        with open(self._manifest_path) as fh:
            manifest = json.load(fh)
        stage = manifest.get("stage")

        data: Dict[str, Any] = {}
        for key, (ext, _, loader) in self._registry.items():
            path = self._ckpt_path(f"{key}{ext}")
            if os.path.isfile(path):
                data[key] = loader(path)
                logger.info(f"  checkpoint  loaded '{key}' from '{path}'")

        logger.info(f"  checkpoint  resuming after stage='{stage}'")
        return stage, data

    # ── stage helpers ──────────────────────────────────────────────────────────

    def past_stage(
        self, completed_stage: Optional[str], target_stage: str
    ) -> bool:
        """True if *completed_stage* is at or past *target_stage*."""
        if completed_stage is None:
            return False
        return self.stages.index(completed_stage) >= self.stages.index(target_stage)
