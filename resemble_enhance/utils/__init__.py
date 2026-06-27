from .distributed import global_leader_only
from .logging import setup_logging
from .utils import save_mels, tree_map

__all__ = [
    "Engine",
    "TrainLoop",
    "gather_attribute",
    "global_leader_only",
    "is_global_leader",
    "save_mels",
    "setup_logging",
    "tree_map",
]


def __getattr__(name):
    if name in {"Engine", "gather_attribute"}:
        from .engine import Engine, gather_attribute

        return {"Engine": Engine, "gather_attribute": gather_attribute}[name]

    if name in {"TrainLoop", "is_global_leader"}:
        from .train_loop import TrainLoop, is_global_leader

        return {"TrainLoop": TrainLoop, "is_global_leader": is_global_leader}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
