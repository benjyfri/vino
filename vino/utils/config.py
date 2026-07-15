from pathlib import Path
from omegaconf import OmegaConf, DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_GROUPS = {"data": "dataset", "image": "image", "model": "model", "train": "train"}

def _resolve_path(value: str, relative_to: Path) -> Path:
    candidates = (REPO_ROOT / value, relative_to / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config path {value!r} relative to {relative_to}")

def _load(path: Path, seen: set[Path]) -> DictConfig:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic config inheritance involving {path}")
    seen = {*seen, path}
    cfg = OmegaConf.load(path)

    merged = OmegaConf.create({})
    root_base = cfg.pop("_base_", None)
    if root_base:
        merged = OmegaConf.merge(merged, _load(_resolve_path(str(root_base), path.parent), seen))

    defaults = cfg.pop("defaults", [])
    for item in defaults:
        if item == "_self_":
            continue
        if not OmegaConf.is_dict(item) or len(item) != 1:
            raise ValueError(f"Unsupported defaults entry {item!r} in {path}")
        group, name = next(iter(item.items()))
        group = str(group).lstrip("/")
        if group not in CONFIG_GROUPS:
            raise ValueError(f"Unknown config group {group!r} in {path}")
        subpath = REPO_ROOT / "configs" / group / f"{name}.yaml"
        merged[CONFIG_GROUPS[group]] = _load(subpath, seen)

    for key in ["dataset", "image", "model", "train"]:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, str) and value.endswith((".yaml", ".yml")):
            cfg[key] = _load(_resolve_path(value, path.parent), seen)
        elif OmegaConf.is_dict(value) and "_base_" in value:
            override = OmegaConf.create(OmegaConf.to_container(value, resolve=False))
            base = override.pop("_base_")
            cfg[key] = OmegaConf.merge(_load(_resolve_path(str(base), path.parent), seen), override)
    return OmegaConf.merge(merged, cfg)

def load_resolved_config(config_path: str | Path) -> DictConfig:
    cfg = _load(Path(config_path), set())
    for key in ("dataset", "image", "model", "train"):
        if key not in cfg:
            cfg[key] = OmegaConf.create({})
    return cfg
