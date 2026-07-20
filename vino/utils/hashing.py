import hashlib
import json
from omegaconf import OmegaConf

def hash_config(config_dict: dict) -> str:
    """Creates a deterministic hash of a config dict."""
    if OmegaConf.is_config(config_dict):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)
    encoded = json.dumps(config_dict, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:8]

def hash_preprocessing_config(config_dict: dict) -> str:
    """Hash only inputs that determine cached graph-image contents.

    ``cache_schema`` is bumped to 4 with the semantic invariant sign features. The fingerprint
    additionally folds in an explicit canonicalization signature (ordered sign pipeline, sign
    rule, skew/projection/lex tolerances, and node/edge feature-construction versions) so that
    changing *any* behavior that alters canonicalization -- even settings not stored in the image
    config, like tolerances or feature versions -- produces a distinct cache identity. Old and new
    canonicalizations can never silently share a cache directory.
    """
    from vino.transforms.fiedler_sign import DEFAULT_PIPELINE, canonicalization_signature

    if OmegaConf.is_config(config_dict):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)
    image = config_dict.get("image", {}) or {}
    canonical = image.get("canonicalization", {}) if isinstance(image, dict) else {}
    sign_rule = canonical.get("sign_rule", "fiedler_cascade")
    sign_pipeline = canonical.get("sign_pipeline") or DEFAULT_PIPELINE
    try:
        signature = canonicalization_signature(sign_pipeline, sign_rule)
    except ValueError:
        # An invalid pipeline still yields a stable (distinct) fingerprint rather than crashing
        # the hash; config validation reports the real error elsewhere.
        signature = {"sign_rule": sign_rule, "sign_pipeline": list(sign_pipeline), "invalid": True}
    return hash_config({
        "cache_schema": 4,
        "canonicalization_signature": signature,
        **{key: config_dict.get(key, {}) for key in ("dataset", "image")},
    })
