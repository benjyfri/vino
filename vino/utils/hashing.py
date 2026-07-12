import hashlib
import json

def hash_config(config_dict: dict) -> str:
    """Creates a deterministic hash of a config dict."""
    encoded = json.dumps(config_dict, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:8]
