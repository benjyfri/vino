import yaml
from omegaconf import OmegaConf, DictConfig
import json

def load_config(path: str) -> DictConfig:
    conf = OmegaConf.load(path)
    return conf

def save_json(data, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_jsonl(data_list, path: str):
    with open(path, 'a') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')
