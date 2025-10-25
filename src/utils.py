import os, yaml, random, numpy as np, torch

# ----------------- 递归合并配置 -----------------
def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "inherit_from" in cfg and cfg["inherit_from"]:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), cfg["inherit_from"]))
        base_cfg = load_yaml_config(base_path)
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "inherit_from"})
    return cfg

# ----------------- 其他通用工具 -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AvgMeter:
    def __init__(self): self.s, self.n = 0, 0
    def update(self, v, k=1): self.s += v*k; self.n += k
    @property
    def avg(self): return self.s / max(1, self.n)
