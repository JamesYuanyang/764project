from .joint import train_joint
from .alt import train_alt
from .alt_temp import train_alt_temp
from .alt_temp_uw import train_alt_temp_uw

__all__ = ["train_joint", "train_alt", "train_alt_temp", "train_alt_temp_uw"]

# ✅ 动态策略映射表
strategy_map = {
    "joint": train_joint,
    "alt": train_alt,
    "alt_temp": train_alt_temp,
    "alt_temp_uw": train_alt_temp_uw,
}
