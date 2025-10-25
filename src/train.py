import os
import sys
import torch
from utils import load_yaml_config, set_seed
from dataset import load_task_loaders
from model import MultiTaskModel
from strategy import strategy_map
from strategy.probe_finetune import linear_probe_and_finetune

# ==========================================================
# 🚀 主程序入口（全自动执行五种策略 × 多种子）
# ==========================================================
def main():
    print("\n==============================")
    print(" 🎯 多任务学习全自动实验面板 ")
    print("==============================")

    # ✅ 五种策略一次性全部执行
    all_experiments = [
        ("JOINT + Proportional", "../config/config_joint.yaml", "joint", False),
        ("ALT + Balanced", "../config/config_alt.yaml", "alt", False),
        ("ALT-Temp", "../config/config_alt_temp.yaml", "alt_temp", False),
        ("ALT-Temp + UW", "../config/config_alt_temp_uw.yaml", "alt_temp_uw", True),
        ("Linear Probe + Fine-tuning", "../config/config_probe.yaml", "probe", False),
    ]

    for exp_name, cfg_path, strategy_key, use_uw in all_experiments:
        print(f"\n========================================")
        print(f"🧪 实验：{exp_name}")
        print(f"📄 配置文件：{cfg_path}")
        print(f"========================================")

        # === 读取配置文件 ===
        cfg = load_yaml_config(cfg_path)

        # === 获取种子设置 ===
        seeds = cfg.get("seeds", [cfg.get("seed", 42)])
        repeat_runs = cfg.get("training", {}).get("repeat_runs", len(seeds))

        # === 加载数据集（只加载一次）===
        print("\n📦 正在加载数据集...")
        train_loaders, val_loaders = load_task_loaders(cfg["dataset"], cfg["dataloader"])
        device = "cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu"

        # === 选择训练策略 ===
        if strategy_key == "probe":
            strategy_fn = linear_probe_and_finetune
        else:
            strategy_fn = strategy_map[strategy_key]

        # === 多种子重复实验 ===
        for i, seed in enumerate(seeds[:repeat_runs]):
            print(f"\n🌱 Run {i+1}/{repeat_runs} | Seed = {seed}")
            set_seed(seed)

            # 每次 run 都重新初始化模型
            model = MultiTaskModel(cfg["model"], use_uw=use_uw)

            # ✅ 单独命名实验输出文件
            cfg["experiment"] = {"name": f"{exp_name}_run{i+1}"}

            print(f"\n🚀 开始训练 [{exp_name}] (Run {i+1})...\n")
            strategy_fn(model, train_loaders, val_loaders, cfg, device)

        print(f"\n✅ {exp_name} 所有 runs 已完成！\n")

    print("\n🎉 全部实验已运行完毕！结果保存在 ./results 目录中。")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    # 🔧 确保可以从任意路径执行
    sys.path.append(os.path.dirname(__file__))
    main()
