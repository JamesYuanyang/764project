import os, json, random, math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from utils import AvgMeter

# ==========================================================
# 📂 自动修正输出路径
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# ⚙️ 优化器构造函数
# ==========================================================
def make_optimizer_from_cfg(params, cfg):
    opt_cfg = cfg["optimizer"]
    opt_type = opt_cfg.get("type", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    wd = opt_cfg.get("weight_decay", 0.0)

    if opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.0)
        nesterov = opt_cfg.get("nesterov", False)
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    elif opt_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        return optim.Adam(params, lr=lr, weight_decay=wd)

# ==========================================================
# 📘 打印辅助函数（含 τ, log_vars, align_err）
# ==========================================================
def print_epoch_summary(strategy_name, epoch, train_loss, accs, tau_value=None, log_vars=None, align_err=None):
    avg_acc = np.mean(accs)
    print(f"\n[{strategy_name}] Epoch {epoch}")
    print(f"  → Train Loss: {train_loss:.4f}")
    print(f"  → Avg Val Accuracy (mean of 3 tasks): {avg_acc:.3f}")
    for i, acc in enumerate(accs):
        print(f"  → Task {i+1}: {acc:.3f}")

    if tau_value is not None:
        if isinstance(tau_value, dict):
            tau_list = [v[-1] if isinstance(v, list) else v for v in tau_value.values()]
            tau_str = ", ".join([f"{t:.3f}" for t in tau_list])
            print(f"  → Learnable τ: [{tau_str}]")
        elif isinstance(tau_value, (list, tuple, np.ndarray)):
            tau_str = ", ".join([f"{float(t):.3f}" for t in tau_value])
            print(f"  → Learnable τ: [{tau_str}]")
        else:
            print(f"  → Learnable τ: {float(tau_value):.3f}")

    if log_vars is not None:
        log_str = ", ".join([f"{float(v):.3f}" for v in log_vars])
        print(f"  → log_vars (Uncertainty): [{log_str}]")

    if align_err is not None:
        if isinstance(align_err, dict):
            print(f"  → Subspace Alignment Error: {align_err['align_err']:.4f}")
            print(f"  → Linear CKA Similarity: {align_err['cka']:.4f}")
        else:
            print(f"  → Subspace Alignment Error: {align_err:.4f}")

    print("------------------------------------------------------------")

    # ✅ 日志文件记录
    with open(os.path.join(RESULT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
        line = f"[{strategy_name}] Epoch {epoch} | Loss={train_loss:.4f} | AvgAcc={avg_acc:.3f} | Tasks={accs}"
        if tau_value is not None:
            line += f" | tau={tau_value}"
        if log_vars is not None:
            line += f" | log_vars={log_vars}"
        if align_err is not None:
            line += f" | align_err={align_err}"
        f.write(line + "\n")

# ==========================================================
# 🔍 验证集准确率计算
# ==========================================================
def evaluate_tasks(model, val_loaders, device):
    model.eval()
    accs = []
    for t, loader in enumerate(val_loaders):
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                logits, _ = model(xb, task_idx=t)
                preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        accs.append(correct / total)
    model.train()
    return accs

# ==========================================================
# 🧩 线性 CKA（Centered Kernel Alignment）
# ==========================================================
def compute_linear_CKA(X, Y):
    """简单实现线性 CKA，相比对齐误差更稳定"""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).mean()
    var_x = (X @ X.T).pow(2).mean().sqrt()
    var_y = (Y @ Y.T).pow(2).mean().sqrt()
    return (hsic / (var_x * var_y + 1e-8)).item()

# ==========================================================
# 🧭 子空间对齐 + CKA 计算
# ==========================================================
def compute_subspace_alignment(model, val_loaders, device, num_samples=300, rank=15):
    """计算任务间几何对齐指标：Subspace Alignment Error + Linear CKA"""
    model.eval()
    task_features = []
    with torch.no_grad():
        for t, loader in enumerate(val_loaders):
            dataset = loader.dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            xb = torch.stack([dataset[i][0] for i in indices]).to(device)
            _, rep = model(xb, task_idx=t)
            rep = rep.detach().cpu()
            rep -= rep.mean(0, keepdim=True)
            task_features.append(rep)

    align_errors, cka_scores = [], []
    for i in range(len(task_features)):
        for j in range(i + 1, len(task_features)):
            # === Subspace Alignment Error ===
            Ui, _, _ = torch.svd(task_features[i])
            Uj, _, _ = torch.svd(task_features[j])
            Ui, Uj = Ui[:, :rank], Uj[:, :rank]
            err = 1 - (torch.norm(Ui.T @ Uj, p='fro')**2) / rank
            align_errors.append(err.item())

            # === Linear CKA ===
            cka = compute_linear_CKA(task_features[i], task_features[j])
            cka_scores.append(cka)

    model.train()
    return {"align_err": float(np.mean(align_errors)), "cka": float(np.mean(cka_scores))}

# ==========================================================
# 📊 保存与绘图（含 AlignErr + CKA）
# ==========================================================
def save_metrics_and_plots(metrics, exp_name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    json_path = os.path.join(RESULT_DIR, f"{exp_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ 指标已保存至 {json_path}")

    # ========= 🔹 Loss =========
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["epoch"], metrics["train_loss"], marker='o', color='tab:red')
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title(f"{exp_name} Loss Curve"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_loss_curve.png"))
    plt.close()

    # ========= 🔹 Accuracy =========
    plt.figure(figsize=(6, 4))
    for task_name, accs in metrics["task_acc"].items():
        plt.plot(metrics["epoch"], accs, marker='s', label=task_name)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title(f"{exp_name} Task Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_task_accuracy.png"))
    plt.close()

    # ========= 🔹 τ =========
    if "tau" in metrics:
        plt.figure(figsize=(6, 4))
        tau_data = metrics["tau"]
        if isinstance(tau_data, dict):
            all_taus = []
            for task_name, tau_list in tau_data.items():
                plt.plot(metrics["epoch"], tau_list, marker='^', label=task_name)
                all_taus.append(tau_list)
            if len(all_taus) > 0:
                avg_tau = np.mean(np.array(all_taus), axis=0)
                plt.plot(metrics["epoch"], avg_tau, '--', color='black', label='Avg τ')
            plt.legend()
        else:
            plt.plot(metrics["epoch"], tau_data, marker='^', color='tab:blue')
        plt.xlabel("Epoch"); plt.ylabel("Learnable τ")
        plt.title(f"{exp_name} Temperature τ Curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_tau_curve.png"))
        plt.close()

    # ========= 🔹 log_vars =========
    if "log_vars" in metrics:
        plt.figure(figsize=(6, 4))
        logvars = np.array(metrics["log_vars"])
        for i in range(logvars.shape[1]):
            plt.plot(metrics["epoch"], logvars[:, i], marker='o', label=f"Task{i+1}")
        plt.xlabel("Epoch"); plt.ylabel("log_vars (Uncertainty)")
        plt.title(f"{exp_name} log_vars Curve")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_logvars_curve.png"))
        plt.close()

    # ========= 🔹 Alignment Error =========
    if "align_err" in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(metrics["epoch"], metrics["align_err"], marker='d', color='tab:green')
        plt.xlabel("Epoch"); plt.ylabel("Subspace Alignment Error")
        plt.title(f"{exp_name} Subspace Alignment Curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_align_curve.png"))
        plt.close()

    # ========= 🔹 Linear CKA =========
    if "cka" in metrics and len(metrics["cka"]) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(metrics["epoch"], metrics["cka"], marker='*', color='tab:purple')
        plt.xlabel("Epoch")
        plt.ylabel("Linear CKA Similarity")
        plt.title(f"{exp_name} Linear CKA Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_cka_curve.png"))
        plt.close()
