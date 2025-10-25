import os, json, random, math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from utils import AvgMeter
'''
# ==========================================================
# 📂 自动修正输出路径
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# ⚙️ 优化器构造函数
# ==========================================================
def _make_optimizer_from_cfg(params, cfg):
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
    print(f"  → Task 1 (Cat 🐱 vs Dog 🐶):       {accs[0]:.3f}")
    print(f"  → Task 2 (Bird 🐦 vs Frog 🐸):     {accs[1]:.3f}")
    print(f"  → Task 3 (Airplane ✈️ vs Ship 🚢): {accs[2]:.3f}")

    if tau_value is not None:
        if isinstance(tau_value, dict):
            tau_list = [v[-1] if isinstance(v, list) else v for v in tau_value.values()]
            tau_str = ", ".join([f"{t:.3f}" for t in tau_list])
            print(f"  → Learnable τ (per-task): [{tau_str}]")
        elif isinstance(tau_value, (list, tuple, np.ndarray)):
            tau_str = ", ".join([f"{float(t):.3f}" for t in tau_value])
            print(f"  → Learnable τ (per-task): [{tau_str}]")
        else:
            print(f"  → Learnable τ (Temperature): {float(tau_value):.3f}")

    if log_vars is not None:
        log_str = ", ".join([f"{float(v):.3f}" for v in log_vars])
        print(f"  → log_vars (Uncertainty): [{log_str}]")

    if align_err is not None:
        print(f"  → Subspace Alignment Error: {align_err:.4f}")

    print("------------------------------------------------------------")

    # 日志文件记录
    with open(os.path.join(RESULT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
        line = f"[{strategy_name}] Epoch {epoch} | Loss={train_loss:.4f} | AvgAcc={avg_acc:.3f} | Tasks={accs}"
        if tau_value is not None:
            line += f" | tau={tau_value}"
        if log_vars is not None:
            line += f" | log_vars={log_vars}"
        if align_err is not None:
            line += f" | align_err={align_err:.4f}"
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


def compute_subspace_alignment(model, val_loaders, device, num_samples=300, rank=15):
    """计算任务间子空间对齐度 (Alignment Error)"""
    model.eval()
    task_features = []
    with torch.no_grad():
        for t, loader in enumerate(val_loaders):
            # ✅ 从整个验证集随机采样
            dataset = loader.dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            xb = torch.stack([dataset[i][0] for i in indices]).to(device)
            _, rep = model(xb, task_idx=t)
            task_features.append(rep.detach().cpu())

    align_errors = []
    for i in range(len(task_features)):
        for j in range(i + 1, len(task_features)):
            Ui, _, _ = torch.svd(task_features[i])
            Uj, _, _ = torch.svd(task_features[j])
            Ui = Ui[:, :rank]; Uj = Uj[:, :rank]
            err = 1 - (torch.norm(Ui.T @ Uj, p='fro')**2) / rank
            align_errors.append(err.item())
    model.train()
    return float(np.mean(align_errors))



# ==========================================================
# 📊 保存与绘图（含 AlignErr）
# ==========================================================
def save_metrics_and_plots(metrics, exp_name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    json_path = os.path.join(RESULT_DIR, f"{exp_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ 指标已保存至 {json_path}")

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["epoch"], metrics["train_loss"], marker='o', color='tab:red')
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title(f"{exp_name} Loss Curve"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(6, 4))
    for task_name, accs in metrics["task_acc"].items():
        plt.plot(metrics["epoch"], accs, marker='s', label=task_name)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title(f"{exp_name} Task Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_task_accuracy.png"))
    plt.close()

    # τ
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

    # log_vars
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

    # AlignErr
    if "align_err" in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(metrics["epoch"], metrics["align_err"], marker='d', color='tab:green')
        plt.xlabel("Epoch"); plt.ylabel("Subspace Alignment Error")
        plt.title(f"{exp_name} Subspace Alignment Curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_align_curve.png"))
        plt.close()
'''

# ==========================================================
# 1️⃣ JOINT + Proportional
# ==========================================================
def train_joint(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    opt = _make_optimizer_from_cfg(model.parameters(), cfg)
    model.train()

    task_sizes = [len(ld.dataset) for ld in train_loaders]
    probs = [s / sum(task_sizes) for s in task_sizes]
    metrics = {"epoch": [], "train_loss": [], "align_err": [], "task_acc": {f"task{i+1}": [] for i in range(len(train_loaders))}}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        for _ in range(sum(task_sizes)//cfg["dataloader"]["batch_size"] + 1):
            t = random.choices(range(len(train_loaders)), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward(); opt.step()
            loss_meter.update(loss.item(), yb.size(0))

        accs = evaluate_tasks(model, val_loaders, device)
        align_err = compute_subspace_alignment(model, val_loaders, device)
        metrics["epoch"].append(epoch+1); metrics["train_loss"].append(loss_meter.avg); metrics["align_err"].append(align_err)
        for i, acc in enumerate(accs): metrics["task_acc"][f"task{i+1}"].append(acc)
        print_epoch_summary("JOINT + Proportional", epoch+1, loss_meter.avg, accs, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
# ==========================================================
# 2️⃣ ALT + Balanced
# ==========================================================
def train_alt(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(model.heads.parameters(), cfg)
    opt_enc = _make_optimizer_from_cfg(model.encoder.parameters(), cfg)
    probs = [1 / len(train_loaders)] * len(train_loaders)

    metrics = {"epoch": [], "train_loss": [], "align_err": [], "task_acc": {f"task{i+1}": [] for i in range(len(train_loaders))}}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"
        step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            t = random.choices(range(len(train_loaders)), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t]))
            xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))

            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        for i, acc in enumerate(accs): metrics["task_acc"][f"task{i+1}"].append(acc)

        print_epoch_summary("ALT + Balanced", epoch + 1, loss_meter.avg, accs, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)


# ==========================================================
# 3️⃣ ALT-Temp（task-specific τ）
# ==========================================================
def train_alt_temp(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(
        [{"params": model.heads.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)
    opt_enc = _make_optimizer_from_cfg(
        [{"params": model.encoder.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {
        "epoch": [], "train_loss": [], "align_err": [],
        "tau": {f"task{i+1}": [] for i in range(num_tasks)},
        "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
    }
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"; step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().numpy()
            weights = [s ** (1.0 / tau_vals[i]) for i, s in enumerate(task_sizes)]
            probs = [w / sum(weights) for w in weights]

            t = random.choices(range(num_tasks), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))
            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().tolist()
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        print_epoch_summary("ALT-Temp (Learnable τ)", epoch + 1, loss_meter.avg, accs,
                            tau_value=tau_vals, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)


# ==========================================================
# 4️⃣ ALT-Temp + UW（τ + 不确定性权重）
# ==========================================================
def train_alt_temp_uw(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(
        [{"params": model.heads.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1},
         {"params": [model.log_vars], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)
    opt_enc = _make_optimizer_from_cfg(
        [{"params": model.encoder.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1},
         {"params": [model.log_vars], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {"epoch": [], "train_loss": [], "align_err": [],
               "tau": {f"task{i+1}": [] for i in range(num_tasks)},
               "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
               "log_vars": []}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"; step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().numpy()
            weights = [s ** (1.0 / tau_vals[i]) for i, s in enumerate(task_sizes)]
            probs = [w / sum(weights) for w in weights]

            t = random.choices(range(num_tasks), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))
            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().tolist()
        logvars = model.log_vars.detach().cpu().tolist()
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        metrics["log_vars"].append(logvars)
        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        print_epoch_summary("ALT-Temp + UW (Learnable τ + UW)", epoch + 1, loss_meter.avg, accs,
                            tau_value=tau_vals, log_vars=logvars, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
'''
import os, json, random, math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from utils import AvgMeter

# ==========================================================
# 📂 自动修正输出路径
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# ⚙️ 优化器构造函数
# ==========================================================
def _make_optimizer_from_cfg(params, cfg):
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
    print(f"  → Task 1 (Cat 🐱 vs Dog 🐶):       {accs[0]:.3f}")
    print(f"  → Task 2 (Bird 🐦 vs Frog 🐸):     {accs[1]:.3f}")
    print(f"  → Task 3 (Airplane ✈️ vs Ship 🚢): {accs[2]:.3f}")

    if tau_value is not None:
        if isinstance(tau_value, dict):
            tau_list = [v[-1] if isinstance(v, list) else v for v in tau_value.values()]
            tau_str = ", ".join([f"{t:.3f}" for t in tau_list])
            print(f"  → Learnable τ (per-task): [{tau_str}]")
        elif isinstance(tau_value, (list, tuple, np.ndarray)):
            tau_str = ", ".join([f"{float(t):.3f}" for t in tau_value])
            print(f"  → Learnable τ (per-task): [{tau_str}]")
        else:
            print(f"  → Learnable τ (Temperature): {float(tau_value):.3f}")

    if log_vars is not None:
        log_str = ", ".join([f"{float(v):.3f}" for v in log_vars])
        print(f"  → log_vars (Uncertainty): [{log_str}]")

    if align_err is not None:
        print(f"  → Subspace Alignment Error: {align_err:.4f}")

    print("------------------------------------------------------------")

    # 日志文件记录
    with open(os.path.join(RESULT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
        line = f"[{strategy_name}] Epoch {epoch} | Loss={train_loss:.4f} | AvgAcc={avg_acc:.3f} | Tasks={accs}"
        if tau_value is not None:
            line += f" | tau={tau_value}"
        if log_vars is not None:
            line += f" | log_vars={log_vars}"
        if align_err is not None:
            line += f" | align_err={align_err:.4f}"
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
# 🔹 子空间对齐误差 (Alignment Error)
# ==========================================================
def compute_subspace_alignment(model, val_loaders, device, num_samples=300, rank=15):
    model.eval()
    task_features = []
    with torch.no_grad():
        for t, loader in enumerate(val_loaders):
            dataset = loader.dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            xb = torch.stack([dataset[i][0] for i in indices]).to(device)
            _, rep = model(xb, task_idx=t)
            task_features.append(rep.detach().cpu())

    align_errors = []
    for i in range(len(task_features)):
        for j in range(i + 1, len(task_features)):
            Ui, _, _ = torch.svd(task_features[i])
            Uj, _, _ = torch.svd(task_features[j])
            Ui = Ui[:, :rank]; Uj = Uj[:, :rank]
            err = 1 - (torch.norm(Ui.T @ Uj, p='fro')**2) / rank
            align_errors.append(err.item())
    model.train()
    return float(np.mean(align_errors))


# ==========================================================
# 📊 保存与绘图（Loss, Acc, τ, log_vars, AlignErr）
# ==========================================================
def save_metrics_and_plots(metrics, exp_name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    json_path = os.path.join(RESULT_DIR, f"{exp_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ 指标已保存至 {json_path}")

    # ---------- Loss ----------
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["epoch"], metrics["train_loss"], marker='o', color='tab:red')
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title(f"{exp_name} Loss Curve"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_loss_curve.png"))
    plt.close()

    # ---------- Accuracy ----------
    plt.figure(figsize=(6, 4))
    for task_name, accs in metrics["task_acc"].items():
        plt.plot(metrics["epoch"], accs, marker='s', label=task_name)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title(f"{exp_name} Task Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_task_accuracy.png"))
    plt.close()

    # ---------- τ ----------
    if "tau" in metrics:
        plt.figure(figsize=(6, 4))
        tau_data = metrics["tau"]
        if isinstance(tau_data, dict):
            for task_name, tau_list in tau_data.items():
                plt.plot(metrics["epoch"], tau_list, marker='^', label=task_name)
            plt.legend()
        else:
            plt.plot(metrics["epoch"], tau_data, marker='^', color='tab:blue')
        plt.xlabel("Epoch"); plt.ylabel("Learnable τ")
        plt.title(f"{exp_name} Temperature τ Curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_tau_curve.png"))
        plt.close()

    # ---------- log_vars ----------
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

    # ---------- AlignErr ----------
    if "align_err" in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(metrics["epoch"], metrics["align_err"], marker='d', color='tab:green')
        plt.xlabel("Epoch"); plt.ylabel("Subspace Alignment Error")
        plt.title(f"{exp_name} Subspace Alignment Curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"{exp_name}_align_curve.png"))
        plt.close()


# ==========================================================
# 1️⃣ JOINT + Proportional
# ==========================================================
def train_joint(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    opt = _make_optimizer_from_cfg(model.parameters(), cfg)
    model.train()

    task_sizes = [len(ld.dataset) for ld in train_loaders]
    probs = [s / sum(task_sizes) for s in task_sizes]
    metrics = {"epoch": [], "train_loss": [], "align_err": [], "task_acc": {f"task{i+1}": [] for i in range(len(train_loaders))}}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        for _ in range(sum(task_sizes)//cfg["dataloader"]["batch_size"] + 1):
            t = random.choices(range(len(train_loaders)), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward(); opt.step()
            loss_meter.update(loss.item(), yb.size(0))

        accs = evaluate_tasks(model, val_loaders, device)
        align_err = compute_subspace_alignment(model, val_loaders, device)
        metrics["epoch"].append(epoch+1); metrics["train_loss"].append(loss_meter.avg); metrics["align_err"].append(align_err)
        for i, acc in enumerate(accs): metrics["task_acc"][f"task{i+1}"].append(acc)
        print_epoch_summary("JOINT + Proportional", epoch+1, loss_meter.avg, accs, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
    return metrics


# ==========================================================
# 2️⃣ ALT + Balanced
# ==========================================================
def train_alt(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(model.heads.parameters(), cfg)
    opt_enc = _make_optimizer_from_cfg(model.encoder.parameters(), cfg)
    probs = [1 / len(train_loaders)] * len(train_loaders)

    metrics = {"epoch": [], "train_loss": [], "align_err": [], "task_acc": {f"task{i+1}": [] for i in range(len(train_loaders))}}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"
        step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            t = random.choices(range(len(train_loaders)), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t]))
            xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))

            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        for i, acc in enumerate(accs): metrics["task_acc"][f"task{i+1}"].append(acc)

        print_epoch_summary("ALT + Balanced", epoch + 1, loss_meter.avg, accs, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
    return metrics


# ==========================================================
# 3️⃣ ALT-Temp（task-specific τ）
# ==========================================================
def train_alt_temp(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(
        [{"params": model.heads.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)
    opt_enc = _make_optimizer_from_cfg(
        [{"params": model.encoder.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {
        "epoch": [], "train_loss": [], "align_err": [],
        "tau": {f"task{i+1}": [] for i in range(num_tasks)},
        "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
    }
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"; step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().numpy()
            weights = [s ** (1.0 / tau_vals[i]) for i, s in enumerate(task_sizes)]
            probs = [w / sum(weights) for w in weights]

            t = random.choices(range(num_tasks), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))
            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().tolist()
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        print_epoch_summary("ALT-Temp (Learnable τ)", epoch + 1, loss_meter.avg, accs,
                            tau_value=tau_vals, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
    return metrics


# ==========================================================
# 4️⃣ ALT-Temp + UW（τ + 不确定性权重）
# ==========================================================
def train_alt_temp_uw(model, train_loaders, val_loaders, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]

    opt_head = _make_optimizer_from_cfg(
        [{"params": model.heads.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1},
         {"params": [model.log_vars], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)
    opt_enc = _make_optimizer_from_cfg(
        [{"params": model.encoder.parameters()},
         {"params": [model.log_taus], "lr": cfg["optimizer"]["lr"] * 0.1},
         {"params": [model.log_vars], "lr": cfg["optimizer"]["lr"] * 0.1}], cfg)

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {"epoch": [], "train_loss": [], "align_err": [],
               "tau": {f"task{i+1}": [] for i in range(num_tasks)},
               "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
               "log_vars": []}
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        loss_meter = AvgMeter()
        phase = "head"; step_count = 0
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().numpy()
            weights = [s ** (1.0 / tau_vals[i]) for i, s in enumerate(task_sizes)]
            probs = [w / sum(weights) for w in weights]

            t = random.choices(range(num_tasks), weights=probs, k=1)[0]
            xb, yb = next(iter(train_loaders[t])); xb, yb = xb.to(device), yb.to(device)

            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))
            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        for p in model.parameters(): p.requires_grad = True
        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.1, 5.0).detach().cpu().tolist()
        logvars = model.log_vars.detach().cpu().tolist()
        align_err = compute_subspace_alignment(model, val_loaders, device)

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        metrics["log_vars"].append(logvars)
        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        print_epoch_summary("ALT-Temp + UW (Learnable τ + UW)", epoch + 1, loss_meter.avg, accs,
                            tau_value=tau_vals, log_vars=logvars, align_err=align_err)

    save_metrics_and_plots(metrics, exp_name)
    return metrics
'''