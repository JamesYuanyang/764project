import math, random, torch, numpy as np
from torch.nn.utils import clip_grad_norm_
from utils import AvgMeter
from .common import (
    make_optimizer_from_cfg, evaluate_tasks, compute_subspace_alignment,
    print_epoch_summary, save_metrics_and_plots
)

def train_alt_temp_uw(model, train_loaders, val_loaders, cfg, device=None):
    """ALT-Temp + UW（Learnable τ + Uncertainty Weighting + Gradient Clipping + CKA）"""
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]
    base_lr = cfg["optimizer"]["lr"]

    # ✅ 学习率分层更稳定：encoder/head正常，tau小10倍，log_vars小100倍
    opt_head = make_optimizer_from_cfg(
        [
            {"params": model.heads.parameters(), "lr": base_lr},
            {"params": [model.log_taus], "lr": base_lr * 0.1},
        ], cfg
    )
    opt_enc = make_optimizer_from_cfg(
        [
            {"params": model.encoder.parameters(), "lr": base_lr},
            {"params": [model.log_taus], "lr": base_lr * 0.1},
            {"params": [model.log_vars], "lr": base_lr * 0.001},  # UW 极低 lr
        ], cfg
    )

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {
        "epoch": [], "train_loss": [], "align_err": [], "cka": [],
        "tau": {f"task{i+1}": [] for i in range(num_tasks)},
        "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
        "log_vars": []
    }
    exp_name = cfg["experiment"]["name"]

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        loss_meter = AvgMeter()
        phase = "head"
        step_count = 0
        iters = [iter(ld) for ld in train_loaders]
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            # === 任务采样（带 τ 调整）===
            tau_vals = torch.exp(model.log_taus).clamp(0.5, 3.0).detach().cpu().numpy()
            weights = [s ** (1.0 / tau_vals[i]) for i, s in enumerate(task_sizes)]
            probs = [w / sum(weights) for w in weights]
            t = random.choices(range(num_tasks), weights=probs, k=1)[0]

            try:
                xb, yb = next(iters[t])
            except StopIteration:
                iters[t] = iter(train_loaders[t])
                xb, yb = next(iters[t])
            xb, yb = xb.to(device), yb.to(device)

            # === 阶段切换 ===
            if phase == "head":
                for p in model.encoder.parameters(): p.requires_grad = False
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = True
                for p in [model.log_vars]: p.requires_grad = False  # ❗ UW 固定
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                for p in [model.log_vars]: p.requires_grad = True
                opt = opt_enc

            # === 优化 ===
            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()

            # ✅ 梯度裁剪（增强稳定性）
            clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])

            opt.step()

            # ✅ 防止 log_vars 发散
            with torch.no_grad():
                model.log_vars.data.clamp_(-5.0, 5.0)

            loss_meter.update(loss.item(), yb.size(0))

            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        # ===== 每 epoch 评估 =====
        for p in model.parameters():
            p.requires_grad = True

        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.5, 3.0).detach().cpu().tolist()
        logvars = model.log_vars.detach().cpu().tolist()

        # ✅ Compute Alignment & CKA
        align_metrics = compute_subspace_alignment(model, val_loaders, device)
        align_err = align_metrics["align_err"]
        cka = align_metrics["cka"]

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        metrics["cka"].append(cka)
        metrics["log_vars"].append(logvars)

        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        # ✅ 打印信息（含 τ, log_vars, CKA）
        print_epoch_summary(
            "ALT-Temp + UW (Stable + CKA)",
            epoch + 1, loss_meter.avg, accs,
            tau_value=tau_vals, log_vars=logvars, align_err=align_metrics
        )

    save_metrics_and_plots(metrics, exp_name)
