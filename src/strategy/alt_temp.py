import math, random, torch, numpy as np
from torch.nn.utils import clip_grad_norm_
from utils import AvgMeter
from .common import (
    make_optimizer_from_cfg, evaluate_tasks, compute_subspace_alignment,
    print_epoch_summary, save_metrics_and_plots
)

def train_alt_temp(model, train_loaders, val_loaders, cfg, device=None):
    """ALT-Temp（task-specific τ, 完全对齐版 + Gradient Clipping + CKA）"""
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    steps_per_alt = cfg["strategy"]["steps_per_alt"]
    base_lr = cfg["optimizer"]["lr"]

    # ✅ 分层学习率结构（保持一致）
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
        ], cfg
    )

    num_tasks = len(train_loaders)
    task_sizes = [len(ld.dataset) for ld in train_loaders]

    metrics = {
        "epoch": [], "train_loss": [], "align_err": [], "cka": [],
        "tau": {f"task{i+1}": [] for i in range(num_tasks)},
        "task_acc": {f"task{i+1}": [] for i in range(num_tasks)},
    }
    exp_name = cfg["experiment"]["name"]

    # ✅ 预算等价控制
    total_backward_steps = 0
    max_backward_steps = cfg["training"].get("max_backward_steps", None)
    if max_backward_steps:
        print(f"⚙️ 当前训练受限于预算上限：最多 {max_backward_steps} 次反向传播。")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        loss_meter = AvgMeter()
        phase = "head"
        step_count = 0
        iters = [iter(ld) for ld in train_loaders]
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            # === 任务采样（基于 learnable τ）===
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
                opt = opt_head
            else:
                for p in model.encoder.parameters(): p.requires_grad = True
                for h in model.heads:
                    for p in h.parameters(): p.requires_grad = False
                opt = opt_enc

            # === 反向传播 ===
            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()

            # ✅ 梯度裁剪，保证稳定性
            clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])

            opt.step()

            # ✅ 防止 τ 发散
            with torch.no_grad():
                model.log_taus.data.clamp_(-1.0, 1.5)  # τ ∈ [0.37, 4.48]

            loss_meter.update(loss.item(), yb.size(0))

            total_backward_steps += 1
            if max_backward_steps and total_backward_steps >= max_backward_steps:
                print(f"⏹️ 达到预算上限 {max_backward_steps} 次反向传播，提前停止训练。")
                break

            step_count += 1
            if step_count >= steps_per_alt:
                phase = "encoder" if phase == "head" else "head"
                step_count = 0

        if max_backward_steps and total_backward_steps >= max_backward_steps:
            break

        # === 评估 ===
        for p in model.parameters():
            p.requires_grad = True

        accs = evaluate_tasks(model, val_loaders, device)
        tau_vals = torch.exp(model.log_taus).clamp(0.5, 3.0).detach().cpu().tolist()

        # ✅ Compute Alignment & CKA
        align_metrics = compute_subspace_alignment(model, val_loaders, device)
        align_err = align_metrics["align_err"]
        cka = align_metrics["cka"]

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        metrics["cka"].append(cka)

        for i, tau in enumerate(tau_vals):
            metrics["tau"][f"task{i+1}"].append(tau)
            metrics["task_acc"][f"task{i+1}"].append(accs[i])

        # ✅ 打印包含 τ & CKA 的信息
        print_epoch_summary(
            "ALT-Temp (Learnable τ + GradClip + CKA)",
            epoch + 1, loss_meter.avg, accs,
            tau_value=tau_vals, align_err=align_metrics
        )

    save_metrics_and_plots(metrics, exp_name)
