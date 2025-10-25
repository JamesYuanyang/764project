import math, random, torch
from torch.nn.utils import clip_grad_norm_
from utils import AvgMeter
from .common import (
    make_optimizer_from_cfg, evaluate_tasks, compute_subspace_alignment,
    print_epoch_summary, save_metrics_and_plots
)

def train_alt(model, train_loaders, val_loaders, cfg, device=None):
    """ALT + Balanced（统一预算 + 梯度裁剪 + 对齐优化结构 + CKA）"""
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)

    # ✅ 对齐分层优化器结构
    base_lr = cfg["optimizer"]["lr"]
    opt_head = make_optimizer_from_cfg(
        [{"params": model.heads.parameters(), "lr": base_lr}],
        cfg
    )
    opt_enc = make_optimizer_from_cfg(
        [{"params": model.encoder.parameters(), "lr": base_lr}],
        cfg
    )

    steps_per_alt = cfg["strategy"]["steps_per_alt"]
    num_tasks = len(train_loaders)
    probs = [1 / num_tasks] * num_tasks
    exp_name = cfg["experiment"]["name"]

    metrics = {
        "epoch": [], "train_loss": [], "align_err": [], "cka": [],
        "task_acc": {f"task{i+1}": [] for i in range(num_tasks)}
    }

    # ✅ 预算等价计数器
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
            # 🎯 任务采样
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

            # === 优化 ===
            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()

            # ✅ 梯度裁剪（提升稳定性）
            clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])

            opt.step()
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

        # === 验证与记录 ===
        for p in model.parameters():
            p.requires_grad = True

        accs = evaluate_tasks(model, val_loaders, device)

        # ✅ Compute Alignment & CKA
        align_metrics = compute_subspace_alignment(model, val_loaders, device)
        align_err = align_metrics["align_err"]
        cka = align_metrics["cka"]

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(loss_meter.avg)
        metrics["align_err"].append(align_err)
        metrics["cka"].append(cka)

        for i, acc in enumerate(accs):
            metrics["task_acc"][f"task{i+1}"].append(acc)

        # ✅ 打印包含 CKA 的信息
        print_epoch_summary("ALT + Balanced", epoch + 1, loss_meter.avg, accs, align_err=align_metrics)

    save_metrics_and_plots(metrics, exp_name)
