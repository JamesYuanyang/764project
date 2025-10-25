import math, random, torch
from torch.nn.utils import clip_grad_norm_
from utils import AvgMeter
from .common import (
    make_optimizer_from_cfg, evaluate_tasks,
    print_epoch_summary, save_metrics_and_plots
)

def linear_probe_and_finetune(model, train_loaders, val_loaders, cfg, device=None):
    """
    Linear Probe + Fine-tuning
    阶段1：冻结 encoder，仅训练 heads（线性探针）
    阶段2：解冻 encoder，做少步微调
    """
    device = device or ("cuda" if torch.cuda.is_available() and cfg["device"]["use_gpu"] else "cpu")
    model.to(device)
    base_lr = cfg["optimizer"]["lr"]

    num_tasks = len(train_loaders)
    exp_name = cfg["experiment"]["name"]

    # ==========================================================
    # 🔹 Stage 1: Linear Probe（冻结 encoder）
    # ==========================================================
    print("\n==============================")
    print("🔍 Stage 1: Linear Probe (Freeze Encoder)")
    print("==============================")

    for p in model.encoder.parameters():
        p.requires_grad = False
    for h in model.heads:
        for p in h.parameters():
            p.requires_grad = True

    opt = make_optimizer_from_cfg(model.heads.parameters(), cfg)

    probe_epochs = cfg["training"].get("probe_epochs", 10)
    metrics_probe = {"epoch": [], "train_loss": [], "task_acc": {f"task{i+1}": [] for i in range(num_tasks)}}

    for epoch in range(probe_epochs):
        model.train()
        loss_meter = AvgMeter()
        iters = [iter(ld) for ld in train_loaders]
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            t = random.choice(range(num_tasks))
            try:
                xb, yb = next(iters[t])
            except StopIteration:
                iters[t] = iter(train_loaders[t])
                xb, yb = next(iters[t])
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            clip_grad_norm_(model.heads.parameters(), cfg["training"]["max_grad_norm"])
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))

        accs = evaluate_tasks(model, val_loaders, device)
        metrics_probe["epoch"].append(epoch + 1)
        metrics_probe["train_loss"].append(loss_meter.avg)
        for i, acc in enumerate(accs):
            metrics_probe["task_acc"][f"task{i+1}"].append(acc)
        print_epoch_summary("Linear Probe", epoch + 1, loss_meter.avg, accs)

    save_metrics_and_plots(metrics_probe, f"{exp_name}_linear_probe")

    # ==========================================================
    # 🔹 Stage 2: Fine-tuning（解冻 encoder）
    # ==========================================================
    print("\n==============================")
    print("🔧 Stage 2: Fine-tuning (Unfreeze Encoder)")
    print("==============================")

    for p in model.encoder.parameters():
        p.requires_grad = True

    opt = make_optimizer_from_cfg(model.parameters(), cfg)
    finetune_epochs = cfg["training"].get("finetune_epochs", 10)

    metrics_finetune = {"epoch": [], "train_loss": [], "task_acc": {f"task{i+1}": [] for i in range(num_tasks)}}

    for epoch in range(finetune_epochs):
        model.train()
        loss_meter = AvgMeter()
        iters = [iter(ld) for ld in train_loaders]
        total_steps = sum(math.ceil(len(ld.dataset) / cfg["dataloader"]["batch_size"]) for ld in train_loaders)

        for _ in range(total_steps):
            t = random.choice(range(num_tasks))
            try:
                xb, yb = next(iters[t])
            except StopIteration:
                iters[t] = iter(train_loaders[t])
                xb, yb = next(iters[t])
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits, _ = model(xb, t)
            loss = model.task_loss(logits, yb, t)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            opt.step()
            loss_meter.update(loss.item(), yb.size(0))

        accs = evaluate_tasks(model, val_loaders, device)
        metrics_finetune["epoch"].append(epoch + 1)
        metrics_finetune["train_loss"].append(loss_meter.avg)
        for i, acc in enumerate(accs):
            metrics_finetune["task_acc"][f"task{i+1}"].append(acc)
        print_epoch_summary("Fine-tuning", epoch + 1, loss_meter.avg, accs)

    save_metrics_and_plots(metrics_finetune, f"{exp_name}_finetune")

    print("\n✅ Linear Probe + Fine-tuning 完成！结果已保存。")
