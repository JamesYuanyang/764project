import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv

# ==========================================================
# 📂 自动定位 results 目录
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# 📊 读取单个 run 文件
# ==========================================================
def load_metrics(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ 找不到文件: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "task_acc" not in data or len(data["task_acc"]) == 0:
        return None

    task_accs = np.array([np.array(v) for v in data["task_acc"].values()])
    macro_acc = task_accs.mean(axis=0)
    worst_acc = task_accs.min(axis=0)
    final_macro = float(macro_acc[-1])
    final_worst = float(worst_acc[-1])

    tau_curve = data.get("tau", [])
    if isinstance(tau_curve, dict):
        tau_avg = np.mean(np.array([v for v in tau_curve.values()]), axis=0)
    elif isinstance(tau_curve, list):
        tau_avg = np.array(tau_curve)
    else:
        tau_avg = np.array([])

    tau_final = float(tau_avg[-1]) if len(tau_avg) > 0 else None
    logvars = data.get("log_vars", [])
    logvars_arr = np.array(logvars) if len(logvars) > 0 else None

    return {
        "macro_curve": macro_acc,
        "worst_curve": worst_acc,
        "final_macro": final_macro,
        "final_worst": final_worst,
        "tau_curve": tau_avg,
        "tau_final": tau_final,
        "logvars": logvars_arr,
        "epoch": data["epoch"],
        "align_err": np.array(data.get("align_err", [])),
        "cka": np.array(data.get("cka", [])),
    }

# ==========================================================
# 📈 统计工具函数
# ==========================================================
def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    more = sum(a > b for a in x for b in y)
    less = sum(a < b for a in x for b in y)
    return (more - less) / (n1 * n2)

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2))
    d = mean_diff / pooled_std
    correction = 1 - (3 / (4*(nx + ny) - 9))
    return d * correction

def holm_bonferroni_correction(p_values):
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min((m - rank) * p_values[idx], 1.0)
    return adjusted

# ==========================================================
# 📈 主分析逻辑
# ==========================================================
def main():
    print(f"📊 正在分析结果文件 (from {RESULT_DIR}) ...")

    methods = {
        "JOINT + Proportional": "JOINT + Proportional",
        "ALT + Balanced": "ALT + Balanced",
        "ALT-Temp (Learnable τ)": "ALT-Temp",
        "ALT-Temp + UW (Learnable τ + UW)": "ALT-Temp + UW",
        "Linear Probe + Fine-tuning": "Linear Probe + Fine-tuning"
    }

    num_runs = 5
    results = {}

    # ======================================================
    # 📦 读取所有方法的 run1–run5
    # ======================================================
    for method_name, prefix in methods.items():
        runs = []
        for i in range(1, num_runs + 1):
            fname = f"{prefix}_run{i}_metrics.json"
            metrics = load_metrics(fname)
            if metrics:
                runs.append(metrics)
        if len(runs) == 0:
            print(f"⚠️ {method_name} 无数据")
            continue

        epochs = runs[0]["epoch"]
        macro_curves = np.array([r["macro_curve"] for r in runs])
        worst_curves = np.array([r["worst_curve"] for r in runs])
        align_curves = np.array([r["align_err"] for r in runs if len(r["align_err"]) == len(epochs)])
        cka_curves = np.array([r["cka"] for r in runs if len(r["cka"]) == len(epochs)])

        results[method_name] = {
            "epoch": epochs,
            "macro_mean": macro_curves.mean(axis=0),
            "macro_std": macro_curves.std(axis=0),
            "worst_mean": worst_curves.mean(axis=0),
            "worst_std": worst_curves.std(axis=0),
            "align_mean": align_curves.mean(axis=0) if len(align_curves) > 0 else None,
            "align_std": align_curves.std(axis=0) if len(align_curves) > 0 else None,
            "cka_mean": cka_curves.mean(axis=0) if len(cka_curves) > 0 else None,
            "cka_std": cka_curves.std(axis=0) if len(cka_curves) > 0 else None,
            "final_macros": [r["final_macro"] for r in runs],
            "final_worsts": [r["final_worst"] for r in runs]
        }

    # ======================================================
    # 📈 绘制 Macro Accuracy（±std）
    # ======================================================
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        plt.plot(r["epoch"], r["macro_mean"], label=name)
        plt.fill_between(r["epoch"], r["macro_mean"]-r["macro_std"], r["macro_mean"]+r["macro_std"], alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("Macro Accuracy")
    plt.title("Macro Accuracy (mean ± std, 5 runs)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "macro_accuracy_run5.png"))
    plt.close()
    print("📈 已生成 macro_accuracy_run5.png")

    # ======================================================
    # 📉 绘制 Worst-task Accuracy
    # ======================================================
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        plt.plot(r["epoch"], r["worst_mean"], label=name)
        plt.fill_between(r["epoch"], r["worst_mean"]-r["worst_std"], r["worst_mean"]+r["worst_std"], alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("Worst-task Accuracy")
    plt.title("Worst-task Accuracy (mean ± std, 5 runs)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "worst_accuracy_run5.png"))
    plt.close()
    print("📉 已生成 worst_accuracy_run5.png")

    # ======================================================
    # 🔹 绘制 Alignment Error
    # ======================================================
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        if r["align_mean"] is not None:
            plt.plot(r["epoch"], r["align_mean"], label=name)
            plt.fill_between(r["epoch"], r["align_mean"]-r["align_std"], r["align_mean"]+r["align_std"], alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("Subspace Alignment Error")
    plt.title("Alignment Error (mean ± std)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "alignment_error_run5.png"))
    plt.close()
    print("✅ 已生成 alignment_error_run5.png")

    # ======================================================
    # 🔹 绘制 CKA
    # ======================================================
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        if r["cka_mean"] is not None:
            plt.plot(r["epoch"], r["cka_mean"], label=name)
            plt.fill_between(r["epoch"], r["cka_mean"]-r["cka_std"], r["cka_mean"]+r["cka_std"], alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("Linear CKA")
    plt.title("CKA Similarity (mean ± std)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "cka_run5.png"))
    plt.close()
    print("📈 已生成 cka_run5.png")

    # ======================================================
    # 📊 最终宏平均对比柱状图
    # ======================================================
    methods_list = list(results.keys())
    means = [np.mean(results[m]["final_macros"]) for m in methods_list]
    stds = [np.std(results[m]["final_macros"]) for m in methods_list]
    plt.figure(figsize=(9, 6))
    plt.bar(methods_list, means, yerr=stds, capsize=5, color=plt.cm.tab10.colors[:len(methods_list)])
    plt.ylabel("Final Macro Accuracy")
    plt.title("Final Macro Accuracy Comparison (mean ± std)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_macro_bar_run5.png"))
    plt.close()
    print("📊 已生成 final_macro_bar_run5.png")

    # ======================================================
    # 🔍 显著性检验 + 效应量
    # ======================================================
    main_method = "ALT-Temp + UW (Learnable τ + UW)"
    if main_method not in results:
        print("❌ 缺少主方法，跳过显著性分析。")
        return

    main_scores = results[main_method]["final_macros"]
    p_values, deltas, g_values, compare_names = [], [], [], []
    print("\n================ 显著性检验与效应量分析 ================")
    for name, r in results.items():
        if name == main_method:
            continue
        vals = r["final_macros"]
        if len(vals) != len(main_scores):
            continue
        t_stat, p = stats.ttest_rel(main_scores, vals)
        delta = cliffs_delta(main_scores, vals)
        g = hedges_g(main_scores, vals)
        p_values.append(p)
        deltas.append(delta)
        g_values.append(g)
        compare_names.append(name)
        print(f"{main_method} vs {name}: p={p:.4f}, δ={delta:.3f}, g={g:.3f}")

    adjusted_p = holm_bonferroni_correction(p_values)
    print("\n🎯 Holm–Bonferroni 校正后结果：")
    for name, p_adj in zip(compare_names, adjusted_p):
        sig = "✅ 显著" if p_adj < 0.05 else "❌ 不显著"
        print(f"{main_method} vs {name}: adj_p={p_adj:.4f} → {sig}")

    csv_path = os.path.join(RESULT_DIR, "significance_results.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Comparison", "p-value", "adj_p (Holm-Bonferroni)", "Cliff’s δ", "Hedges’ g", "Significant"])
        for name, p, p_adj, d, g in zip(compare_names, p_values, adjusted_p, deltas, g_values):
            sig = "Yes" if p_adj < 0.05 else "No"
            writer.writerow([f"{main_method} vs {name}", f"{p:.4f}", f"{p_adj:.4f}", f"{d:.3f}", f"{g:.3f}", sig])
    print(f"📁 显著性检验结果已保存至: {csv_path}")


if __name__ == "__main__":
    main()
