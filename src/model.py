import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 🧩 简单 MLP 编码器
# ==========================================================
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, dropout=0.1, init_std=0.02):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # ✅ Gaussian 初始化（保持任务间分布一致）
        nn.init.normal_(self.fc.weight, mean=0.0, std=init_std)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop(self.norm(self.act(self.fc(x))))
        return x


# ==========================================================
# 🎯 多任务模型（支持 task-specific τᵢ 与不确定性加权 UW）
# ==========================================================
class MultiTaskModel(nn.Module):
    def __init__(self, cfg_model, use_uw=False):
        super().__init__()

        # ---------------- Encoder & Head ----------------
        enc = cfg_model["encoder"]
        head = cfg_model["heads"]

        self.encoder = MLPEncoder(enc["input_dim"], enc["hidden_dim"], enc["dropout"])
        self.num_tasks = head["num_tasks"]
        self.out_dim = head["out_dim"]

        # 多任务分类头
        self.heads = nn.ModuleList([
            nn.Linear(enc["hidden_dim"], self.out_dim)
            for _ in range(self.num_tasks)
        ])

        # ✅ 初始化 heads 权重
        for h in self.heads:
            nn.init.normal_(h.weight, mean=0.0, std=0.02)
            if h.bias is not None:
                nn.init.zeros_(h.bias)

        # ---------------- Task-specific τ 参数 ----------------
        self.log_taus = nn.Parameter(torch.log(torch.ones(self.num_tasks) * 2.5))

        # ---------------- UW 参数（可选） ----------------
        self.use_uw = use_uw
        if use_uw:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.register_buffer("log_vars", torch.zeros(self.num_tasks))

        # ---------------- 可选正则参数 λₐ ----------------
        self.lambda_a = head.get("lambda_a", 0.0)

    # ======================================================
    # 🔹 前向传播
    # ======================================================
    def forward(self, x, task_idx):
        """
        输入:
            x: Tensor [B, ...]
            task_idx: 当前任务索引 (int)
        输出:
            logits: 当前任务输出
            h: 共享编码特征
        """
        h = self.encoder(x)
        logits = self.heads[task_idx](h)
        return logits, h

    # ======================================================
    # 🔹 任务专属 Loss 计算
    # ======================================================
    def task_loss(self, logits, y, t):
        """
        每任务使用自己的 τᵢ:
            τᵢ = exp(log_τᵢ)
        若启用 UW:
            L_t = 0.5 * exp(-s_t) * CE + 0.5 * s_t
        """
        # --- 温度缩放 ---
        tau = torch.exp(self.log_taus[t]).clamp(0.5, 3.0)
        scaled_logits = logits / tau
        ce = F.cross_entropy(scaled_logits, y)

        # --- 不确定性加权 UW ---
        if self.use_uw:
            # ✅ 限制 log_var 避免溢出
            clamped_log_var = torch.clamp(self.log_vars[t], min=-5.0, max=5.0)
            precision = torch.exp(-clamped_log_var)
            loss = 0.5 * precision * ce + 0.5 * clamped_log_var
        else:
            loss = ce

        # --- 可选任务头正则化 ---
        if self.lambda_a > 0:
            reg = 0.0
            for p in self.heads[t].parameters():
                reg += torch.sum(p ** 2)
            loss = loss + self.lambda_a * reg

        return loss

