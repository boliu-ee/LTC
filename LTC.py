import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# 1. 生成一个小型时序任务
# =========================
# 任务：输入 noisy signal，预测下一时刻的 clean signal
# 这个信号的频率会分段变化，方便观察“液态网络”是否会改变自己的时间常数 tau

torch.manual_seed(0)
np.random.seed(0)


def generate_signal(T=500, dt=0.1):
    t = np.arange(T) * dt

    # 分段变化频率：低频 -> 高频 -> 低频
    freq = np.where(
        t < T * dt / 3,
        1.0,
        np.where(t < 2 * T * dt / 3, 2.5, 0.7)
    )

    # 幅值也轻微变化
    amp = 1.0 + 0.3 * np.sin(0.15 * t)

    clean = amp * np.sin(2 * np.pi * freq * t)
    noisy = clean + 0.15 * np.random.randn(T)

    # 用 noisy[t] 预测 clean[t+1]
    x = noisy[:-1].astype(np.float32)[:, None]   # shape: [T-1, 1]
    y = clean[1:].astype(np.float32)[:, None]    # shape: [T-1, 1]

    return t[:-1], x, y, clean[:-1], freq[:-1]


# =========================
# 2. 一个教学版 Liquid Cell
# =========================
# 核心思想：
# h_{t+1} = h_t + dt * ( -h_t + candidate ) / tau(x_t, h_t)
#
# 普通 RNN / leaky RNN 往往 tau 是固定的；
# 液态网络里 tau 会根据当前输入 x 和状态 h 动态变化。
#
# tau 小 -> 神经元响应更快
# tau 大 -> 神经元变化更慢，记忆更长


class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau_min=0.2, tau_max=2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max

        # 生成候选状态
        self.candidate_net = nn.Linear(input_size + hidden_size, hidden_size)

        # 动态生成 tau
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)

        # 输出层
        self.output_net = nn.Linear(hidden_size, 1)

    def forward_step(self, x_t, h_t, dt=0.1):
        """
        x_t: [B, input_size]
        h_t: [B, hidden_size]
        """
        z = torch.cat([x_t, h_t], dim=-1)

        candidate = torch.tanh(self.candidate_net(z))

        # tau in [tau_min, tau_max]
        tau = self.tau_min + torch.sigmoid(self.tau_net(z)) * (self.tau_max - self.tau_min)

        # Euler discretization of a continuous-time dynamics
        dh = (-h_t + candidate) / tau
        h_next = h_t + dt * dh

        y_t = self.output_net(h_next)
        return y_t, h_next, tau

    def forward_sequence(self, x_seq, dt=0.1):
        """
        x_seq: [T, B, input_size]
        returns:
            y_seq:   [T, B, 1]
            tau_seq: [T, B, hidden_size]
        """
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_size, device=x_seq.device)

        outputs = []
        taus = []

        for t in range(T):
            y, h, tau = self.forward_step(x_seq[t], h, dt=dt)
            outputs.append(y)
            taus.append(tau)

        y_seq = torch.stack(outputs, dim=0)
        tau_seq = torch.stack(taus, dim=0)
        return y_seq, tau_seq


# =========================
# 3. 准备数据
# =========================
t, x_np, y_np, clean_np, freq_np = generate_signal(T=500, dt=0.1)

x = torch.tensor(x_np).unsqueeze(1)   # [T, B=1, 1]
y = torch.tensor(y_np).unsqueeze(1)   # [T, B=1, 1]

device = torch.device("cpu")
x = x.to(device)
y = y.to(device)

# =========================
# 4. 定义模型并训练
# =========================
model = LiquidCell(input_size=1, hidden_size=16, tau_min=0.2, tau_max=2.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

loss_history = []

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred, tau_seq = model.forward_sequence(x, dt=0.1)
    loss = loss_fn(pred, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.6f}")

# =========================
# 5. 推理并可视化
# =========================
model.eval()
with torch.no_grad():
    pred, tau_seq = model.forward_sequence(x, dt=0.1)

pred_np = pred.squeeze(1).squeeze(-1).cpu().numpy()      # [T]
tau_np = tau_seq.squeeze(1).cpu().numpy()                # [T, hidden_size]

# 选几个神经元看看 tau 如何随时间变化
tau_show_idx = [0, 1, 2, 3]

plt.figure(figsize=(14, 10))

# (1) 训练 loss
plt.subplot(3, 1, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)

# (2) 预测结果 vs 真实信号
plt.subplot(3, 1, 2)
plt.plot(t, y_np.squeeze(-1), label="Target (clean next step)")
plt.plot(t, x_np.squeeze(-1), label="Input (noisy current step)", alpha=0.6)
plt.plot(t, pred_np, label="Liquid model prediction", linestyle="--")
plt.title("Prediction on a Piecewise-Frequency Signal")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend()
plt.grid(True)

# 用背景色标出不同频率区间
T_total = len(t)
split1 = T_total // 3
split2 = 2 * T_total // 3
plt.axvspan(t[0], t[split1], alpha=0.08)
plt.axvspan(t[split1], t[split2], alpha=0.12)
plt.axvspan(t[split2], t[-1], alpha=0.08)

# (3) tau 动态变化
plt.subplot(3, 1, 3)
for i in tau_show_idx:
    plt.plot(t, tau_np[:, i], label=f"Neuron {i} tau")
plt.title("Dynamic Time Constants (tau) -- This is the 'Liquid' part")
plt.xlabel("Time")
plt.ylabel("tau")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

