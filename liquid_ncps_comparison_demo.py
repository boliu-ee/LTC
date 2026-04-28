import argparse
import json
import time  # 先导入 time 模块
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ncps.torch import LTC


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# 2. Synthetic task: configurable piecewise-frequency sequence
# ============================================================

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_piecewise_frequency_array(
    total_steps: int,
    freqs: List[float],
    change_points: List[float],
) -> np.ndarray:
    """
    freqs: [f1, f2, ..., fk]
    change_points: normalized positions in (0,1), length = k-1
    Example:
        freqs=[1.0, 3.0, 0.8]
        change_points=[0.3, 0.7]
    """
    assert len(freqs) >= 1
    assert len(change_points) == len(freqs) - 1
    cps = [0] + [int(total_steps * c) for c in change_points] + [total_steps]
    arr = np.zeros(total_steps, dtype=np.float32)
    for i in range(len(freqs)):
        arr[cps[i]:cps[i + 1]] = freqs[i]
    return arr


def generate_single_sequence(
    total_steps: int,
    history_len: int,
    dt: float,
    freqs: List[float],
    change_points: List[float],
    noise_std: float,
    amp_mod: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_seq:      [T, 1] noisy observed sequence for input windows
      y_clean:    [T, 1] clean signal aligned with x_seq
      freq_arr:   [T]
      t_arr:      [T]
    """
    T = total_steps
    t = np.arange(T, dtype=np.float32) * dt
    freq_arr = make_piecewise_frequency_array(T, freqs, change_points)

    phase = np.cumsum(2.0 * np.pi * freq_arr * dt).astype(np.float32)
    if amp_mod:
        amp = 1.0 + 0.25 * np.sin(0.17 * t) + 0.1 * np.sin(0.043 * t + 1.3)
    else:
        amp = np.ones_like(t, dtype=np.float32)

    clean = amp * np.sin(phase)
    noisy = clean + noise_std * np.random.randn(T).astype(np.float32)

    return noisy[:, None], clean[:, None], freq_arr, t


def build_dataset(
    num_sequences: int,
    total_steps: int,
    history_len: int,
    dt: float,
    freqs: List[float],
    change_points: List[float],
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window next-step prediction dataset.
    Input shape:  [N, history_len, 1]
    Target shape: [N, 1]
    """
    xs = []
    ys = []
    for _ in range(num_sequences):
        noisy, clean, _, _ = generate_single_sequence(
            total_steps=total_steps,
            history_len=history_len,
            dt=dt,
            freqs=freqs,
            change_points=change_points,
            noise_std=noise_std,
        )
        # use past history_len samples to predict clean next sample
        for i in range(total_steps - history_len - 1):
            xs.append(noisy[i:i + history_len])
            ys.append(clean[i + history_len])
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    return x, y


# ============================================================
# 3. Models: FNN / RNN / LSTM / LTC (official ncps)
# ============================================================

class FNNPredictor(nn.Module):
    def __init__(self, history_len: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_len, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LTCPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.ltc = LTC(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.ltc(x)
        return self.head(out[:, -1, :])


# ============================================================
# 4. Parameter matching for fair comparison
# ============================================================

def build_model(model_name: str, history_len: int, hidden_size: int) -> nn.Module:
    if model_name == "FNN":
        return FNNPredictor(history_len=history_len, hidden_size=hidden_size)
    if model_name == "RNN":
        return RNNPredictor(input_size=1, hidden_size=hidden_size)
    if model_name == "LSTM":
        return LSTMPredictor(input_size=1, hidden_size=hidden_size)
    if model_name == "LTC":
        return LTCPredictor(input_size=1, hidden_size=hidden_size)
    raise ValueError(f"Unknown model_name={model_name}")


def match_hidden_size(
    model_name: str,
    history_len: int,
    target_params: int,
    search_max_hidden: int = 128,
) -> Tuple[int, int]:
    best_h = None
    best_p = None
    best_gap = None
    for h in range(2, search_max_hidden + 1):
        model = build_model(model_name, history_len=history_len, hidden_size=h)
        p = count_parameters(model)
        gap = abs(p - target_params)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_h = h
            best_p = p
    return best_h, best_p


# ============================================================
# 5. Training / evaluation
# ============================================================
@dataclass
class TrainResult:
    model_name: str
    hidden_size: int
    n_params: int
    train_losses: List[float]
    test_mse: float
    model: nn.Module



def train_model(
    model_name: str,
    hidden_size: int,
    history_len: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> TrainResult:
    model = build_model(model_name, history_len=history_len, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    for epoch in range(epochs):
        start_time = time.time()  # 记录 epoch 开始时间
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
        train_losses.append(total_loss / max(total_count, 1))
        epoch_time = time.time() - start_time  # 计算耗时（秒）
        # 实时打印到终端
        print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {train_losses[-1]:.4f} | Time: {epoch_time:.2f}s", flush=True)

    model.eval()
    preds = []
    tgts = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            tgts.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    tgts = np.concatenate(tgts, axis=0)
    test_mse = float(np.mean((preds - tgts) ** 2))

    return TrainResult(
        model_name=model_name,
        hidden_size=hidden_size,
        n_params=count_parameters(model),
        train_losses=train_losses,
        test_mse=test_mse,
        model=model,
    )


# ============================================================
# 6. LTC effective tau extraction from official ncps LTCCell
# ============================================================

def _sigmoid(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, x.shape[-1], 1)
    mues = x - mu
    return torch.sigmoid(sigma * mues)


def extract_ltc_effective_tau(
    ltc_model: LTCPredictor,
    x_seq: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Approximate time-varying effective tau from the official ncps LTCCell internals.

    Following the cell's ODE structure, we build:
        tau_eff ~ cm / (gleak + sum dynamic conductances)

    This is not a separate teaching model. It is an interpretation extracted from the
    trained official LTCCell parameters and its gating/conductance structure.
    """
    ltc_model.eval()
    cell = ltc_model.ltc.rnn_cell

    with torch.no_grad():
        B, T, C = x_seq.shape
        hidden = torch.zeros(B, cell.state_size, device=device)
        taus_all = []

        input_w = cell.sensory_w
        input_mu = cell.sensory_mu
        input_sigma = cell.sensory_sigma
        input_erev = cell.sensory_erev

        rec_w = cell.w
        rec_mu = cell.mu
        rec_sigma = cell.sigma

        gleak = torch.abs(cell.gleak)
        cm = torch.abs(cell.cm)
        epsilon = cell._epsilon

        for t in range(T):
            xt = x_seq[:, t, :].to(device)

            sensory_act = input_w * _sigmoid(xt, input_mu, input_sigma)
            sensory_cond = sensory_act.sum(dim=1)

            rec_act = rec_w * _sigmoid(hidden, rec_mu, rec_sigma)
            rec_cond = rec_act.sum(dim=1)

            total_cond = gleak + sensory_cond + rec_cond + epsilon
            tau_eff = cm / total_cond
            taus_all.append(tau_eff.cpu().numpy())

            # advance hidden using the official cell itself
            hidden, _ = cell(xt, hidden)

        taus_all = np.stack(taus_all, axis=1)  # [B, T, H]
        return taus_all[0]  # [T, H] for first sample


# ============================================================
# 7. Visualization helpers
# ============================================================

def evaluate_on_single_sequence(
    model: nn.Module,
    noisy: np.ndarray,
    history_len: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    xs = []
    for i in range(len(noisy) - history_len - 1):
        xs.append(noisy[i:i + history_len])
    x = torch.tensor(np.stack(xs).astype(np.float32), device=device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze(-1)
    return pred


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="demo_outputs")

    parser.add_argument("--history_len", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--total_steps", type=int, default=420)
    parser.add_argument("--noise_std", type=float, default=0.12)

    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--freqs", type=str, default="0.8,3.0,1.2,4.2")
    parser.add_argument("--change_points", type=str, default="0.22,0.48,0.74")

    parser.add_argument("--target_params", type=int, default=1100)
    parser.add_argument("--tau_neurons_to_plot", type=int, default=6)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs = parse_csv_floats(args.freqs)
    change_points = parse_csv_floats(args.change_points)
    assert len(change_points) == len(freqs) - 1, "Need len(change_points) = len(freqs)-1"

    print("Building dataset...")
    x_train, y_train = build_dataset(
        num_sequences=args.train_samples,
        total_steps=args.total_steps,
        history_len=args.history_len,
        dt=args.dt,
        freqs=freqs,
        change_points=change_points,
        noise_std=args.noise_std,
    )
    x_test, y_test = build_dataset(
        num_sequences=args.test_samples,
        total_steps=args.total_steps,
        history_len=args.history_len,
        dt=args.dt,
        freqs=freqs,
        change_points=change_points,
        noise_std=args.noise_std,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_names = ["FNN", "RNN", "LSTM", "LTC"]
    hidden_choices: Dict[str, int] = {}
    param_counts: Dict[str, int] = {}

    print("Matching parameter counts...")
    for name in model_names:
        h, p = match_hidden_size(name, history_len=args.history_len, target_params=args.target_params)
        hidden_choices[name] = h
        param_counts[name] = p
        print(f"  {name:>4s}: hidden={h:>3d}, params={p}")

    results: Dict[str, TrainResult] = {}
    for name in model_names:
        print(f"Training {name}...")
        res = train_model(
            model_name=name,
            hidden_size=hidden_choices[name],
            history_len=args.history_len,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results[name] = res
        print(f"  -> test MSE = {res.test_mse:.6f}")

    # Prepare one demo sequence for qualitative visualization
    noisy_demo, clean_demo, freq_demo, t_demo = generate_single_sequence(
        total_steps=args.total_steps,
        history_len=args.history_len,
        dt=args.dt,
        freqs=freqs,
        change_points=change_points,
        noise_std=args.noise_std,
    )

    preds_demo = {}
    for name in model_names:
        preds_demo[name] = evaluate_on_single_sequence(
            results[name].model,
            noisy_demo,
            history_len=args.history_len,
            device=device,
        )

    # Effective tau for LTC
    x_demo_windows = []
    for i in range(len(noisy_demo) - args.history_len - 1):
        x_demo_windows.append(noisy_demo[i:i + args.history_len])
    x_demo_windows = np.stack(x_demo_windows).astype(np.float32)
    # Use the first rolling window as a representative internal tau trajectory over the window
    x_tau = torch.tensor(x_demo_windows[0:1], device=device)  # [1, history_len, 1]
    tau_eff = extract_ltc_effective_tau(results["LTC"].model, x_tau, device=device)  # [history_len, H]

    # ========================================================
    # Plot 1: comparison + tau
    # ========================================================
    fig = plt.figure(figsize=(15, 12))

    # (a) demo signal and frequency schedule
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t_demo, clean_demo.squeeze(-1), label="Clean signal")
    ax1.plot(t_demo, noisy_demo.squeeze(-1), label="Noisy input", alpha=0.6)
    ax1_t = ax1.twinx()
    ax1_t.plot(t_demo, freq_demo, linestyle="--", alpha=0.6, label="Frequency schedule")
    ax1.set_title("Configurable piecewise-frequency task")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Signal")
    ax1_t.set_ylabel("Frequency")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True)

    # (b) qualitative predictions
    ax2 = plt.subplot(4, 1, 2)
    target_demo = clean_demo[args.history_len + 1:].squeeze(-1)
    t_pred = t_demo[args.history_len + 1:]
    ax2.plot(t_pred, target_demo, label="Target", linewidth=2)
    for name in model_names:
        ax2.plot(t_pred, preds_demo[name], label=name, alpha=0.9)
    ax2.set_title("Rolling one-step predictions on one held-out sequence")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Prediction")
    ax2.legend()
    ax2.grid(True)

    # (c) bar chart for test mse + parameter counts
    ax3 = plt.subplot(4, 1, 3)
    xs = np.arange(len(model_names))
    mses = [results[name].test_mse for name in model_names]
    params = [results[name].n_params for name in model_names]
    bars = ax3.bar(xs, mses)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{name}\n({params[i]} params)" for i, name in enumerate(model_names)])
    ax3.set_ylabel("Test MSE")
    ax3.set_title("Fair comparison: similar parameter counts")
    for b, mse in zip(bars, mses):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{mse:.4f}", ha="center", va="bottom")
    ax3.grid(True, axis="y")

    # (d) LTC tau dynamics
    ax4 = plt.subplot(4, 1, 4)
    H = tau_eff.shape[1]
    n_plot = min(args.tau_neurons_to_plot, H)
    t_tau = np.arange(args.history_len) * args.dt
    for i in range(n_plot):
        ax4.plot(t_tau, tau_eff[:, i], label=f"Neuron {i}")
    ax4.set_title("LTC effective tau dynamics (extracted from official ncps LTCCell)")
    ax4.set_xlabel("Time within one input window")
    ax4.set_ylabel("Effective tau")
    ax4.legend(ncol=3)
    ax4.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "comparison_and_tau.png"), dpi=180)
    plt.close(fig)

    # ========================================================
    # Plot 2: training curves
    # ========================================================
    fig = plt.figure(figsize=(10, 6))
    for name in model_names:
        plt.plot(results[name].train_losses, label=name)
    plt.title("Training curves")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=180)
    plt.close(fig)

    summary = {
        "task": {
            "history_len": args.history_len,
            "dt": args.dt,
            "total_steps": args.total_steps,
            "noise_std": args.noise_std,
            "freqs": freqs,
            "change_points": change_points,
        },
        "results": {
            name: {
                "hidden_size": results[name].hidden_size,
                "n_params": results[name].n_params,
                "test_mse": results[name].test_mse,
            }
            for name in model_names
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Saved figures and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()

