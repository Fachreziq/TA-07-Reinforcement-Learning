"""
TA-07 Pembelajaran Mesin - Implementasi Q-Learning untuk Skenario RL Diskrit
Skenario (dari TP-07): Kurir Antar Paket di Peta Kota Kecil (Grid 4x4)

Nama : Fadhly Muhammad Fachrezi | NIM : 301240001 | Kelas : 5A

Jalankan: python ta07_kurir_qlearning.py
(Kompatibel dengan Jupyter / Google Colab: salin per bagian ke sel notebook.)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =====================================================================
# 1. PERSIAPAN LINGKUNGAN (Environment Setup) - sesuai TP-07
# =====================================================================
GRID = 4
WAREHOUSE = (0, 0)           # gudang (baris, kolom)
CUSTOMER = (3, 3)            # rumah pelanggan
BLOCKED = {(1, 1), (2, 2)}   # jalan tertutup

ACTION_NAMES = ["Atas", "Bawah", "Kiri", "Kanan", "Ambil", "Antar"]
ACTION_SYMBOLS = ["↑", "↓", "←", "→", "P", "D"]
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# Reward function (TP-07)
R_STEP = -1       # setiap langkah biasa
R_WALL = -5       # menabrak tembok / petak tertutup (posisi tidak berubah)
R_ILLEGAL = -10   # ambil di luar gudang / ambil saat sudah bawa / antar di luar rumah
R_PICKUP = 20     # berhasil mengambil paket di gudang
R_DELIVER = 100   # berhasil mengantar paket di rumah pelanggan (terminal)


class CourierEnv:
    """State = (baris, kolom, status_paket) -> 16 x 2 = 32 state (28 reachable).
    Status paket: 0 = belum diambil, 1 = sedang dibawa."""
    n_states = GRID * GRID * 2
    n_actions = 6

    @staticmethod
    def encode(r, c, p):
        return (r * GRID + c) * 2 + p

    @staticmethod
    def decode(s):
        p = s % 2
        rc = s // 2
        return rc // GRID, rc % GRID, p

    def reset(self):
        return self.encode(*WAREHOUSE, 0)

    def step(self, state, action):
        """Menerima (state, action) -> mengembalikan (new_state, reward, done)."""
        r, c, p = self.decode(state)
        if action in MOVES:
            dr, dc = MOVES[action]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID and 0 <= nc < GRID) or (nr, nc) in BLOCKED:
                return self.encode(r, c, p), R_WALL, False
            return self.encode(nr, nc, p), R_STEP, False
        if action == 4:  # ambil paket
            if (r, c) == WAREHOUSE and p == 0:
                return self.encode(r, c, 1), R_PICKUP, False
            return state, R_ILLEGAL, False
        if action == 5:  # antar paket
            if (r, c) == CUSTOMER and p == 1:
                return self.encode(r, c, 0), R_DELIVER, True
            return state, R_ILLEGAL, False
        raise ValueError("Aksi tidak dikenal")


# =====================================================================
# 2. IMPLEMENTASI Q-LEARNING
# =====================================================================
def greedy_action(q_row, rng):
    """argmax dengan pemecah seri acak (agar tidak bias ke aksi indeks 0)."""
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train(alpha=0.8, gamma=0.9, episodes=2000, max_steps=50,
          eps_start=1.0, eps_min=0.01, decay_rate=0.005, seed=42):
    env = CourierEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))          # Inisialisasi Q-Table = 0
    rewards, steps_log, success, eps_log = [], [], [], []
    epsilon = eps_start

    for ep in range(episodes):
        s = env.reset()
        total, done = 0, False
        for t in range(max_steps):
            # epsilon-greedy
            if rng.random() < epsilon:
                a = int(rng.integers(env.n_actions))      # eksplorasi
            else:
                a = greedy_action(q[s], rng)              # eksploitasi
            s2, r, done = env.step(s, a)
            # Persamaan Bellman: Q <- (1-a)Q + a[r + g max Q(s',a')]
            target = r if done else r + gamma * np.max(q[s2])
            q[s, a] = (1 - alpha) * q[s, a] + alpha * target
            s, total = s2, total + r
            if done:
                break
        rewards.append(total)
        steps_log.append(t + 1)
        success.append(int(done))
        eps_log.append(epsilon)
        # Peluruhan epsilon (eksponensial)
        epsilon = eps_min + (eps_start - eps_min) * np.exp(-decay_rate * ep)
    return q, dict(rewards=rewards, steps=steps_log, success=success, epsilon=eps_log)


def evaluate_greedy(q, max_steps=50, seed=0):
    """Jalankan kebijakan greedy murni dari start; kembalikan lintasan & total reward."""
    env = CourierEnv()
    rng = np.random.default_rng(seed)
    s, total, path, acts = env.reset(), 0, [], []
    path.append(env.decode(s))
    for _ in range(max_steps):
        a = greedy_action(q[s], rng)
        s, r, done = env.step(s, a)
        total += r
        acts.append(ACTION_NAMES[a])
        path.append(env.decode(s))
        if done:
            break
    return total, path, acts, done


# =====================================================================
# 3. ANALISIS KONVERGENSI
# =====================================================================
def moving_average(x, window=100):
    return pd.Series(x).rolling(window=window, min_periods=1).mean()


def convergence_episode(ma, tol=0.05, tail=200):
    """Episode pertama setelah mana MA selalu berada dalam +-tol dari rata-rata `tail` episode akhir."""
    final = np.mean(ma[-tail:])
    ok = np.abs(np.asarray(ma) - final) <= tol * abs(final)
    for i in range(len(ok)):
        if ok[i:].all():
            return i, final
    return None, final


def plot_learning_curve(hist, window=100, fname="kurva_pembelajaran.png"):
    ma = moving_average(hist["rewards"], window)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist["rewards"], color="lightgray", lw=0.6, label="Reward per episode")
    ax.plot(ma, color="green", lw=2, label=f"Moving average ({window} episode)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.set_title(f"Kurva Pembelajaran Q-Learning Kurir (Rata-Rata {window} Episode)")
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(hist["epsilon"], color="tab:red", ls="--", lw=1.2, label="Epsilon")
    ax2.set_ylabel("Epsilon"); ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_ylim(min(ma) - 15, 130)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.show()


# =====================================================================
# 4. POLICY
# =====================================================================
def plot_policy(q, fname="kebijakan_optimal.png"):
    env = CourierEnv()
    titles = ["Status 0: belum membawa paket", "Status 1: sedang membawa paket"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for p, ax in enumerate(axes):
        ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.set_aspect("equal")
        for r in range(GRID):
            for c in range(GRID):
                if (r, c) in BLOCKED:
                    face, txt = "#444444", "X"
                else:
                    s = env.encode(r, c, p)
                    a = int(np.argmax(q[s]))
                    face = cmap(a) if q[s].any() else "white"
                    txt = ACTION_SYMBOLS[a] if q[s].any() else "?"
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="black", alpha=0.75))
                ax.text(c + .5, r + .5, txt, ha="center", va="center", fontsize=24,
                        color="white", fontweight="bold")
        ax.text(0.06, 0.93, "Gudang", fontsize=8, color="white")
        ax.text(3.06, 3.93, "Rumah", fontsize=8, color="white")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[p])
    fig.suptitle("Kebijakan Optimal (argmax Q) - ↑↓←→ = bergerak, P = ambil, D = antar, X = jalan tertutup")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(fname, dpi=150); plt.show()


def plot_value_heatmap(q, fname="heatmap_qmax.png"):
    env = CourierEnv()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for p, ax in enumerate(axes):
        grid = np.full((GRID, GRID), np.nan)
        for r in range(GRID):
            for c in range(GRID):
                if (r, c) not in BLOCKED:
                    grid[r, c] = q[env.encode(r, c, p)].max()
        im = ax.imshow(grid, cmap="viridis")
        for r in range(GRID):
            for c in range(GRID):
                t = "X" if (r, c) in BLOCKED else f"{grid[r, c]:.1f}"
                ax.text(c, r, t, ha="center", va="center", color="white", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(["max Q - belum membawa paket", "max Q - membawa paket"][p])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.show()


def q_table_dataframe(q):
    env = CourierEnv()
    rows = []
    for s in range(env.n_states):
        r, c, p = env.decode(s)
        rows.append([s, f"({r},{c})", "Belum" if p == 0 else "Dibawa"] + list(np.round(q[s], 2)))
    return pd.DataFrame(rows, columns=["State", "Posisi", "Paket"] + ACTION_NAMES)


if __name__ == "__main__":
    q, hist = train(alpha=0.8, gamma=0.9, episodes=2000, decay_rate=0.005, seed=42)
    ma = moving_average(hist["rewards"], 100)
    ep_conv, final = convergence_episode(ma.values)
    total, path, acts, ok = evaluate_greedy(q)
    print(f"Episode konvergensi (MA100 stabil +-5%): {ep_conv}, plateau = {final:.2f}")
    print(f"Reward greedy: {total}, berhasil: {ok}, langkah: {len(acts)}")
    print("Lintasan:", path); print("Aksi:", acts)
    plot_learning_curve(hist); plot_policy(q); plot_value_heatmap(q)
    q_table_dataframe(q).to_csv("q_table_akhir.csv", index=False)
    print(q_table_dataframe(q).to_string(index=False))
