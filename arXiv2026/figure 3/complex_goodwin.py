import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from matplotlib.patches import FancyArrowPatch

# ============================================================
# 1. Define network (EQS) exactly as your structure
# ============================================================

EQS = []

# ---- Layer 1 ----
EQS += [
    ("x1_1", ["w"]),
    ("x1_2", ["w"]),
]

# ---- Layer 2 ----
EQS += [
    ("x2_1", ["x1_1"]),
    ("x2_2", ["w", "x1_2"]),
]

# ---- Layer 3 ----
EQS += [
    ("x3_1", ["x2_1"]),
    ("x3_2", ["w", "x2_1"]),
    ("x3_3", ["x1_2", "x2_1"]),
    ("x3_4", ["w", "x1_2", "x2_1"]),
    ("x3_5", ["x2_2"]),
    ("x3_6", ["w", "x2_2"]),
    ("x3_7", ["x1_2", "x2_2"]),
    ("x3_8", ["w", "x1_2", "x2_2"]),
]

# ---- Layer 4 (8 combos for each x3_*) ----
parents_L3 = [n for n, _ in EQS if n.startswith("x3")]
count = 1
for p in parents_L3:
    for combo in [
        [p],
        ["w", p],
        ["x1_2", p],
        ["x2_2", p],
        ["w", "x1_2", p],
        ["w", "x2_2", p],
        ["x1_2", "x2_2", p],
        ["w", "x1_2", "x2_2", p],
    ]:
        EQS.append((f"x4_{count}", combo))
        count += 1

names = [n for n, _ in EQS]  # all outputs (x1..x4)
deps  = {n: ins for n, ins in EQS}

# ============================================================
# 2. Helpers: layer and correct distance-to-w (back-tracking)
# ============================================================

def get_layer(node_name: str) -> int:
    # node like "x3_7" -> layer 3
    return int(node_name[1])

memo_dist = {"w": 0}

def distance_to_w(node):
    if node in memo_dist:
        return memo_dist[node]
    inputs = deps[node]
    if "w" in inputs:
        memo_dist[node] = 1
        return 1
    d = min(distance_to_w(inp) for inp in inputs) + 1
    memo_dist[node] = d
    return d

distances = {n: distance_to_w(n) for n in names}
layers    = {n: get_layer(n) for n in names}

# ============================================================
# 3. Goodwin clock + downstream network simulation
# ============================================================

def run_simulation(
    k=1.0, K=1.0, d=10.0, T=39.7,
    D=1e-5,
    num_steps=3_000_000,
    measure_start=500_000,
    dt=1e-4
):
    epsilon = 0.01
    m = 10

    u = v = w = 0.1
    x = {n: 0.0 for n in names}

    noise = np.random.randn(num_steps)

    w_traj = np.zeros(num_steps - measure_start)
    traj = {n: np.zeros(num_steps - measure_start) for n in names}

    for t in range(num_steps):
        # Goodwin oscillator
        du = T*(k/(K + w**m) - 0.1*u)*dt + np.sqrt(epsilon*D*dt)*noise[t]
        dv = T*(k*u - 0.1*v)*dt
        dw = T*(k*v - 0.1*w)*dt
        u += du; v += dv; w += dw

        # outputs (Euler)
        x_new = {}
        for n, inputs in EQS:
            s = 1.0
            for inp in inputs:
                s += w if inp == "w" else x[inp]
            x_new[n] = x[n] + (s - d*x[n]) * dt
        x = x_new

        if t >= measure_start:
            idx = t - measure_start
            w_traj[idx] = w
            for n in names:
                traj[n][idx] = x[n]

    return w_traj, traj, dt

# ============================================================
# 4. CV computation
# ============================================================

def calc_cv(signal, dt=1e-4, order=300):
    peaks = argrelmax(signal, order=order)[0]
    if len(peaks) < 2:
        return np.nan
    periods = np.diff(peaks) * dt
    if np.mean(periods) <= 0:
        return np.nan
    return 100 * np.std(periods) / np.mean(periods)

# ============================================================
# 5. Run one simulation + compute CVs
# ============================================================

w_sig, x_sig, dt = run_simulation()

CV = {"w": calc_cv(w_sig, dt=dt)}
for n in names:
    CV[n] = calc_cv(x_sig[n], dt=dt)

# ============================================================
# 6A. PLOT 1 (REPLACED): CV vs NETWORK (1-column) — COLOR BY LAYER
# ============================================================

layer_colors = {
    1: "orange",
    2: "green",
    3: "magenta",
    4: "red",
}
clock_color = "blue"

rng = np.random.default_rng(0)

# --- group nodes by layer ---
nodes_by_layer = {L: [n for n in names if layers[n] == L] for L in [1,2,3,4]}

plt.figure(figsize=(7, 7))  # narrow figure => not wide/spread

x_jitter = 0.03   # small horizontal jitter to avoid overlap
y_jitter = 0.0    # keep y exact (CV)

# plot outputs: all centered around x=0 (one column)
for L in [1, 2, 3, 4]:
    for n in nodes_by_layer[L]:
        x = rng.normal(0.0, x_jitter)
        y = CV[n] + rng.normal(0.0, y_jitter)
        plt.scatter(
            x, y,
            color=layer_colors[L],
            s=500,
            marker="o",
            alpha=0.85,
            edgecolor="none"
        )

# plot clock at exact center
plt.scatter(
    0.0, CV["w"],
    color=clock_color,
    s=500,
    marker="o",
    edgecolor="none",
    label="clock (w)",
    zorder=5
)

plt.ylabel("CV of period (%)")
plt.xlabel("Network (1-column layout)")
plt.title("CV spread across network (color = layer)")
#plt.grid(alpha=0.25)

# force it to look like one column
plt.xlim(-0.09, 0.09)
plt.xticks([])  # optional: hide meaningless x ticks

plt.tight_layout()
plt.savefig("fig4CV.pdf", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 6B. PLOT 2: CV vs DISTANCE (WITH JITTER) — COLOR BY DISTANCE (red shades)
# ============================================================

distance_colors = {
    1: "black",
    2: "black",
    3: "black",
    4: "black",
}

rng = np.random.default_rng(0)
jitter_strength = 0.15

plt.figure(figsize=(7,7))

for n in names:
    d = distances[n]
    if d <= 4:
        x_jitter = d + rng.uniform(-jitter_strength, jitter_strength)
        plt.scatter(
            x_jitter, CV[n],
            marker="x",
            s=500,
            linewidths=2,
            color=distance_colors[d],
            alpha=0.85
        )

plt.xticks([1,2,3,4])
plt.xlabel("Clock distance")
plt.ylabel("CV of period (%)")
plt.title("CV vs Clock Distance (Goodwin model)")
#plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig4cd.pdf", dpi=300)
plt.show()

# ============================================================
# 6C. PLOT 3: NETWORK STRUCTURE (arrows, spread, color by LAYER)
# ============================================================

# Simple layered layout:
# y by distance: w at top, distance1 below, ...
# x spread within each distance by ordering

# Collect nodes by distance
nodes_by_dist = {d: [] for d in [1,2,3,4]}
for n in names:
    d = min(distances[n], 4)
    nodes_by_dist[d].append(n)

# Sort for stable layout
for d in [1,2,3,4]:
    nodes_by_dist[d] = sorted(nodes_by_dist[d])

pos = {}
pos["w"] = (0.0, 1.2)

layer_gap = 1.8
x_gap = 0.7

for d in [1,2,3,4]:
    layer_nodes = nodes_by_dist[d]
    nL = len(layer_nodes)
    xs = (np.arange(nL) - (nL-1)/2) * x_gap
    y  = 1.2 - d*layer_gap
    # add tiny y jitter so arrows/nodes overlap less
    yjit = rng.normal(0, 0.08, size=nL)
    for i, node in enumerate(layer_nodes):
        pos[node] = (xs[i], y + yjit[i])

plt.figure(figsize=(7, 7))
ax = plt.gca()

# Draw arrows first (so nodes appear on top)
def draw_arrow(p0, p1, color="0.4", alpha=0.6):
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.8,
        color=color,
        alpha=alpha,
        shrinkA=10,  # makes arrow touch node edge instead of center
        shrinkB=10
    )
    ax.add_patch(arrow)

for target, inputs in EQS:
    for src in inputs:
        p0 = pos["w"] if src == "w" else pos[src]
        p1 = pos[target]
        draw_arrow(p0, p1)

# Draw nodes
# clock
ax.scatter(pos["w"][0], pos["w"][1], s=220, color=clock_color, edgecolor="black", zorder=3)

# outputs
for n in names:
    L = layers[n]  # color by layer
    ax.scatter(pos[n][0], pos[n][1], s=220, color=layer_colors[L], edgecolor="none", zorder=3)

ax.set_title("Network structure (color = layer, arrows = direction)")
ax.axis("off")
plt.tight_layout()
plt.savefig("Network_structure_layercolor_arrows.pdf", dpi=300)
plt.show()

# ============================================================
# 6A. PLOT 4 (UPDATED): CV vs NETWORK (1-column)
#     color = layer, marker = clock distance
# ============================================================

layer_colors = {
    1: "orange",
    2: "green",
    3: "magenta",
    4: "red",
}
clock_color = "blue"

# marker by distance
distance_markers = {
    1: "o",  # circle
    2: "s",  # square
    3: "^",  # triangle
    4: "x",  # x
}

rng = np.random.default_rng(0)

# --- group nodes by layer ---
nodes_by_layer = {L: [n for n in names if layers[n] == L] for L in [1,2,3,4]}

plt.figure(figsize=(9, 7))

x_jitter = 0.03   # horizontal jitter only
y_jitter = 0.0

for L in [1, 2, 3, 4]:
    for n in nodes_by_layer[L]:
        d = min(distances[n], 4)  # cap at 4 just in case
        m = distance_markers[d]

        x = rng.normal(0.0, x_jitter)
        y = CV[n] + rng.normal(0.0, y_jitter)

        if m == "x":
            # 'x' marker uses 'color' (no facecolor), and linewidths matters
            plt.scatter(
                x, y,
                marker="x",
                color=layer_colors[L],
                s=500,
                linewidths=3,
                alpha=0.85
            )
        else:
            plt.scatter(
                x, y,
                marker=m,
                color=layer_colors[L],
                s=500,
                alpha=0.85,
                edgecolor="none"
            )

# plot clock at exact center
plt.scatter(
    0.0, CV["w"],
    color=clock_color,
    s=500,
    marker="o",
    edgecolor="none",
    label="clock (w)",
    zorder=5
)

plt.ylabel("CV of period (%)")
plt.xlabel("Network (1-column layout)")
plt.title("CV spread across network (color = layer, marker = distance)")
plt.xlim(-0.09, 0.09)
plt.xticks([])

# optional legend for distance markers (clean, no extra points on plot)
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=200, label="distance 1"),
    Line2D([0], [0], marker="s", color="k", linestyle="None", markersize=200, label="distance 2"),
    Line2D([0], [0], marker="^", color="k", linestyle="None", markersize=200, label="distance 3"),
    Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=200, label="distance 4"),
]
#plt.legend(handles=legend_elems, frameon=False, loc="best")

plt.tight_layout()
plt.savefig("fig4CVdistance.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. Sanity prints
# ============================================================

print("Counts by layer:")
for L in [1,2,3,4]:
    print(f"  layer {L}: {sum(1 for n in names if layers[n]==L)} nodes")

print("Counts by distance:")
for d in [1,2,3,4]:
    print(f"  distance {d}: {sum(1 for n in names if distances[n]==d)} nodes")
