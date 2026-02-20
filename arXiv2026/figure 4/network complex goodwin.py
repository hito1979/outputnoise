import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# 1) Define network (EQS) exactly as your structure
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

names = [n for n, _ in EQS]  # all output nodes

# ============================================================
# 2) Define layers + colors + RNG (these were missing)
# ============================================================
def infer_layer(node: str) -> int:
    if node == "w":
        return 0
    m = re.match(r"x(\d+)_", node)
    if not m:
        raise ValueError(f"Cannot infer layer for node: {node}")
    return int(m.group(1))

layers = {n: infer_layer(n) for n in names}
rng = np.random.default_rng(0)

clock_color = "blue"
layer_colors = {
    1: "orange",  # blue-ish (layer 1)
    2: "green",  # orange (layer 2)
    3: "magenta",  # green (layer 3)
    4: "red",  # red (layer 4)
}

# ============================================================
# 3) FIXED LAYER LAYOUT (use layers, NOT distances)
# ============================================================
nodes_by_layer = {L: [] for L in [1, 2, 3, 4]}
for n in names:
    L = layers[n]
    if L in nodes_by_layer:
        nodes_by_layer[L].append(n)

# preserve your generation order (no sorting)
for L in [1, 2, 3, 4]:
    nodes_by_layer[L] = list(nodes_by_layer[L])

pos = {}
pos["w"] = (0.0, 1.2)

layer_gap = 1.8

# NOTE: layer 4 has 64 nodes -> need smaller x gap or it becomes insanely wide
x_gap_by_layer = {1: 1.2, 2: 1.2, 3: 0.7, 4: 3.0}

for L in [1, 2, 3, 4]:
    layer_nodes = nodes_by_layer[L]
    nL = len(layer_nodes)

    x_gap = x_gap_by_layer[L]
    xs = (np.arange(nL) - (nL - 1) / 2) * x_gap
    y  = 1.2 - L * layer_gap

    # tiny jitter only to reduce exact overlaps
    yjit = rng.normal(0, 0.05, size=nL)

    for i, node in enumerate(layer_nodes):
        pos[node] = (xs[i], y + yjit[i])
# ============================================================
# Compute shortest distance to clock (w)
# ============================================================
deps = {n: ins for n, ins in EQS}

memo_dist = {"w": 0}
def distance_to_w(node: str) -> int:
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

distance_markers = {
    1: "o",  # circle
    2: "s",  # square
    3: "^",  # triangle
    4: "x",  # cross
}

# ============================================================
# 4) PLOT (arrows unchanged)
# ============================================================
# ============================================================
# 6C. PLOT 3: NETWORK STRUCTURE (arrows, FIXED by LAYER, color by LAYER)
# ============================================================

nodes_by_layer = {L: [] for L in [1, 2, 3, 4]}
for n in names:
    L = layers[n]
    nodes_by_layer[L].append(n)

# stable order
for L in [1, 2, 3, 4]:
    nodes_by_layer[L] = sorted(nodes_by_layer[L])

pos = {"w": (0.0, 1.2)}

layer_gap = 1.8

# spacing per layer (layer 4 will be wrapped, so it can be wider spacing than before)
x_gap_by_layer = {1: 300.0, 2: 600.0, 3: 200.0, 4:100.0}

# --- place layers 1-3 in one row each ---
for L in [1, 2, 3]:
    layer_nodes = nodes_by_layer[L]
    nL = len(layer_nodes)

    x_gap = x_gap_by_layer[L]
    xs = (np.arange(nL) - (nL - 1) / 2) * x_gap
    y  = 1.2 - L * layer_gap

    yjit = rng.normal(0, 0.05, size=nL)
    for i, node in enumerate(layer_nodes):
        pos[node] = (xs[i], y + yjit[i])

# --- place layer 4 in 3 wrapped rows ---
L = 4
layer4_nodes = nodes_by_layer[4]
n4 = len(layer4_nodes)

n_rows4 = 4
rows = np.array_split(layer4_nodes, n_rows4)

y_base = 1.2 - L * layer_gap
row_offsets = np.linspace(+0.9, -0.9, n_rows4)  # spread 3 rows within "layer 4"

x_gap4 = x_gap_by_layer[4]
for r, row_nodes in enumerate(rows):
    nr = len(row_nodes)
    xs = (np.arange(nr) - (nr - 1) / 2) * x_gap4
    y  = y_base + row_offsets[r]

    yjit = rng.normal(0, 0.04, size=nr)
    for i, node in enumerate(row_nodes):
        pos[node] = (xs[i], y + yjit[i])

# ---- Plot ----
plt.figure(figsize=(7, 7))   # wider because layer 4 is huge
ax = plt.gca()

def draw_arrow(p0, p1, color="black", alpha=0.6):
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=0.8,
        color=color,
        alpha=alpha,
        shrinkA=10,
        shrinkB=10
    )
    ax.add_patch(arrow)

# Draw arrows first
for target, inputs in EQS:
    for src in inputs:
        p0 = pos["w"] if src == "w" else pos[src]
        p1 = pos[target]
        draw_arrow(p0, p1)

# Draw nodes
# Draw nodes
# clock (w) as STAR
ax.scatter(
    pos["w"][0], pos["w"][1],
    s=500,
    color=clock_color,
    marker="*",
    edgecolor="none",
    linewidths=0.8,
    zorder=6
)

# outputs: color by layer, marker by distance
for n in names:
    L = layers[n]                 # color = layer
    d = distances[n]
    d_plot = d if d <= 3 else 4   # distance 4+ => 'x'
    m = distance_markers[d_plot]

    if m == "x":
        # 'x' marker: linewidths matters, no facecolor
        ax.scatter(
            pos[n][0], pos[n][1],
            s=500,
            marker="x",
            color=layer_colors[L],
            linewidths=2.5,
            zorder=5
        )
    else:
        ax.scatter(
            pos[n][0], pos[n][1],
            s=500,
            marker=m,
            color=layer_colors[L],
            edgecolor="none",
            zorder=5
        )

ax.set_title("Network structure (fixed by layer, arrows unchanged)")
ax.axis("off")
plt.tight_layout()
plt.savefig("fig4network.pdf", dpi=300)
plt.show()

