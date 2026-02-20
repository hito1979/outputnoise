import numpy as np
import matplotlib.pyplot as plt

# ===== Data =====
cv_0edge = np.array([0.1586, 0.1589, 0.1587, 0.1591, 0.1597, 0.1594, 0.1596, 0.1594])
cv_1edge = np.array([0.1498, 0.1501, 0.1502, 0.1503, 0.1506, 0.1505, 0.1507, 0.1504,
                     0.1497, 0.1499,0.15029, 0.1501,0.1506, 0.1501, 0.15034, 0.1497])

cv_2edge = np.array([0.1442, 0.1445, 0.1451, 0.1450, 0.1442, 0.1443, 0.1446])

cv_3edge = np.array([0.1413])   # Single value

# ===== X positions =====
clock = np.full(len(cv_0edge), 0.0)
x_1 = np.full(len(cv_1edge), 1)
x_2 = np.full(len(cv_2edge), 2)
x_3 = np.full(len(cv_3edge), 3)

# ===== Jitter (horizontal) =====
rng = np.random.default_rng(0)     # reproducible
jitter_strength = 0.3             # try 0.05 ~ 0.15

clockj = clock + rng.uniform(-jitter_strength, jitter_strength, size=len(clock))
x_1j = x_1 + rng.uniform(-jitter_strength, jitter_strength, size=len(x_1))
x_2j = x_2 + rng.uniform(-jitter_strength, jitter_strength, size=len(x_2))
x_3j = x_3 + rng.uniform(-jitter_strength, jitter_strength, size=len(x_3))

# ===== Plot =====
plt.figure(figsize=(7, 7))
plt.scatter(clockj, cv_0edge, marker='*', color="black", s=500, label='0 edge', alpha=0.75)
plt.scatter(x_1j, cv_1edge, marker='o', color="black", s=500, label='1 edge', alpha=0.40)
plt.scatter(x_2j, cv_2edge, marker='s', color="black", s=500, label='2 edges', alpha=0.60)
plt.scatter(x_3j, cv_3edge, marker='^', color="black", s=500, label='3 edges', alpha=0.75)

plt.xticks([0, 1, 2, 3])
plt.yscale('log')
plt.ylim(0.13, 0.17)
plt.xlabel("Shortest edge distance from clock")
plt.ylabel("CV")
plt.title("CV vs Topological Distance")
#plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("phase edge.pdf", dpi=300)
plt.show()
