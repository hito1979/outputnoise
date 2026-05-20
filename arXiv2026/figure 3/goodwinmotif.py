import numpy as np
import matplotlib.pyplot as plt

# ----- Motif numbers -----
motifs = np.arange(1, 9)

# ----- Mean values only (no SD used in plotting) -----
mean_w  = np.array([0.502, 0.502, 0.501, 0.504, 0.503, 0.500, 0.500, 0.501])
mean_x1 = np.array([0.486, 0.487, 0.486, 0.488, 0.489, 0.485, 0.485, 0.486])
mean_x2 = np.array([0.478, 0.479, 0.485, 0.487, 0.480, 0.477, 0.484, 0.485])
mean_x3 = np.array([0.474, 0.479, 0.477, 0.479, 0.488, 0.484, 0.484, 0.485])

# ----- Plot (no error bars) -----
plt.figure(figsize=(7, 7))

motifs = np.asarray(motifs)

offset = 0.12  # spacing between series (tune)
plt.plot(motifs - 1.5*offset, mean_w,  'o', markersize=20, label='w')
plt.plot(motifs - 0.5*offset, mean_x1, 'o', markersize=20, label='x1')
plt.plot(motifs + 0.5*offset, mean_x2, 'o', markersize=20, label='x2')
plt.plot(motifs + 1.5*offset, mean_x3, 'o', markersize=20, label='x3')

plt.yscale('log')
plt.ylim(0.47, 0.51)
plt.xlabel("Motif")
plt.ylabel("CV (mean)")
plt.title("CV vs Motif (K = 12.59)")
plt.xticks(motifs)
#plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("goodwin network motifs.pdf", dpi=300)
plt.show()
