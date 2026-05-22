import numpy as np
from scipy.signal import argrelmax
import matplotlib.pyplot as plt


def run_simulation(a, b, c, k, D=0, num_steps=80000000, measure_start=500000):
    dt = 0.0001
    epsilon = 0.01

    v = w = 0.1
    x1 = x2 = x3 = 0

    noise = np.random.normal(0, 1, size=num_steps)

    w_list  = np.zeros(num_steps)
    x1_list = np.zeros(num_steps)
    x2_list = np.zeros(num_steps)
    x3_list = np.zeros(num_steps)

    for t in range(num_steps):
        dv = (v*(a - v)*(v - 1) - w) * dt + np.sqrt(epsilon * D * dt) * noise[t]
        dw = (b * v - c * w) * dt

        dx1 = (1 + w  - k*x1) * dt
        dx2 = (1 + x1 - k*x2) * dt
        dx3 = (1 + x2 - k*x3) * dt

        v += dv
        w += dw
        x1 += dx1
        x2 += dx2
        x3 += dx3

        w_list[t]  = w
        x1_list[t] = x1
        x2_list[t] = x2
        x3_list[t] = x3

    # --- Cut off transient ---
    w_meas  = w_list[measure_start:]
    x1_meas = x1_list[measure_start:]
    x2_meas = x2_list[measure_start:]
    x3_meas = x3_list[measure_start:]

    # --- Helper: compute CV of period ---
    def calc_cv(signal, dt=0.0001, order=100):
        peaks = argrelmax(signal, order=order)[0]
        if len(peaks) < 2:
            return np.nan
        periods = np.diff(peaks) * dt
        m, s = np.mean(periods), np.std(periods)
        return (s / m)*100 if m > 0 else np.nan

    return calc_cv(w_meas), calc_cv(x1_meas), calc_cv(x2_meas), calc_cv(x3_meas)

# --- Sweep decay parameter k ---
k_values = np.logspace(-1, 1, 21)
n_repeat = 100

a = 0.4
b = 1
c = -0.5
D = 0.00001

# Collect means and stds for error bars
mean_w, mean_x1, mean_x2, mean_x3 = [], [], [], []
std_w,  std_x1,  std_x2,  std_x3  = [], [], [], []
n_w, n_x1, n_x2, n_x3 = [], [], [], []  # counts used (after NaN filtering)

def filter_outliers(cvs, threshold=1.5):
    """Filter out outliers based on a given threshold"""
    mean_cv = np.mean(cvs)
    std_cv = np.std(cvs)
    filtered_cvs = [cv for cv in cvs if abs(cv - mean_cv) <= threshold * std_cv]
    return filtered_cvs

for k in k_values:
    w_runs, x1_runs, x2_runs, x3_runs = [], [], [], []
    for _ in range(n_repeat):
        cv_w, cv_x1, cv_x2, cv_x3 = run_simulation(a, b, c, k, D=D)
        if not np.isnan(cv_w):  w_runs.append(cv_w)
        if not np.isnan(cv_x1): x1_runs.append(cv_x1)
        if not np.isnan(cv_x2): x2_runs.append(cv_x2)
        if not np.isnan(cv_x3): x3_runs.append(cv_x3)

    # Apply outlier filtering to each of the variables
    filtered_w = filter_outliers(w_runs)
    filtered_x1 = filter_outliers(x1_runs)
    filtered_x2 = filter_outliers(x2_runs)
    filtered_x3 = filter_outliers(x3_runs)

    # function to compute mean/std safely
    def m_s(arr):
        if len(arr) == 0:
            return np.nan, np.nan, 0
        return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), len(arr)

    mW, sW, nW = m_s(filtered_w)
    m1, s1, n1 = m_s(filtered_x1)
    m2, s2, n2 = m_s(filtered_x2)
    m3, s3, n3 = m_s(filtered_x3)

    mean_w.append(mW); std_w.append(sW); n_w.append(nW)
    mean_x1.append(m1); std_x1.append(s1); n_x1.append(n1)
    mean_x2.append(m2); std_x2.append(s2); n_x2.append(n2)
    mean_x3.append(m3); std_x3.append(s3); n_x3.append(n3)

    print(f"k={k:.2f} | n=[w:{nW}, x1:{n1}, x2:{n2}, x3:{n3}] | "
          f"CV(w)={mW:.4f}±{sW:.4f}, CV(x1)={m1:.4f}±{s1:.4f}, CV(x2)={m2:.4f}±{s2:.4f}, CV(x3)={m3:.4f}±{s3:.4f}")
          

mean_w  = np.array(mean_w);  std_w  = np.array(std_w)
mean_x1 = np.array(mean_x1); std_x1 = np.array(std_x1)
mean_x2 = np.array(mean_x2); std_x2 = np.array(std_x2)
mean_x3 = np.array(mean_x3); std_x3 = np.array(std_x3)

# --- Plot with error bars (±1 s.d.) ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xscale('log')

ax.errorbar(k_values, mean_w,  yerr=std_w,  fmt='o', capsize=3, elinewidth=1.2, label='w')
ax.errorbar(k_values, mean_x1, yerr=std_x1, fmt='o', capsize=3, elinewidth=1.2, label='x₁')
ax.errorbar(k_values, mean_x2, yerr=std_x2, fmt='o', capsize=3, elinewidth=1.2, label='x₂')
ax.errorbar(k_values, mean_x3, yerr=std_x3, fmt='o', capsize=3, elinewidth=1.2, label='x₃')

ax.set_xlabel("Decay rate k")
ax.set_ylabel("Coefficient of variation (CV)")
ax.set_title("Oscillation precision vs decay rate (mean ± 1 s.d., n repeats)")
#ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()

