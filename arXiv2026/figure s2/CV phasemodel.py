import numpy as np
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import time, os

# --- Simulation function (theta-driven phase model) ---
def run_simulation_phase(d, D=0.001, K=2, a=1.0, b=1.0, num_steps=15_000_000, measure_start=1_000_000):
    np.random.seed(int(time.time()) + os.getpid())

    dt = 1e-4
    theta = 0.0
    x1 = x2 = x3 = 0.0

    sqrt_D_dt = np.sqrt(D * dt)
    noise = np.random.normal(0, 1, size=num_steps)

    theta_list = np.zeros(num_steps)
    x1_list = np.zeros(num_steps)
    x2_list = np.zeros(num_steps)
    x3_list = np.zeros(num_steps)

    for i in range(num_steps):
        # phase clock (your exact form)
        theta += 2 * np.pi * dt + sqrt_D_dt * noise[i]
        theta %= 2 * np.pi

        # downstream
        x1 += (a + b * np.sin(theta) - d * x1) * dt
        x2 += (a + b * x1         - d * x2) * dt
        x3 += (a + b * x2         - d * x3) * dt

        theta_list[i] = np.sin(theta)
        x1_list[i] = x1
        x2_list[i] = x2
        x3_list[i] = x3

    # cut off transient
    th_meas = theta_list[measure_start:]
    x1_meas = x1_list[measure_start:]
    x2_meas = x2_list[measure_start:]
    x3_meas = x3_list[measure_start:]

    # CV helper
    def calc_cv(signal, dt=1e-4, order=100):
        peaks = argrelmax(signal, order=order)[0]
        if len(peaks) < 2:
            return np.nan
        periods = np.diff(peaks) * dt
        m, s = np.mean(periods), np.std(periods)
        return (s / m) * 100 if m > 0 else np.nan

    return calc_cv(th_meas), calc_cv(x1_meas), calc_cv(x2_meas), calc_cv(x3_meas)


# --- Sweep decay parameter d ---
d_values = np.logspace(0, 2, 21)  # 1 to 100
n_repeat = 5

# phase params (you can tune these)
D = 0.0001
a = 1.0
b = 11.3

# Collect means and stds
mean_th, mean_x1, mean_x2, mean_x3 = [], [], [], []
std_th,  std_x1,  std_x2,  std_x3  = [], [], [], []
n_th, n_x1, n_x2, n_x3 = [], [], [], []

def filter_outliers(cvs, threshold=1.5):
    if len(cvs) == 0:
        return []
    mean_cv = np.mean(cvs)
    std_cv = np.std(cvs)
    if std_cv == 0:
        return cvs
    return [cv for cv in cvs if abs(cv - mean_cv) <= threshold * std_cv]

for d in d_values:
    th_runs, x1_runs, x2_runs, x3_runs = [], [], [], []

    for _ in range(n_repeat):
        cv_th, cv_x1, cv_x2, cv_x3 = run_simulation_phase(
            d=d, D=D, a=a, b=b,
            num_steps=15_000_000,
            measure_start=1_000_000
        )
        if not np.isnan(cv_th):  th_runs.append(cv_th)
        if not np.isnan(cv_x1):  x1_runs.append(cv_x1)
        if not np.isnan(cv_x2):  x2_runs.append(cv_x2)
        if not np.isnan(cv_x3):  x3_runs.append(cv_x3)

    filtered_th = filter_outliers(th_runs)
    filtered_x1 = filter_outliers(x1_runs)
    filtered_x2 = filter_outliers(x2_runs)
    filtered_x3 = filter_outliers(x3_runs)

    def m_s(arr):
        if len(arr) == 0:
            return np.nan, np.nan, 0
        return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), len(arr)

    mT, sT, nT = m_s(filtered_th)
    m1, s1, n1 = m_s(filtered_x1)
    m2, s2, n2 = m_s(filtered_x2)
    m3, s3, n3 = m_s(filtered_x3)

    mean_th.append(mT); std_th.append(sT); n_th.append(nT)
    mean_x1.append(m1); std_x1.append(s1); n_x1.append(n1)
    mean_x2.append(m2); std_x2.append(s2); n_x2.append(n2)
    mean_x3.append(m3); std_x3.append(s3); n_x3.append(n3)

    print(f"d={d:.2f} | n=[θ:{nT}, x1:{n1}, x2:{n2}, x3:{n3}] | "
          f"CV(sinθ)={mT:.4f}±{sT:.4f}, CV(x1)={m1:.4f}±{s1:.4f}, CV(x2)={m2:.4f}±{s2:.4f}, CV(x3)={m3:.4f}±{s3:.4f}")

mean_th = np.array(mean_th); std_th = np.array(std_th)
mean_x1 = np.array(mean_x1); std_x1 = np.array(std_x1)
mean_x2 = np.array(mean_x2); std_x2 = np.array(std_x2)
mean_x3 = np.array(mean_x3); std_x3 = np.array(std_x3)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xscale('log')
ax.set_yscale('log')

ax.errorbar(d_values, mean_th, yerr=std_th, fmt='o', capsize=3, elinewidth=1.2, label='sin(θ)')
ax.errorbar(d_values, mean_x1, yerr=std_x1, fmt='o', capsize=3, elinewidth=1.2, label='x₁')
ax.errorbar(d_values, mean_x2, yerr=std_x2, fmt='o', capsize=3, elinewidth=1.2, label='x₂')
ax.errorbar(d_values, mean_x3, yerr=std_x3, fmt='o', capsize=3, elinewidth=1.2, label='x₃')

ax.set_xlabel("Decay rate d")
ax.set_ylabel("Coefficient of variation (CV)")
ax.set_title("Oscillation precision vs decay rate (θ-driven phase model, mean ± 1 s.d., n repeats)")
ax.legend()
plt.tight_layout()
plt.show()
