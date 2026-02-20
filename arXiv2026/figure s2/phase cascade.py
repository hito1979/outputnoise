import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import time, os

# ============================================================
# 1) Theta-driven simulation (x1..x15)
# ============================================================
def run_simulation(D=0.0001, K=10, num_steps=10_000_000, n_chain=15, a=1.0, b=11.5):
    np.random.seed(int(time.time()) + os.getpid())

    dt = 1e-4
    sqrt_D_dt = np.sqrt(D * dt)

    theta = 0.0
    x = np.zeros(n_chain)  # x[0]=x1 ... x[14]=x15

    noise = np.random.normal(0, 1, size=num_steps)

    theta_list = np.zeros(num_steps)
    x_list = np.zeros((n_chain, num_steps))

    for i in range(num_steps):
        theta += 2 * np.pi * dt + sqrt_D_dt * noise[i]
        theta %= 2 * np.pi

        # x1 driven by sin(theta)
        x[0] += (a + b * np.sin(theta) - K * x[0]) * dt

        # downstream chain
        for j in range(1, n_chain):
            x[j] += (a + b * x[j-1] - K * x[j]) * dt

        theta_list[i] = np.sin(theta)
        x_list[:, i] = x

    return theta_list, x_list, dt


# ============================================================
# 2) Metrics: CV and amplitude
# ============================================================
def calc_amp(signal):
    return np.max(signal) - np.min(signal)

def calc_cv(signal, dt=1e-4, order=300):
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size < 2 * order + 1:
        return np.nan

    peaks = argrelmax(signal, order=order)[0]
    if len(peaks) < 2:
        return np.nan

    periods = np.diff(peaks) * dt
    m = np.mean(periods)
    if m <= 0:
        return np.nan
    return 100.0 * np.std(periods) / m


# ============================================================
# 3) Run simulations (sweep b as "coupling")
# ============================================================
b_list = [11.3]   # <-- this plays the role of your previous coupling
repeats = 5
n_chain = 15


results_amp = {}
results_cv  = {}

for b in b_list:
    print(f"Running b = {b}")
    amp_all = []
    cv_all  = []

    for r in range(repeats):
        theta_sig, x_sig, dt = run_simulation(b=b, n_chain=n_chain)

        amps = [calc_amp(x_sig[i]) for i in range(n_chain)]
        cvs  = [calc_cv(x_sig[i], dt=dt) for i in range(n_chain)]

        amp_all.append(amps)
        cv_all.append(cvs)

    results_amp[b] = np.array(amp_all)  # (repeats, 15)
    results_cv[b]  = np.array(cv_all)   # (repeats, 15)


# ============================================================
# 4) Plot settings
# ============================================================
colors = {11.3: "black"}

x_pos  = np.arange(n_chain)
labels = [f"x{i+1}" for i in range(n_chain)]


# ============================================================
# 5) Figure 1: Amplitude (mean ± SD)
# ============================================================
plt.figure(figsize=(7,5))

for b in b_list:
    amp_mean = np.nanmean(results_amp[b], axis=0)
    amp_std  = np.nanstd(results_amp[b], axis=0)

    plt.fill_between(x_pos, amp_mean - amp_std, amp_mean + amp_std,
                     alpha=0.25, color=colors.get(b, "black"))

    plt.plot(x_pos, amp_mean, marker="o", linewidth=2.5, markersize=20,
             color=colors.get(b, "black"), label=f"b = {b}")

plt.xticks(x_pos, labels, rotation=0)
plt.xlabel("Output")
plt.ylabel("Amplitude")
plt.ylim(0.0, 4.0)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5amplitude_theta_D_0.0001.pdf", dpi=300)
plt.show()


# ============================================================
# 6) Figure 2: CV (mean ± SE) with log scale
# ============================================================
plt.figure(figsize=(7,5))

for b in b_list:
    cv_mean = np.nanmean(results_cv[b], axis=0)
    cv_std  = np.nanstd(results_cv[b], axis=0)

    n = np.sum(~np.isnan(results_cv[b]), axis=0)
    cv_se = cv_std / np.sqrt(n)

    plt.fill_between(x_pos, cv_mean - cv_se, cv_mean + cv_se,
                     alpha=0.25, color=colors.get(b, "black"))

    plt.plot(x_pos, cv_mean, marker="o", linewidth=2.5, markersize=20,
             color=colors.get(b, "black"), label=f"b = {b}")

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CV_theta_D_0.0001.pdf", dpi=300)
plt.show()


# ============================================================
# 7) Optional: CV error bars (mean ± SD)
# ============================================================
plt.figure(figsize=(7,5))

for b in b_list:
    cv_mean = np.nanmean(results_cv[b], axis=0)
    cv_std  = np.nanstd(results_cv[b], axis=0)

    plt.errorbar(x_pos, cv_mean, yerr=cv_std, fmt="o-",
                 linewidth=2.5, markersize=10, capsize=4,
                 color=colors.get(b, "black"), label=f"b = {b}")

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")
plt.title("CV propagation (mean ± SD error bars, log scale)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CVerrorbar_theta_b11p5_SD.pdf", dpi=300)
plt.show()
