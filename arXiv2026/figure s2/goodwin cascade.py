import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

# ============================================================
# 1. Goodwin clock + linear feed-forward chain
# ============================================================

def run_simulation(
    coupling,
    k=1.0, K=1.0, d=10.0, T=39.7,
    D=0.00001,
    num_steps=5_000_000,
    measure_start=1_000_000,
    dt=1e-4
):
    epsilon = 0.01
    m = 10

    # Goodwin variables
    u = v = w = 0.1

    # downstream chain x1 ... x10
    n_chain = 15
    x = np.zeros(n_chain)

    noise = np.random.normal(0, 1, size=num_steps)

    # record trajectories after transient
    w_traj = np.zeros(num_steps - measure_start)
    x_traj = np.zeros((n_chain, num_steps - measure_start))

    for t in range(num_steps):

        # ---- Goodwin oscillator ----
        du = T * (k / (K + w**m) - 0.1*u) * dt + np.sqrt(epsilon*D*dt) * noise[t]
        dv = T * (k*u - 0.1*v) * dt
        dw = T * (k*v - 0.1*w) * dt

        u += du
        v += dv
        w += dw

        # ---- Downstream chain ----
        dx = np.zeros(n_chain)
        dx[0] = (1 + w - d*x[0]) * dt
        for i in range(1, n_chain):
            dx[i] = (1 + coupling * x[i-1] - d*x[i]) * dt

        x += dx

        # ---- store ----
        if t >= measure_start:
            idx = t - measure_start
            w_traj[idx] = w
            x_traj[:, idx] = x

    return w_traj, x_traj


# ============================================================
# 2. Metrics: CV and amplitude
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
# 3. Run simulations
# ============================================================

couplings = [11.5]
repeats = 10

results_amp = {}
results_cv = {}

for c in couplings:
    print(f"Running coupling = {c}")
    amp_all = []
    cv_all = []

    for r in range(repeats):
        w_sig, x_sig = run_simulation(coupling=c)

        # x1..x10 only
        amps = [calc_amp(x_sig[i]) for i in range(15)]
        cvs  = [calc_cv(x_sig[i])  for i in range(15)]

        amp_all.append(amps)
        cv_all.append(cvs)

    results_amp[c] = np.array(amp_all)  # shape: (repeats, 10)
    results_cv[c]  = np.array(cv_all)   # shape: (repeats, 10)


# ============================================================
# 4. Plot settings
# ============================================================

colors = {
    11.5: "black"
}

x_pos = np.arange(15)
labels = [f"x{i+1}" for i in range(15)]


# ============================================================
# 5. Figure 1: Amplitude (single figure, all couplings)
# ============================================================

plt.figure(figsize=(7,5))

for c in couplings:
    amp_mean = np.nanmean(results_amp[c], axis=0)
    amp_std  = np.nanstd(results_amp[c], axis=0)

    plt.fill_between(
        x_pos,
        amp_mean - amp_std,
        amp_mean + amp_std,
        alpha=0.25,
        color=colors[c]
    )

    plt.plot(
        x_pos,
        amp_mean,
        marker="o",
        linewidth=2.5,
        markersize=20,
        color=colors[c],
        label=f"coupling = {c}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("Amplitude")
plt.ylim(0.0, 0.2)
#plt.yscale("log")
#plt.title("Propagation of oscillation amplitude along feed-forward chain (mean ± SD)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5amplitude11,5se.pdf", dpi=300)
plt.show()


# ============================================================
# 6. Figure 2: CV (single figure, all couplings)
# ============================================================

plt.figure(figsize=(7,5))

for c in couplings:
    cv_mean = np.nanmean(results_cv[c], axis=0)
    cv_std  = np.nanstd(results_cv[c], axis=0)

    # ✅ standard error (SE) = SD / sqrt(n_nonan)
    n = np.sum(~np.isnan(results_cv[c]), axis=0)
    cv_se = cv_std / np.sqrt(n)

    plt.fill_between(
        x_pos,
        cv_mean - cv_se,
        cv_mean + cv_se,
        alpha=0.25,
        color=colors[c]
    )

    plt.plot(
        x_pos,
        cv_mean,
        marker="o",
        linewidth=2.5,
        markersize=20,
        color=colors[c],
        label=f"coupling = {c}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CV11,5se.pdf", dpi=300)
plt.show()

plt.figure(figsize=(7,5))

for c in couplings:
    cv_mean = np.nanmean(results_cv[c], axis=0)
    cv_std  = np.nanstd(results_cv[c], axis=0)

    plt.errorbar(
        x_pos,
        cv_mean,
        yerr=cv_std,
        fmt="o-",
        linewidth=2.5,
        markersize=20,
        capsize=4,
        color=colors[c],
        label=f"coupling = {c}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")
plt.title("CV propagation (mean ± SD error bars, log scale)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CVerrorbar11,5se.pdf", dpi=300)
plt.show()



