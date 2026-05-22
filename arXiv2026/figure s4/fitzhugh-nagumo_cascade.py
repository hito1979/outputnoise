import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

# ============================================================
# 1. FitzHugh–Nagumo clock + downstream chain (x1..x15)
# ============================================================

def run_simulation(
    coupling,
    a=0.5, b=1.0, c=-0.5,     # FHN parameters (set to what you use)
    d=10.0,                 # downstream decay (your "d" style)
    D=0.00001,                   # noise strength on v
    num_steps=12_000_000,
    measure_start=1_000_000,
    dt=1e-4,
    n_chain=15,
    seed=None
):
    epsilon = 0.01
    rng = np.random.default_rng(seed)

    v = 0.1
    w = 0.1
    x = np.zeros(n_chain)

    noise = rng.normal(0, 1, size=num_steps)

    n_rec = num_steps - measure_start
    w_traj = np.zeros(n_rec)
    x_traj = np.zeros((n_chain, n_rec))

    for t in range(num_steps):
        # ---- FHN clock ----
        dv = 7.0*(v*(a - v)*(v - 1.0) - w) * dt + np.sqrt(epsilon * D * dt) * noise[t]
        dw = 7.0*(b * v - c * w) * dt
        v += dv
        w += dw

        # ---- Downstream chain (your requested form) ----
        dx = np.zeros(n_chain)
        dx[0] = (1 + coupling * w - d*x[0]) * dt
        for i in range(1, n_chain):
            dx[i] = (1 + coupling * x[i-1] - d*x[i]) * dt
        x += dx

        # ---- store ----
        if t >= measure_start:
            idx = t - measure_start
            w_traj[idx] = w
            x_traj[:, idx] = x

    return w_traj, x_traj, dt


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

couplings = [11.7]
repeats = 10
n_chain = 15

results_amp = {}
results_cv = {}

for cpl in couplings:
    print(f"Running coupling = {cpl}")
    amp_all = []
    cv_all = []

    for r in range(repeats):
        w_sig, x_sig, dt = run_simulation(
            coupling=cpl,
            a=0.5, b=1.0, c=-0.5,   # <-- set your FHN params
            d=10.0,
            D=0.00001,
            num_steps=12_000_000,
            measure_start=1_000_000,
            dt=1e-4,
            n_chain=n_chain,
            seed=r
        )

        amps = [calc_amp(x_sig[i]) for i in range(n_chain)]
        cvs  = [calc_cv(x_sig[i], dt=dt) for i in range(n_chain)]

        amp_all.append(amps)
        cv_all.append(cvs)

    results_amp[cpl] = np.array(amp_all)  # (repeats, 15)
    results_cv[cpl]  = np.array(cv_all)   # (repeats, 15)


# ============================================================
# 4. Plot settings
# ============================================================

colors = {11.7: "black"}

x_pos = np.arange(n_chain)
labels = [f"x{i+1}" for i in range(n_chain)]


# ============================================================
# 5. Figure 1: Amplitude (mean ± SD)
# ============================================================

plt.figure(figsize=(7,5))

for cpl in couplings:
    amp_mean = np.nanmean(results_amp[cpl], axis=0)
    amp_std  = np.nanstd(results_amp[cpl], axis=0)

    plt.fill_between(
        x_pos,
        amp_mean - amp_std,
        amp_mean + amp_std,
        alpha=0.25,
        color=colors.get(cpl, "black")
    )

    plt.plot(
        x_pos,
        amp_mean,
        marker="o",
        linewidth=2.5,
        markersize=20,
        color=colors.get(cpl, "black"),
        label=f"coupling = {cpl}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("Amplitude")
plt.ylim(0.01, 0.1)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5amplitude_FHN_11p5.pdf", dpi=300)
plt.show()


# ============================================================
# 6. Figure 2: CV (mean ± SE)
# ============================================================

plt.figure(figsize=(7,5))

for cpl in couplings:
    cv_mean = np.nanmean(results_cv[cpl], axis=0)
    cv_std  = np.nanstd(results_cv[cpl], axis=0)

    n = np.sum(~np.isnan(results_cv[cpl]), axis=0)
    cv_se = cv_std / np.sqrt(n)

    plt.fill_between(
        x_pos,
        cv_mean - cv_se,
        cv_mean + cv_se,
        alpha=0.25,
        color=colors.get(cpl, "black")
    )

    plt.plot(
        x_pos,
        cv_mean,
        marker="o",
        linewidth=2.5,
        markersize=20,
        color=colors.get(cpl, "black"),
        label=f"coupling = {cpl}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CV_FHN_11p5_SE.pdf", dpi=300)
plt.show()


# ============================================================
# 7. Optional: CV error bars (mean ± SD)
# ============================================================

plt.figure(figsize=(7,5))

for cpl in couplings:
    cv_mean = np.nanmean(results_cv[cpl], axis=0)
    cv_std  = np.nanstd(results_cv[cpl], axis=0)

    plt.errorbar(
        x_pos,
        cv_mean,
        yerr=cv_std,
        fmt="o-",
        linewidth=2.5,
        markersize=20,
        capsize=4,
        color=colors.get(cpl, "black"),
        label=f"coupling = {cpl}"
    )

plt.xticks(x_pos, labels)
plt.xlabel("Output")
plt.ylabel("CV of period (%)")
plt.yscale("log")
plt.title("CV propagation (mean ± SD error bars, log scale)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig5CVerrorbar_FHN_11p5_SD.pdf", dpi=300)
plt.show()
