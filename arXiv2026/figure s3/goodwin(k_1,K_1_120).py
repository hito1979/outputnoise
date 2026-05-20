import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import csv

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def safe_mean_std(arr):
    if len(arr) == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)
    return mean, std, len(arr)

def calc_cv_from_signal(signal, dt, order=100, low_ratio=0.5, high_ratio=1.5):
    peaks = argrelmax(signal, order=order)[0]
    if len(peaks) < 2:
        return np.nan

    periods = np.diff(peaks) * dt
    if len(periods) < 2:
        return np.nan

    med = np.median(periods)
    if not np.isfinite(med) or med <= 0:
        return np.nan

    lo = low_ratio * med
    hi = high_ratio * med
    good = (periods >= lo) & (periods <= hi)
    periods_f = periods[good]

    if len(periods_f) < 2:
        return np.nan

    m, s = np.mean(periods_f), np.std(periods_f)
    return (s / m) * 100 if m > 0 else np.nan

# -------------------------------------------------------------
# Simulation
# -------------------------------------------------------------
def run_simulation(k, K, d, D=0.00001, T=39.7,
                   num_steps=15000000,
                   measure_start=1000000,
                   dt=1e-4, epsilon=0.01, m_hill=10,
                   peak_order=100):

    u = v = w = x1 = x2 = x3 = 0.0
    noise = np.random.normal(0, 1, size=num_steps)

    w_list  = np.zeros(num_steps)
    x1_list = np.zeros(num_steps)
    x2_list = np.zeros(num_steps)
    x3_list = np.zeros(num_steps)

    sqrt_term = np.sqrt(epsilon * D * dt) if D > 0 else 0.0

    for t in range(num_steps):
        du = T * (k / (K + w**m_hill) - 0.1 * u) * dt + sqrt_term * noise[t]
        dv = T * (k * u - 0.1 * v) * dt
        dw = T * (k * v - 0.1 * w) * dt

        dx1 = (1 + 12*w  - d * x1) * dt
        dx2 = (1 + 12*x1 - d * x2) * dt
        dx3 = (1 + 12*x2 - d * x3) * dt

        u += du; v += dv; w += dw
        x1 += dx1; x2 += dx2; x3 += dx3

        w_list[t]  = w
        x1_list[t] = x1
        x2_list[t] = x2
        x3_list[t] = x3

    w_meas  = w_list[measure_start:]
    x1_meas = x1_list[measure_start:]
    x2_meas = x2_list[measure_start:]
    x3_meas = x3_list[measure_start:]

    cv_w  = calc_cv_from_signal(w_meas,  dt, order=peak_order)
    cv_x1 = calc_cv_from_signal(x1_meas, dt, order=peak_order)
    cv_x2 = calc_cv_from_signal(x2_meas, dt, order=peak_order)
    cv_x3 = calc_cv_from_signal(x3_meas, dt, order=peak_order)

    return cv_w, cv_x1, cv_x2, cv_x3

# -------------------------------------------------------------
# Sweep
# -------------------------------------------------------------
def sweep_min_cv_over_d(k_list, K_list, d_values,
                        n_repeat=10,
                        D=1e-5, T=39.7,
                        num_steps=15000000,
                        measure_start=1000000,
                        peak_order=100):

    nk = len(k_list)
    nK = len(K_list)

    minCV_x1 = np.full((nk, nK), np.nan)
    minCV_x2 = np.full((nk, nK), np.nan)
    minCV_x3 = np.full((nk, nK), np.nan)

    for ik, k in enumerate(k_list):
        for jK, K in enumerate(K_list):

            mean_vs_d_x1 = []
            mean_vs_d_x2 = []
            mean_vs_d_x3 = []

            for d in d_values:
                x1_runs, x2_runs, x3_runs = [], [], []

                for _ in range(n_repeat):
                    _, cv_x1, cv_x2, cv_x3 = run_simulation(
                        k=k, K=K, d=d,
                        D=D, T=T,
                        num_steps=num_steps,
                        measure_start=measure_start,
                        peak_order=peak_order
                    )

                    if not np.isnan(cv_x1): x1_runs.append(cv_x1)
                    if not np.isnan(cv_x2): x2_runs.append(cv_x2)
                    if not np.isnan(cv_x3): x3_runs.append(cv_x3)

                m1, _, _ = safe_mean_std(x1_runs)
                m2, _, _ = safe_mean_std(x2_runs)
                m3, _, _ = safe_mean_std(x3_runs)

                mean_vs_d_x1.append(m1)
                mean_vs_d_x2.append(m2)
                mean_vs_d_x3.append(m3)

            mean_vs_d_x1 = np.array(mean_vs_d_x1)
            mean_vs_d_x2 = np.array(mean_vs_d_x2)
            mean_vs_d_x3 = np.array(mean_vs_d_x3)

            if not np.all(np.isnan(mean_vs_d_x1)):
                minCV_x1[ik, jK] = np.nanmin(mean_vs_d_x1)
            if not np.all(np.isnan(mean_vs_d_x2)):
                minCV_x2[ik, jK] = np.nanmin(mean_vs_d_x2)
            if not np.all(np.isnan(mean_vs_d_x3)):
                minCV_x3[ik, jK] = np.nanmin(mean_vs_d_x3)

    return {
        "minCV_x1": minCV_x1,
        "minCV_x2": minCV_x2,
        "minCV_x3": minCV_x3,
        "K_list": np.array(K_list)
    }

# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == "__main__":
    k_list = [1.0]
    K_list = np.array([1] + list(range(10, 121, 10)), dtype=float)
    d_values = np.logspace(0, 2, 21)

    results = sweep_min_cv_over_d(
        k_list=k_list,
        K_list=K_list,
        d_values=d_values,
        n_repeat=10
    )

    K = results["K_list"]
    min_x1 = results["minCV_x1"][0]
    min_x2 = results["minCV_x2"][0]
    min_x3 = results["minCV_x3"][0]

    # ---------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------
    with open("minCV_vs_K_40.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "minCV_x1", "minCV_x2", "minCV_x3"])
        for i in range(len(K)):
            writer.writerow([K[i], min_x1[i], min_x2[i], min_x3[i]])

    print("CSV saved: minCV_vs_K.csv")

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(K, min_x1, "o", color="orange", label="x1")
    ax.plot(K, min_x2, "o", color="green",  label="x2")
    ax.plot(K, min_x3, "o", color="magenta",label="x3")

    # 👉 ADD THIS LINE (left/right spacing)
    ax.margins(x=0.05)   # 5% padding

    ax.set_xticks([1, 20, 40, 60, 80, 100, 120])

    ax.set_xlabel("K")
    ax.set_ylabel("Minimum CV over d (%)")
    ax.set_title("Best achievable precision vs K (k=1)")
    ax.legend()

    plt.tight_layout()

    # ---------------------------------------------------------
    # SAVE FIGURE
    # ---------------------------------------------------------
    plt.savefig("minCV_vs_K.pdf", format="pdf", dpi=300)
    print("Figure saved: minCV_vs_K.pdf")

    plt.show()