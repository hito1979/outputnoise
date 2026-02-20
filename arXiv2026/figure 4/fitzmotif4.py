import numpy as np
from scipy.signal import argrelmax
import pandas as pd

# ---------------- Outlier filter ----------------
def filter_outliers(cvs, threshold=1.5):
    if len(cvs) < 3:
        return cvs
    mean_cv = np.mean(cvs)
    std_cv = np.std(cvs)
    return [cv for cv in cvs if abs(cv - mean_cv) <= threshold * std_cv]

# ---------------- Simulation ----------------
def run_simulation(a, c, d, D=0, num_steps=12000000, measure_start=1000000):
    dt = 0.0001
    epsilon = 0.01
    b=1
    v = w = 0.1
    x1 = x2 = x3 = 0
    noise = np.random.normal(0, 1, size=num_steps)

    w_list  = np.zeros(num_steps)
    x1_list = np.zeros(num_steps)
    x2_list = np.zeros(num_steps)
    x3_list = np.zeros(num_steps)

    for t in range(num_steps):
        dv = 7*(v*(a - v)*(v - 1) - w) * dt + np.sqrt(epsilon * D * dt) * noise[t]
        dw = 7*(b * v - c * w) * dt

        dx1 = (1 + w - d * x1) * dt
        dx2 = (1 + w + x1 - d * x2) * dt
        dx3 = (1 + x1+ x2 - d * x3) * dt

        v += dv; w += dw
        x1 += dx1; x2 += dx2; x3 += dx3

        w_list[t] = w
        x1_list[t] = x1
        x2_list[t] = x2
        x3_list[t] = x3

    w_meas  = w_list[measure_start:]
    x1_meas = x1_list[measure_start:]
    x2_meas = x2_list[measure_start:]
    x3_meas = x3_list[measure_start:]

    def calc_cv(signal, dt=0.0001, order=100):
        peaks = argrelmax(signal, order=order)[0]
        if len(peaks) < 2:
            return np.nan
        periods = np.diff(peaks) * dt
        m, s = np.mean(periods), np.std(periods)
        return (s/m)*100 if m > 0 else np.nan

    return calc_cv(w_meas), calc_cv(x1_meas), calc_cv(x2_meas), calc_cv(x3_meas)

# ---------------- Parameters ----------------
d_target = 10     # fixed decay rate
n_repeat = 50      # repetitions
a = 0.5; b = 1; c = -0.5; D = 1e-4

results = []
w_vals = []; x1_vals = []; x2_vals = []; x3_vals = []

# ---------------- Run 100 repetitions ----------------
for i in range(n_repeat):
    cv_w, cv_x1, cv_x2, cv_x3 = run_simulation(a, c, d_target, D=D)
    results.append({"repeat": i+1, "CV_w": cv_w, "CV_x1": cv_x1,
                    "CV_x2": cv_x2, "CV_x3": cv_x3})

    if not np.isnan(cv_w):  w_vals.append(cv_w)
    if not np.isnan(cv_x1): x1_vals.append(cv_x1)
    if not np.isnan(cv_x2): x2_vals.append(cv_x2)
    if not np.isnan(cv_x3): x3_vals.append(cv_x3)

    print(f"Run {i+1}/{n_repeat}: CV_w={cv_w:.3f}, CV_x1={cv_x1:.3f}, CV_x2={cv_x2:.3f}, CV_x3={cv_x3:.3f}")

# ---------------- Outlier filtering ----------------
w_filt  = filter_outliers(w_vals)
x1_filt = filter_outliers(x1_vals)
x2_filt = filter_outliers(x2_vals)
x3_filt = filter_outliers(x3_vals)

# ---------------- Compute final stats ----------------
def mean_sd(arr):
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1))

mW, sW = mean_sd(w_filt)
m1, s1 = mean_sd(x1_filt)
m2, s2 = mean_sd(x2_filt)
m3, s3 = mean_sd(x3_filt)

# ---------------- Summary DataFrame ----------------
summary = {
    "repeat": "mean±sd (filtered)",
    "d": d_target,
    "CV_w": f"{mW:.3f} ± {sW:.3f}",
    "CV_x1": f"{m1:.3f} ± {s1:.3f}",
    "CV_x2": f"{m2:.3f} ± {s2:.3f}",
    "CV_x3": f"{m3:.3f} ± {s3:.3f}"
}

df = pd.DataFrame(results)
df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

# ---------------- Save CSV ----------------
csv_filename = f"fitz_K_{d_target:.2f}motif4.csv"
df.to_csv(csv_filename, index=False)

print("\n===== SUMMARY (FILTERED) =====")
print(summary)
print(f"\nSaved results to: {csv_filename}")
