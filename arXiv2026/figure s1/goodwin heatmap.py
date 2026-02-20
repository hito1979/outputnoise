import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmax
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

# ============================================================
# Peak detection for amplitude (robust)
# ============================================================
def detect_peaks_amp(signal, dt, smooth_sigma=4, prominence_factor=0.05, min_peak_separation_time=0.2):
    smooth_signal = gaussian_filter1d(signal, sigma=smooth_sigma)
    amp = np.percentile(smooth_signal, 95) - np.percentile(smooth_signal, 5)

    # enforce minimum peak distance (in samples)
    min_dist = max(1, int(min_peak_separation_time / dt))

    peaks, _ = find_peaks(
        smooth_signal,
        prominence=prominence_factor * amp,
        distance=min_dist
    )
    return peaks

def is_real_oscillation(peaks, signal, dt):
    if len(peaks) < 10:
        return False

    amp_total = np.max(signal) - np.min(signal)
    if amp_total < 0.01:
        return False

    periods = np.diff(peaks) * dt
    cv = np.std(periods) / np.mean(periods)
    if cv > 0.2:
        return False

    half = len(peaks) // 2
    amp1 = np.max(signal[peaks[:half]]) - np.min(signal[peaks[:half]])
    amp2 = np.max(signal[peaks[half:]]) - np.min(signal[peaks[half:]])
    if amp2 < 0.5 * amp1:
        return False

    return True

def amplitude_over_available(signal, dt, max_cycles=100):
    peaks = detect_peaks_amp(signal, dt)
    if not is_real_oscillation(peaks, signal, dt):
        return np.nan

    amps = []
    cycles = min(len(peaks) - 1, max_cycles)
    for i in range(1, cycles + 1):
        seg = signal[peaks[i - 1]: peaks[i]]
        amps.append(np.max(seg) - np.min(seg))

    return np.mean(amps)

# ============================================================
# Peak detection for correlation (argrelmax)
# ============================================================
def detect_peaks_corr(signal, order=2000):
    return argrelmax(signal, order=order)[0]

def best_sine_correlation(w, dt, Nphase=200, order=2000, r_min=0.8):
    peaks = detect_peaks_corr(w, order=order)
    if len(peaks) < 12:
        return np.nan

    periods = np.diff(peaks) * dt
    T = np.mean(periods)
    if np.isnan(T) or T <= 0:
        return np.nan

    f = 1.0 / T
    t = np.arange(len(w)) * dt

    # last 10 cycles
    sel_peaks = peaks[-11:]
    t0, t1 = sel_peaks[0], sel_peaks[-1]
    segment = w[t0:t1]
    t_seg = t[t0:t1]

    phases = np.linspace(0, 2*np.pi, Nphase)
    best_r = -1.0

    for phi in phases:
        sine_wave = np.sin(2*np.pi*f*t_seg + phi)
        r, _ = pearsonr(segment, sine_wave)
        if r > best_r:
            best_r = r

    if best_r < r_min:
        return np.nan

    return best_r

# ============================================================
# Goodwin simulation (return w only, discard transient)
# ============================================================
def run_simulation(p, q, D=0.0, K=10.0,
                   num_steps=10_000_000,
                   measure_start=8_000_000,
                   dt=1e-4):
    """
    Goodwin-like oscillator with downstream cascade (x1-x3).
    Returns: w time series AFTER discarding transient, and dt
    """
    epsilon = 0.01
    Tscale = 39.7
    m = 10
    du_decay = dv_decay = dw_decay = 0.1

    u = v = w = 0.0
    x1 = x2 = x3 = 0.0

    noise = np.random.normal(0.0, 1.0, size=num_steps)
    w_list = np.zeros(num_steps, dtype=float)

    for t in range(num_steps):
        # core oscillator
        du = Tscale * (p * (1.0 / (q + w**m)) - du_decay * u) * dt + np.sqrt(epsilon * D * dt) * noise[t]
        dv = Tscale * (p * u - dv_decay * v) * dt
        dw = Tscale * (p * v - dw_decay * w) * dt

        # downstream cascade (as you wrote; keep unscaled)
        dx1 = (1.0 + w  - K * x1) * dt
        dx2 = (1.0 + x1 - K * x2) * dt
        dx3 = (1.0 + x2 - K * x3) * dt

        u += du
        v += dv
        w += dw
        x1 += dx1
        x2 += dx2
        x3 += dx3

        w_list[t] = w

    return w_list[measure_start:], dt

# ============================================================
# Parameter sweep (p horizontal, q vertical)
# ============================================================
p_values = np.linspace(0.1, 1.0, 10)
q_values = np.logspace(0, 2, 10)  # 1 to 100 (log10)

D = 0.0
K = 10.0

heatmap_amp  = np.full((len(q_values), len(p_values)), np.nan)
heatmap_corr = np.full((len(q_values), len(p_values)), np.nan)

dt = 1e-4
# order for argrelmax: ~0.2 time units separation (adjust if needed)
order_corr = max(10, int(0.2 / dt))

for i, q in enumerate(q_values):
    for j, p in enumerate(p_values):

        w_sig, dt = run_simulation(p=p, q=q, D=D, K=K)

        amp = amplitude_over_available(w_sig, dt)
        heatmap_amp[i, j] = amp

        # If amplitude is NaN, force correlation NaN too
        if np.isnan(amp):
            r = np.nan
        else:
            r = best_sine_correlation(w_sig, dt, order=order_corr)

        heatmap_corr[i, j] = r

        print(f"p={p:.3f}, q={q:.3f}, amp={amp}, r={r}")

# ============================================================
# Plot amplitude heatmap
# ============================================================
P, Q = np.meshgrid(p_values, q_values)

plt.figure(figsize=(8, 6))
cmap_amp = plt.cm.viridis_r.copy()
cmap_amp.set_bad(color='white')

masked_amp = np.ma.masked_invalid(heatmap_amp)
pcm = plt.pcolormesh(P, Q, masked_amp, cmap=cmap_amp, shading='auto', vmin=0, vmax=2)
plt.yscale("log")
plt.colorbar(pcm, label="Mean amplitude (up to 100 cycles)")
plt.xlabel("p parameter")
plt.ylabel("q parameter (log scale)")
plt.title("Goodwin amplitude heatmap (damped/noisy removed)")
plt.tight_layout()
plt.savefig("Goodwin_amplitude_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Plot correlation heatmap (masked by amplitude validity)
# ============================================================
plt.figure(figsize=(8, 6))
cmap_corr = plt.cm.coolwarm.copy()
cmap_corr.set_bad(color='white')

masked_corr = np.ma.masked_invalid(heatmap_corr)
pcm = plt.pcolormesh(P, Q, masked_corr, cmap=cmap_corr, shading='auto', vmin=0.95, vmax=1.0)
plt.yscale("log")
plt.colorbar(pcm, label="Best sine correlation (r ≥ 0.8)")
plt.xlabel("p parameter")
plt.ylabel("q parameter (log scale)")
plt.title("Goodwin correlation heatmap (only where amplitude is valid)")
plt.tight_layout()
plt.savefig("Goodwin_correlation_heatmap_masked_by_amplitude.pdf",
            format="pdf", dpi=300, bbox_inches="tight")
plt.show()
