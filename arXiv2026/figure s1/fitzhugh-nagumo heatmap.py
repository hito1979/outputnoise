import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmax
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

# ============================================================
# Amplitude-side peak detection (robust)
# ============================================================
def detect_peaks_amp(signal, dt, smooth_sigma=4, prominence_factor=0.05):
    smooth_signal = gaussian_filter1d(signal, sigma=smooth_sigma)
    amp = np.percentile(smooth_signal, 95) - np.percentile(smooth_signal, 5)

    peaks, _ = find_peaks(
        smooth_signal,
        prominence=prominence_factor * amp
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
# Correlation-side peak detection (argrelmax)
# ============================================================
def detect_peaks_corr(signal, order=100):
    return argrelmax(signal, order=order)[0]

def best_sine_correlation(w, dt, Nphase=200, order=100, r_min=0.8):
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
# FHN simulation (w only)
# ============================================================
def run_simulation(a, c, b=1, D=0.0,
                   num_steps=12000000,
                   measure_start=10000000,
                   dt=0.0001):
    epsilon = 0.01
    v = w = 0.1
    noise = np.random.randn(num_steps)

    w_list = np.zeros(num_steps)

    for t in range(num_steps):
        dv = 7*(v*(a - v)*(v - 1) - w) * dt + np.sqrt(epsilon * D * dt) * noise[t]
        dw = 7*(b * v - c * w) * dt
        v += dv
        w += dw
        w_list[t] = w

    return w_list[measure_start:], dt

# ============================================================
# Parameter sweep
# ============================================================
a_values = np.arange(0.1, 0.55 + 1e-9, 0.05)
c_values = np.arange(-0.5, -0.05 + 1e-9, 0.05)

heatmap_amp  = np.full((len(c_values), len(a_values)), np.nan)
heatmap_corr = np.full((len(c_values), len(a_values)), np.nan)

for i, c in enumerate(c_values):
    for j, a in enumerate(a_values):

        w_sig, dt = run_simulation(a=a, c=c)

        amp = amplitude_over_available(w_sig, dt)
        heatmap_amp[i, j] = amp

        # KEY: if amplitude is NaN, force correlation to NaN
        if np.isnan(amp):
            r = np.nan
        else:
            r = best_sine_correlation(w_sig, dt)

        heatmap_corr[i, j] = r

        print(f"a={a:.2f}, c={c:.2f}, amp={amp}, r={r}")

# ============================================================
# Plot amplitude heatmap
# ============================================================
plt.figure(figsize=(8, 6))
cmap_amp = plt.cm.viridis_r.copy()
cmap_amp.set_bad(color='white')

masked_amp = np.ma.masked_invalid(heatmap_amp)
A, C = np.meshgrid(a_values, c_values)
pcm = plt.pcolormesh(A, C, masked_amp, cmap=cmap_amp, shading='auto', vmin=0, vmax=2)

plt.colorbar(pcm, label="Mean amplitude (up to 100 cycles)")
plt.xlabel("a parameter")
plt.ylabel("c parameter")
plt.title("FHN amplitude heatmap (damped/noisy removed)")
plt.tight_layout()
plt.savefig("FHN_amplitude_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Plot correlation heatmap (masked by amplitude validity)
# ============================================================
plt.figure(figsize=(8, 6))
cmap_corr = plt.cm.coolwarm.copy()
cmap_corr.set_bad(color='white')

masked_corr = np.ma.masked_invalid(heatmap_corr)
pcm = plt.pcolormesh(A, C, masked_corr, cmap=cmap_corr, shading='auto', vmin=0.95, vmax=1.0)

plt.colorbar(pcm, label="Best sine correlation (r ≥ 0.8)")
plt.xlabel("a parameter")
plt.ylabel("c parameter")
plt.title("FHN correlation heatmap (only where amplitude is valid)")
plt.tight_layout()
plt.savefig("FHN_correlation_heatmap_masked_by_amplitude.pdf",
            format="pdf", dpi=300, bbox_inches="tight")
plt.show()




