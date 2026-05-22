import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax


# -------------------------------------------------------------
# 1. Simulation (same as your previous one)
# -------------------------------------------------------------
def run_simulation(D=0.001, k=1, K=1, T=39.7, num_steps=15000000):
    dt = 0.0001
    epsilon = 0.01
    m = 10

    u = v = w = x1 = x2 = x3 = 0
    noise = np.random.normal(0, 1, size=num_steps)

    w_list  = np.zeros(num_steps)
    x1_list = np.zeros(num_steps)
    x2_list = np.zeros(num_steps)
    x3_list = np.zeros(num_steps)

    for t in range(num_steps):
        du = T*(k/(K + w**m) - 0.1*u)*dt + np.sqrt(epsilon*D*dt)*noise[t]
        dv = T*(k*u - 0.1*v)*dt
        dw = T*(k*v - 0.1*w)*dt

        dx1 = (1 + 11.5*w  - 12.59*x1)*dt
        dx2 = (1 + 11.5*x1 - 12.59*x2)*dt
        dx3 = (1 + 11.5*x2 - 12.59*x3)*dt

        u += du
        v += dv
        w += dw
        x1 += dx1
        x2 += dx2
        x3 += dx3

        w_list[t]  = w
        x1_list[t] = x1
        x2_list[t] = x2
        x3_list[t] = x3

    return w_list, x1_list, x2_list, x3_list, dt


# -------------------------------------------------------------
# 2. Detect peaks
# -------------------------------------------------------------
def get_peaks(signal, order=100, start=1000000):
    return argrelmax(signal[start:], order=order)[0] + start


# -------------------------------------------------------------
# NEW: extract by time window (e.g., 200–210)
# -------------------------------------------------------------
def extract_time_window(signal, dt, tmin=190.0, tmax=212.0):
    i0 = int(tmin / dt)
    i1 = int(tmax / dt)
    t = np.arange(len(signal)) * dt
    seg = signal[i0:i1]
    t_seg = t[i0:i1]

    # peaks inside this segment (local indices)
    pw = argrelmax(seg, order=100)[0]
    return t_seg, seg, pw



# -------------------------------------------------------------
# 4. RUN EVERYTHING
# -------------------------------------------------------------
w, x1, x2, x3, dt = run_simulation()

# extract time window 200–210 (and peaks within the window)
t_w,  w_seg,  pw_w  = extract_time_window(w,  dt, 190.0, 212.0)
t_x1, x1_seg, pw_x1 = extract_time_window(x1, dt, 190.0, 212.0)
t_x2, x2_seg, pw_x2 = extract_time_window(x2, dt, 190.0, 212.0)
t_x3, x3_seg, pw_x3 = extract_time_window(x3, dt, 190.0, 212.0)

# -------------------------------------------------------------
# 5. Plot the window (200–210)
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(3, 7), sharex=True)

axes[0].plot(t_w, w_seg, 'k-')
axes[0].plot(t_w[pw_w], w_seg[pw_w], 'ro', linestyle='None')
axes[0].set_ylabel("w(t)")
axes[0].set_ylim(1.0, 3.0)
axes[0].set_title("Goodwin Oscillator — t = 200 to 210")

axes[1].plot(t_x1, x1_seg, 'b-')
axes[1].plot(t_x1[pw_x1], x1_seg[pw_x1], 'ro', linestyle='None')
axes[1].set_ylim(1.0, 3.0)
axes[1].set_ylabel("x1(t)")

axes[2].plot(t_x2, x2_seg, 'g-')
axes[2].plot(t_x2[pw_x2], x2_seg[pw_x2], 'ro', linestyle='None')
axes[2].set_ylim(1.0, 3.0)
axes[2].set_ylabel("x2(t)")

axes[3].plot(t_x3, x3_seg, color='purple')
axes[3].plot(t_x3[pw_x3], x3_seg[pw_x3], 'ro', linestyle='None')
axes[3].set_ylim(1.0, 3.0)
axes[3].set_ylabel("x3(t)")
axes[3].set_xlabel("Time")

plt.tight_layout()
plt.show()

