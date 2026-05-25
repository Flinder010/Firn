import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# File paths (use raw strings to avoid escape sequences)
initial_file = r"D:\Data\S-Velocity\inital_model.txt"
inverted_file = r"D:\Data\S-Velocity\inverted_model.txt"

try:
    # Load initial model (skip the first header line)
    data_initial = np.loadtxt(initial_file, skiprows=1)
    depth_initial = data_initial[:, 0]
    Vp_initial = data_initial[:, 1]
    Vs_initial = data_initial[:, 2]
except Exception as e:
    print(f"Error reading initial model file: {e}")
    exit()

try:
    # Load inverted result (skip the first header line)
    data_inverted = np.loadtxt(inverted_file, skiprows=1)
    depth_inverted = data_inverted[:, 0]
    Vp_inverted = data_inverted[:, 1]
    Vs_inverted = data_inverted[:, 2]
except Exception as e:
    print(f"Error reading inverted model file: {e}")
    exit()

# Calculate Vp/Vs ratios
VpVs_initial = Vp_initial / Vs_initial
VpVs_inverted = Vp_inverted / Vs_inverted

# ------------------- Smoothing -------------------
# Generate a common dense depth grid covering both models
depth_min = min(depth_initial.min(), depth_inverted.min())
depth_max = max(depth_initial.max(), depth_inverted.max())
depth_smooth = np.linspace(depth_min, depth_max, 500)

# Interpolate initial model
cs_Vp_init = CubicSpline(depth_initial, Vp_initial)
Vp_init_smooth = cs_Vp_init(depth_smooth)

cs_Vs_init = CubicSpline(depth_initial, Vs_initial)
Vs_init_smooth = cs_Vs_init(depth_smooth)

cs_VpVs_init = CubicSpline(depth_initial, VpVs_initial)
VpVs_init_smooth = cs_VpVs_init(depth_smooth)

# Interpolate inverted model
cs_Vp_inv = CubicSpline(depth_inverted, Vp_inverted)
Vp_inv_smooth = cs_Vp_inv(depth_smooth)

cs_Vs_inv = CubicSpline(depth_inverted, Vs_inverted)
Vs_inv_smooth = cs_Vs_inv(depth_smooth)

cs_VpVs_inv = CubicSpline(depth_inverted, VpVs_inverted)
VpVs_inv_smooth = cs_VpVs_inv(depth_smooth)

# ------------------- Plotting -------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Subplot (a) - P-wave velocity
ax1.plot(Vp_init_smooth, depth_smooth, 'b--', linewidth=2, label='Initial')
ax1.plot(Vp_inv_smooth, depth_smooth, 'b-', linewidth=2, label='Inverted')
ax1.set_xlabel('Vp (km/s)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('(a)')
ax1.invert_yaxis()
ax1.legend(loc='lower left')

# Subplot (b) - S-wave velocity
ax2.plot(Vs_init_smooth, depth_smooth, 'r--', linewidth=2, label='Initial')
ax2.plot(Vs_inv_smooth, depth_smooth, 'r-', linewidth=2, label='Inverted')
ax2.set_xlabel('Vs (km/s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('(b)')
ax2.invert_yaxis()
ax2.legend(loc='lower left')

# Subplot (c) - Vp/Vs ratio
ax3.plot(VpVs_init_smooth, depth_smooth, 'g--', linewidth=2, label='Initial')
ax3.plot(VpVs_inv_smooth, depth_smooth, 'g-', linewidth=2, label='Inverted')
ax3.set_xlabel('Vp/Vs')
ax3.set_ylabel('Depth (m)')
ax3.set_title('(c)')
ax3.invert_yaxis()
ax3.legend(loc='lower left')

plt.tight_layout()
plt.show()