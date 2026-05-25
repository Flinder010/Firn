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

# ------------------- Differences -------------------
diff_Vp = Vp_inv_smooth - Vp_init_smooth
diff_Vs = Vs_inv_smooth - Vs_init_smooth
diff_VpVs = VpVs_inv_smooth - VpVs_init_smooth

# ------------------- Plotting -------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes

# Subplot (a) - P-wave velocity
ax1.plot(Vp_init_smooth, depth_smooth, 'b--', linewidth=2, label='Initial $V_{\mathrm{P}}$')
ax1.plot(Vp_inv_smooth, depth_smooth, 'b-', linewidth=2, label='Inverted $V_{\mathrm{P}}$')
ax1.set_xlabel('$V_{\mathrm{P}}$ (km/s)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('(a)')
ax1.invert_yaxis()
ax1.legend(loc='lower left')

# Subplot (b) - S-wave velocity
ax2.plot(Vs_init_smooth, depth_smooth, 'r--', linewidth=2, label='Initial $V_{\mathrm{S}}$')
ax2.plot(Vs_inv_smooth, depth_smooth, 'r-', linewidth=2, label='Inverted $V_{\mathrm{S}}$')
ax2.set_xlabel('$V_{\mathrm{S}}$ (km/s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('(b)')
ax2.invert_yaxis()
ax2.legend(loc='lower left')

# Subplot (c) - Vp/Vs ratio
ax3.plot(VpVs_init_smooth, depth_smooth, 'g--', linewidth=2, label='Initial $V_{\mathrm{P}}/V_{\mathrm{S}}$')
ax3.plot(VpVs_inv_smooth, depth_smooth, 'g-', linewidth=2, label='Inverted $V_{\mathrm{P}}/V_{\mathrm{S}}$')
ax3.set_xlabel('$V_{\mathrm{P}}/V_{\mathrm{S}}$')
ax3.set_ylabel('Depth (m)')
ax3.set_title('(c)')
ax3.invert_yaxis()
ax3.legend(loc='lower left')

# Subplot (d) - Vp difference
ax4.plot(diff_Vp, depth_smooth, 'b-.', linewidth=2, label='Difference of $V_{\mathrm{P}}$')
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax4.set_xlabel('Velocity difference of $V_{\mathrm{P}}$ (km/s)')
ax4.set_ylabel('Depth (m)')
ax4.set_title('(d)')
ax4.invert_yaxis()
ax4.legend(loc='lower right') 

# Subplot (e) - Vs difference
ax5.plot(diff_Vs, depth_smooth, 'r-.', linewidth=2, label='Difference of $V_{\mathrm{S}}$')
ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax5.set_xlabel('Velocity difference of $V_{\mathrm{S}}$ (km/s)')
ax5.set_ylabel('Depth (m)')
ax5.set_title('(e)')
ax5.invert_yaxis()
ax5.legend(loc='lower right')

# Subplot (f) - Vp/Vs difference
ax6.plot(diff_VpVs, depth_smooth, 'g-.', linewidth=2, label='Difference of $V_{\mathrm{P}}/V_{\mathrm{S}}$')
ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax6.set_xlabel('Difference of $V_{\mathrm{P}}/V_{\mathrm{S}}$')
ax6.set_ylabel('Depth (m)')
ax6.set_title('(f)')
ax6.invert_yaxis()
ax6.legend(loc='lower left')

plt.tight_layout()
plt.show()