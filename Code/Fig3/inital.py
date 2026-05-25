import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # requires scipy

# File path (use raw string to avoid escape sequences)
file_path = r"D:\Data\S-Velocity\inital_model.txt"

try:
    # Skip the first line (header), load data
    data = np.loadtxt(file_path, skiprows=1)          # columns: depth (m), Vp (km/s), Vs (km/s)
    depth_orig = data[:, 0]                           # original depth
    Vp_orig = data[:, 1]                              # original P-wave velocity
    Vs_orig = data[:, 2]                              # original S-wave velocity
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Calculate original Vp/Vs ratio
VpVs_orig = Vp_orig / Vs_orig

# ------------------- Smoothing -------------------
# Generate a denser depth array (e.g., 500 points between min and max depth)
depth_smooth = np.linspace(depth_orig.min(), depth_orig.max(), 500)

# Cubic spline interpolation for Vp
cs_Vp = CubicSpline(depth_orig, Vp_orig)
Vp_smooth = cs_Vp(depth_smooth)

# Cubic spline interpolation for Vs
cs_Vs = CubicSpline(depth_orig, Vs_orig)
Vs_smooth = cs_Vs(depth_smooth)

# Cubic spline interpolation for Vp/Vs
cs_VpVs = CubicSpline(depth_orig, VpVs_orig)
VpVs_smooth = cs_VpVs(depth_smooth)

# ------------------- Plotting -------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Subplot (a) - P-wave velocity
ax1.plot(Vp_smooth, depth_smooth, 'b-', linewidth=2)
ax1.set_xlabel('P-wave velocity (km/s)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('(a) P-wave velocity')
ax1.invert_yaxis()

# Subplot (b) - S-wave velocity
ax2.plot(Vs_smooth, depth_smooth, 'r-', linewidth=2)
ax2.set_xlabel('S-wave velocity (km/s)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('(b) S-wave velocity')
ax2.invert_yaxis()

# Subplot (c) - Vp/Vs ratio
ax3.plot(VpVs_smooth, depth_smooth, 'g-', linewidth=2)
ax3.set_xlabel('Vp/Vs ratio')
ax3.set_ylabel('Depth (m)')
ax3.set_title('(c) Vp/Vs ratio')
ax3.invert_yaxis()

plt.tight_layout()
plt.show()