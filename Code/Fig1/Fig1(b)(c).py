import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

file_b = r"D:\Data\S-Velocity\Active_Shots_StackedSP1_ID241-316.npy"
file_c = r"D:\Data\S-Velocity\Cross_Correlation_T.npy"

data_b = np.load(file_b)
data_c = np.load(file_c)

data_b = data_b[200:7200, :2000]

ntraces, nsamples = data_c.shape
fs = 100
dt = 1.0 / fs
t = np.arange(nsamples) * dt

spacing = 0.6
y_offsets = np.arange(ntraces) * spacing

scaling_factor = 0.5
traces_scaled = data_c / np.max(np.abs(data_c), axis=1, keepdims=True) * scaling_factor

fig, axes = plt.subplots(2, 1, figsize=(8, 14))

ax0 = axes[0]
im = ax0.imshow(data_b, aspect='auto', cmap='seismic',
                extent=[0, data_b.shape[1] / 500, data_b.shape[0], 0],
                vmin=-0.000005, vmax=0.000005)
cbar = fig.colorbar(im, ax=ax0)
cbar.set_label('Strain Rate')
ax0.set_title('(b)')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Channel')

ax1 = axes[1]
for i in range(ntraces):
    ax1.plot(t, traces_scaled[i] + y_offsets[i], 'k-', linewidth=0.8)

ax1.set_xlim(0, 20)
ax1.set_yticks(y_offsets)
ax1.set_yticklabels([f'{val:.1f}' for val in y_offsets])
ax1.invert_yaxis()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Virtual source distance (km)')
ax1.set_title('(c)')

plt.tight_layout()
plt.show()