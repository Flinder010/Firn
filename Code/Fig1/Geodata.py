import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:\Data\S-Velocity\Cross_Correlation_T.npy"
data = np.load(file_path)          # shape: (12, 5000)
ntraces, nsamples = data.shape
fs = 100                           
dt = 1.0 / fs                       
t = np.arange(nsamples) * dt       

spacing = 0.6
y_offsets = np.arange(ntraces) * spacing   

scaling_factor = 0.5                
traces_scaled = data / np.max(np.abs(data), axis=1, keepdims=True) * scaling_factor

plt.figure(figsize=(10, 6))

for i in range(ntraces):
    plt.plot(t, traces_scaled[i] + y_offsets[i], 'k-', linewidth=0.8)

plt.xlim(0, 20)

plt.yticks(y_offsets, [f'{val:.1f}' for val in y_offsets])

plt.gca().invert_yaxis()

plt.xlabel('Time (s)')
plt.ylabel('Distance to virtual source (km)') 
plt.title('(c)')
plt.tight_layout()
plt.show()