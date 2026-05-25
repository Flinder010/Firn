import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:\Data\S-Velocity\Active_Shots_StackedSP1_ID241-316.npy"
data = np.load(file_path)

print("Shape:", data.shape)
data = data[200:7200, :2000]

plt.figure(figsize=(10, 8))
plt.imshow(data, aspect='auto', cmap='seismic', 
           extent=[0, data.shape[1] / 500, data.shape[0], 0],
           vmin=-0.000005, vmax=0.000005) 
plt.colorbar(label='Strain Rate')
plt.title('(b)')
plt.xlabel('Time (s)')
plt.ylabel('Channel')
plt.show()