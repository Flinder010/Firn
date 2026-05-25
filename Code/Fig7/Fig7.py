import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal import correlate, detrend, resample
import string

plt.rcParams.update({'font.size': 12})

# ------------------------------
# Parameter settings
# ------------------------------
fs_das = 500          # DAS sampling rate (Hz)
fs_geo = 100          # Geophone sampling rate (Hz)
dx = 1.0              # DAS channel spacing (m)
geo_spacing = 600.0   # Geophone spacing (m) – not directly used here

# File paths (modify according to your data)
das_file = r"D:\Data\S-Velocity\Active_Shots_StackedSP1_ID241-316.npy"
geo_file = r"D:\Data\S-Velocity\Cross_Correlation_T.npy"

# ------------------------------
# Load data
# ------------------------------
das_data = np.load(das_file)          # shape: (n_das, n_time_das)
geo_data = np.load(geo_file)          # shape: (n_geo, n_time_geo)  (assume n_geo = 12)

print("DAS data shape:", das_data.shape)
print("Geophone data shape:", geo_data.shape)

# Select DAS channels and time window (as in original code)
ch_start = 200
ch_end = 7200
das_data = das_data[ch_start:ch_end, :2000]   # take first 2000 time samples
n_das, n_time_das = das_data.shape
print("Selected DAS shape:", das_data.shape)

# Number of geophone channels (assume 12)
n_geo = geo_data.shape[0] if geo_data.ndim > 1 else 1
print("Number of geophone channels:", n_geo)

# Generate letter labels: ['(a)', '(b)', ...]
labels = [f'({letter})' for letter in string.ascii_lowercase[:n_geo]]

# Create a common ScalarMappable for the colorbar (shared by all subplots)
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap='jet')
sm.set_array([])   # dummy array, needed for colorbar

# Create a figure with 12 subplots (3 rows, 4 columns)
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()  # flatten for easy indexing

# Loop over each geophone channel
for geo_ch in range(n_geo):
    print(f"Processing geophone channel {geo_ch+1}/{n_geo}")

    # Extract the geophone signal for this channel
    geo_signal = geo_data[geo_ch, :] if n_geo > 1 else geo_data[:]

    # Resample geophone data to DAS sampling rate if needed
    if fs_geo != fs_das:
        n_target = int(len(geo_signal) * fs_das / fs_geo)
        geo_signal = resample(geo_signal, n_target)

    # Trim both signals to the same length (minimum length)
    min_len = min(n_time_das, len(geo_signal))
    das_trimmed = das_data[:, :min_len]
    geo_trimmed = geo_signal[:min_len]

    # ------------------------------
    # Cross-correlation between each DAS channel and the geophone
    # ------------------------------
    ccf_list = []
    for i in range(n_das):
        # Remove mean (constant detrend)
        x = detrend(das_trimmed[i, :], type='constant')
        y = detrend(geo_trimmed, type='constant')
        # Frequency-domain cross-correlation, 'same' mode keeps length = len(x)
        corr = correlate(x, y, method='fft', mode='same')
        ccf_list.append(corr)
    ccf = np.array(ccf_list)   # shape: (n_das, n_time)

    # ------------------------------
    # Distance vector (assuming geophone is at the first DAS channel)
    # ------------------------------
    distances = np.arange(n_das) * dx   # in meters

    # ------------------------------
    # 2D FFT analysis
    # ------------------------------
    # Apply Hanning windows to reduce spectral leakage
    window_t = np.hanning(ccf.shape[1])
    window_x = np.hanning(ccf.shape[0])
    window_2d = np.outer(window_x, window_t)
    ccf_windowed = ccf * window_2d

    # 2D FFT
    nfft_x = ccf.shape[0]   # number of FFT points in space
    nfft_t = ccf.shape[1]   # number of FFT points in time
    spec = fft2(ccf_windowed, s=(nfft_x, nfft_t))
    spec = fftshift(spec, axes=0)   # shift wavenumber axis so zero is at center
    spec = fftshift(spec, axes=1)   # shift frequency axis

    # Wavenumber and frequency axes
    kx = fftfreq(nfft_x, d=dx)
    kx = fftshift(kx)                # aligned with spec after fftshift
    freq = fftfreq(nfft_t, d=1/fs_das)
    freq = fftshift(freq)

    # Keep only positive wavenumbers (propagation direction)
    pos_k_idx = kx > 0
    kx_pos = kx[pos_k_idx]
    spec_pos = spec[pos_k_idx, :]

    # Power spectrum
    power = np.abs(spec_pos)**2

    # Limit frequency range (e.g., 1–200 Hz)
    freq_idx = (freq >= 1) & (freq <= 200)
    freq_pos = freq[freq_idx]
    power = power[:, freq_idx]

    # Convert to phase velocity: v = f / k  (m/s) -> km/s
    K, F = np.meshgrid(kx_pos, freq_pos, indexing='ij')
    V = F / K                # velocity in m/s
    V = V / 1000.0           # convert to km/s

    # Velocity display range (adjust as needed)
    vmin, vmax = 0.5, 1.5    # km/s
    # Clip velocities outside the range for display (optional)
    valid = (V >= vmin) & (V <= vmax)

    # Normalize power per frequency column to enhance dispersion curves
    power_norm = np.zeros_like(power)
    for i_freq in range(power.shape[1]):
        col = power[:, i_freq]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val > min_val:
            power_norm[:, i_freq] = (col - min_val) / (max_val - min_val)
        else:
            power_norm[:, i_freq] = 0.5

    display_data = power_norm

    # ------------------------------
    # Apply mask (black dots) according to the reference code
    # ------------------------------
    # Create frequency grid matching display_data shape
    freq_grid = np.tile(freq_pos, (display_data.shape[0], 1))
    # Define conditions
    value_condition = display_data > 0.99
    speed_condition = V <= 1.6          # velocity <= 1.6 km/s
    freq_condition = freq_grid <= 120   # frequency <= 120 Hz
    combined_mask = value_condition & speed_condition & freq_condition

    x_highlight = freq_grid[combined_mask]
    y_highlight = V[combined_mask]

    # ------------------------------
    # Plot in the corresponding subplot
    # ------------------------------
    ax = axes[geo_ch]
    # First plot the black dots (zorder=5 to keep them on top)
    ax.scatter(x_highlight, y_highlight, s=5, color='black', marker='.', zorder=5)
    # Then plot the f-v image, using the shared norm
    pcm = ax.pcolormesh(freq_pos, V, display_data, shading='auto',
                        cmap='jet', norm=norm)
    ax.set_ylim(vmin, vmax)
    ax.set_xlim(30, 50)
    # Set axis labels with smaller font
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Love wave phase velocity (km/s)')
    # Set title with letter label, smaller font
    ax.set_title(f'{labels[geo_ch]}')

# After all subplots are drawn, add a vertical colorbar on the right
plt.tight_layout()
plt.subplots_adjust(right=0.9)   # make room for the colorbar on the right
cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                    fraction=0.02, pad=0.02, aspect=40, shrink=0.8)  # vertical orientation
cbar.set_label('Normalized amplitude')   # colorbar label font size

plt.show()