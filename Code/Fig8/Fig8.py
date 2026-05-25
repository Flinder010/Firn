import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal import correlate, detrend, resample, savgol_filter
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 14})

fs_das = 500          # DAS sampling rate (Hz)
fs_geo = 100          # Geophone sampling rate (Hz)
dx = 1.0              # DAS channel spacing (m)

# File paths (adjust as needed)
das_file = r"D:\Data\S-Velocity\Active_Shots_StackedSP1_ID241-316.npy"
geo_file = r"D:\Data\S-Velocity\Cross_Correlation_T.npy"

# ------------------------------
# Load data
# ------------------------------
das_data = np.load(das_file)          # shape: (n_das, n_time_das)
geo_data = np.load(geo_file)          # shape: (n_geo, n_time_geo), n_geo assumed 12

print("DAS data shape:", das_data.shape)
print("Geophone data shape:", geo_data.shape)

# Select DAS channels and time window
ch_start, ch_end = 200, 7200
das_data = das_data[ch_start:ch_end, :2000]   # take first 2000 time samples
n_das, n_time_das = das_data.shape
print("Selected DAS shape:", das_data.shape)

n_geo = geo_data.shape[0] if geo_data.ndim > 1 else 1
print("Number of geophone channels:", n_geo)

# List to store dispersion picks for each channel
all_picks = []

# Loop over each geophone channel
for geo_ch in range(n_geo):
    print(f"Processing geophone channel {geo_ch+1}/{n_geo}")

    # Extract geophone signal
    geo_signal = geo_data[geo_ch, :] if n_geo > 1 else geo_data[:]

    # Resample geophone to DAS rate if needed
    if fs_geo != fs_das:
        n_target = int(len(geo_signal) * fs_das / fs_geo)
        geo_signal = resample(geo_signal, n_target)

    # Trim to same length
    min_len = min(n_time_das, len(geo_signal))
    das_trimmed = das_data[:, :min_len]
    geo_trimmed = geo_signal[:min_len]

    # ------------------------------
    # Cross-correlation (DAS channels vs geophone)
    # ------------------------------
    ccf_list = []
    for i in range(n_das):
        x = detrend(das_trimmed[i, :], type='constant')
        y = detrend(geo_trimmed, type='constant')
        corr = correlate(x, y, method='fft', mode='same')
        ccf_list.append(corr)
    ccf = np.array(ccf_list)   # shape: (n_das, n_time)

    # ------------------------------
    # 2D FFT for f-k analysis
    # ------------------------------
    # Apply 2D Hanning window
    window_t = np.hanning(ccf.shape[1])
    window_x = np.hanning(ccf.shape[0])
    window_2d = np.outer(window_x, window_t)
    ccf_windowed = ccf * window_2d

    # 2D FFT and shift
    spec = fft2(ccf_windowed)
    spec = fftshift(spec, axes=0)   # shift wavenumber axis
    spec = fftshift(spec, axes=1)   # shift frequency axis

    # Wavenumber and frequency axes
    kx = fftshift(fftfreq(ccf.shape[0], d=dx))
    freq = fftshift(fftfreq(ccf.shape[1], d=1/fs_das))

    # Keep positive wavenumbers only
    pos_k_idx = kx > 0
    kx_pos = kx[pos_k_idx]
    spec_pos = spec[pos_k_idx, :]

    # Power spectrum
    power = np.abs(spec_pos)**2

    # Limit frequency range
    freq_idx = (freq >= 1) & (freq <= 200)
    freq_pos = freq[freq_idx]
    power = power[:, freq_idx]

    # Convert to phase velocity (m/s -> km/s)
    K, F = np.meshgrid(kx_pos, freq_pos, indexing='ij')
    V = (F / K) / 1000.0   # velocity in km/s

    # Normalize power per frequency column
    power_norm = np.zeros_like(power)
    for i_freq in range(power.shape[1]):
        col = power[:, i_freq]
        min_val, max_val = np.min(col), np.max(col)
        if max_val > min_val:
            power_norm[:, i_freq] = (col - min_val) / (max_val - min_val)
        else:
            power_norm[:, i_freq] = 0.5

    # ------------------------------
    # Extract dispersion curve (one velocity per frequency)
    # ------------------------------
    freq_grid = np.tile(freq_pos, (power_norm.shape[0], 1))
    # Conditions: power > 0.99, velocity <= 1.25 km/s, frequency <= 120 Hz
    mask = (power_norm > 0.99) & (V <= 1.25) & (freq_grid <= 120)

    # Lists to hold picked points
    freq_pick = []
    vel_pick = []

    # For each frequency, find the velocity with maximum power among those satisfying the mask
    for i_f, f in enumerate(freq_pos):
        # Indices of wavenumbers where mask is True at this frequency
        valid_idx = np.where(mask[:, i_f])[0]
        if len(valid_idx) == 0:
            continue
        # Among those, select the one with highest normalized power
        powers_at_f = power_norm[valid_idx, i_f]
        best_idx = valid_idx[np.argmax(powers_at_f)]
        freq_pick.append(f)
        vel_pick.append(V[best_idx, i_f])

    freq_pick = np.array(freq_pick)
    vel_pick = np.array(vel_pick)

    # ------------------------------
    # Smooth the dispersion curve (if enough points)
    # ------------------------------
    if len(freq_pick) > 5:
        # Apply Savitzky-Golay filter
        window_len = min(11, len(freq_pick) // 2 * 2 + 1)  # odd number
        vel_smooth = savgol_filter(vel_pick, window_length=window_len, polyorder=2)
    else:
        vel_smooth = vel_pick

    # Optional: interpolate to a denser frequency grid for smoother plotting
    if len(freq_pick) > 3:
        f_dense = np.linspace(freq_pick.min(), freq_pick.max(), 200)
        interp_func = interp1d(freq_pick, vel_smooth, kind='cubic', fill_value='extrapolate')
        vel_dense = interp_func(f_dense)
    else:
        f_dense, vel_dense = freq_pick, vel_smooth

    # Store the picks for this channel
    all_picks.append((f_dense, vel_dense))

# ------------------------------
# Plot all dispersion curves together
# ------------------------------
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20(np.linspace(0, 1, n_geo))

for ch in range(n_geo):
    freqs, vels = all_picks[ch]
    if len(freqs) > 0:
        plt.plot(freqs, vels, color=colors[ch], linewidth=0.8, label=f'Dispersion curve {ch+1}')
    else:
        print(f"Dispersion curve {ch+1} has no picks, skipped.")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Love wave phase velocity (km/s)')
plt.xlim(30, 50)
plt.ylim(0.75, 1.25)
plt.tight_layout()

all_freqs = []
all_vels = []
for f, v in all_picks:
    if len(f) > 0:          # skip channels with no picks
        all_freqs.append(f)
        all_vels.append(v)

if all_freqs:   # proceed only if there is at least one valid channel
    f_min = min([np.min(f) for f in all_freqs])
    f_max = max([np.max(f) for f in all_freqs])
    f_common = np.arange(f_min, f_max, 0.5)   # step 0.5 Hz, adjust as needed

    v_interp_list = []
    for f, v in zip(all_freqs, all_vels):
        v_interp = np.interp(f_common, f, v, left=np.nan, right=np.nan)
        v_interp_list.append(v_interp)

    v_interp_array = np.array(v_interp_list)

    v_min = np.nanmin(v_interp_array, axis=0)
    v_max = np.nanmax(v_interp_array, axis=0)

    width = v_max - v_min
    expansion = 0.1 * width
    v_min_expanded = v_min - expansion
    v_max_expanded = v_max + expansion

    plt.fill_between(f_common, v_min, v_max,
                     color='gray', alpha=0.2,
                     label='Lateral variation range')
    
    plt.legend(loc='best')

plt.show()