import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D          # For custom legend handles
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal import correlate, detrend, resample, medfilt
from scipy.interpolate import UnivariateSpline

plt.rcParams.update({'font.size': 12})

# ============================================
# Global parameters
# ============================================
fs_das = 500          # DAS sampling rate (Hz)
fs_geo = 100          # Geophone sampling rate (Hz)
dx = 1.0              # DAS channel spacing (m)
geo_spacing = 600.0   # Geophone spacing (m) - for reference only

# File paths
das_file = r"D:\Data\S-Velocity\Active_Shots_StackedSP1_ID241-316.npy"
geo_file = r"D:\Data\S-Velocity\Cross_Correlation_T.npy"

# ============================================
# Function: compute DAS-DAS f-v data (returns everything needed)
# ============================================
def compute_das_das():
    """
    Load DAS data, compute 2D FFT, and produce the DAS-DAS f-v dispersion image.
    Returns:
        freq: frequency axis (Hz)
        V: phase velocity axis (km/s)
        disp: normalized power spectrum (2D array)
        x_hl: frequency coordinates of highlighted points
        y_hl: velocity coordinates of highlighted points
        amp_hl: amplitude of highlighted points
    """
    # Load data
    data = np.load(das_file)                     # shape: (n_channels, n_time_samples)
    # Select channel range and truncate time samples
    ch_start, ch_end = 200, 7200
    data = data[ch_start:ch_end, :2000]
    n_channels, n_time = data.shape

    # Preprocessing: remove mean per channel + apply 2D Hanning window
    data = data - np.mean(data, axis=1, keepdims=True)
    window_t = np.hanning(n_time)
    window_x = np.hanning(n_channels)
    window_2d = np.outer(window_x, window_t)
    data_windowed = data * window_2d

    # 2D FFT
    nfft_x, nfft_t = n_channels, n_time
    spec = fft2(data_windowed, s=(nfft_x, nfft_t))
    spec = fftshift(spec, axes=0)   # shift wavenumber axis to center
    spec = fftshift(spec, axes=1)   # shift frequency axis to center

    # Wavenumber and frequency axes
    kx = fftshift(fftfreq(nfft_x, d=dx))
    freq = fftshift(fftfreq(nfft_t, d=1/fs_das))

    # Keep only positive wavenumbers (propagation direction)
    pos_k_idx = kx > 0
    kx_pos = kx[pos_k_idx]
    spec_pos = spec[pos_k_idx, :]

    # Power spectrum
    power = np.abs(spec_pos)**2

    # Limit frequency range (1-200 Hz)
    freq_idx = (freq >= 1) & (freq <= 200)
    freq_pos = freq[freq_idx]
    power = power[:, freq_idx]

    # Convert to phase velocity: v = f / k
    K, F = np.meshgrid(kx_pos, freq_pos, indexing='ij')
    V = F / K                     # m/s
    V = V / 1000.0                # km/s

    # Normalize each frequency column independently
    power_norm = np.zeros_like(power)
    for i in range(power.shape[1]):
        col = power[:, i]
        min_val, max_val = np.min(col), np.max(col)
        if max_val > min_val:
            power_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            power_norm[:, i] = 0.5
    display_data = power_norm

    # Create masks to highlight specific points (used for dispersion curve picking)
    freq_grid = np.tile(freq_pos, (display_data.shape[0], 1))
    value_condition = display_data > 0.99
    speed_condition = (V > 0.7) & (V <= 1.6)
    freq_condition = ((freq_grid >= 10) & (freq_grid <= 25)) | \
                     ((freq_grid >= 32) & (freq_grid <= 50)) | \
                     ((freq_grid >= 55) & (freq_grid <= 60))
    combined_mask = value_condition & speed_condition & freq_condition

    x_highlight = freq_grid[combined_mask]
    y_highlight = V[combined_mask]
    amp_highlight = display_data[combined_mask]

    return freq_pos, V, display_data, x_highlight, y_highlight, amp_highlight

# ============================================
# Function: compute DAS-transverse f-v data
# ============================================
def compute_das_transverse():
    """
    Load DAS and geophone data, compute cross‑correlation per DAS channel,
    then perform 2D FFT to produce the DAS-transverse f-v dispersion image.
    Returns:
        freq: frequency axis (Hz)
        V: phase velocity axis (km/s)
        disp: normalized power spectrum (2D array)
        x_hl: frequency coordinates of highlighted points
        y_hl: velocity coordinates of highlighted points
        amp_hl: amplitude of highlighted points
    """
    # Load data
    das_data = np.load(das_file)          # shape (n_das, n_time_das)
    geo_data = np.load(geo_file)          # shape (n_geo, n_time_geo)

    # Select DAS channels and truncate time
    ch_start, ch_end = 200, 7200
    das_data = das_data[ch_start:ch_end, :2000]
    n_das, n_time_das = das_data.shape

    # Use geophone channel average as reference
    geo_stack = np.mean(geo_data, axis=0)
    geo_signal = geo_stack

    # Resample geophone signal to DAS sampling rate if needed
    if fs_geo != fs_das:
        n_target = int(len(geo_signal) * fs_das / fs_geo)
        geo_signal = resample(geo_signal, n_target)

    # Truncate to same length
    min_len = min(n_time_das, len(geo_signal))
    das_data = das_data[:, :min_len]
    geo_signal = geo_signal[:min_len]

    # Compute cross-correlation between each DAS channel and geophone signal
    ccf_list = []
    for i in range(n_das):
        x = detrend(das_data[i, :], type='constant')
        y = detrend(geo_signal, type='constant')
        corr = correlate(x, y, method='fft', mode='same')
        ccf_list.append(corr)
    ccf = np.array(ccf_list)               # shape (n_das, n_corr)

    # 2D FFT analysis
    window_t = np.hanning(ccf.shape[1])
    window_x = np.hanning(ccf.shape[0])
    window_2d = np.outer(window_x, window_t)
    ccf_windowed = ccf * window_2d

    nfft_x, nfft_t = ccf.shape[0], ccf.shape[1]
    spec = fft2(ccf_windowed, s=(nfft_x, nfft_t))
    spec = fftshift(spec, axes=0)           # shift wavenumber to center
    spec = fftshift(spec, axes=1)           # shift frequency to center

    kx = fftshift(fftfreq(nfft_x, d=dx))
    freq = fftshift(fftfreq(nfft_t, d=1/fs_das))

    # Keep only positive wavenumbers
    pos_k_idx = kx > 0
    kx_pos = kx[pos_k_idx]
    spec_pos = spec[pos_k_idx, :]

    power = np.abs(spec_pos)**2

    # Limit frequency range
    freq_idx = (freq >= 1) & (freq <= 200)
    freq_pos = freq[freq_idx]
    power = power[:, freq_idx]

    # Convert to phase velocity (m/s -> km/s)
    K, F = np.meshgrid(kx_pos, freq_pos, indexing='ij')
    V = F / K
    V = V / 1000.0

    # Normalize each frequency column
    power_norm = np.zeros_like(power)
    for i in range(power.shape[1]):
        col = power[:, i]
        min_val, max_val = np.min(col), np.max(col)
        if max_val > min_val:
            power_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            power_norm[:, i] = 0.5
    display_data = power_norm

    # Create masks to highlight specific points
    freq_grid = np.tile(freq_pos, (display_data.shape[0], 1))
    value_condition = display_data > 0.99
    speed_condition = (V > 0.7) & (V <= 1.6)
    freq_condition = ((freq_grid >= 10) & (freq_grid <= 25)) | \
                     ((freq_grid >= 32) & (freq_grid <= 50)) | \
                     ((freq_grid >= 55) & (freq_grid <= 60))
    combined_mask = value_condition & speed_condition & freq_condition

    x_highlight = freq_grid[combined_mask]
    y_highlight = V[combined_mask]
    amp_highlight = display_data[combined_mask]

    return freq_pos, V, display_data, x_highlight, y_highlight, amp_highlight

# ============================================
# Helper function: plot dispersion segments with optional smoothing and monotonicity
# ============================================
def plot_dispersion_segments(ax, x, y, bands, freq_thresh=0.5, vel_thresh=0.05,
                             color='red', linewidth=1.5, smooth=True, num_points=200,
                             spline_s=None, median_filter=False):
    """
    Plot smooth or straight line segments connecting selected points.
    Points are connected only if they are within the given frequency and velocity
    thresholds, preventing large jumps across gaps.
    The resulting curve is forced to be monotonically decreasing (velocity decreases
    as frequency increases).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x, y : array-like
        Frequency and velocity coordinates.
    bands : list of tuples
        List of frequency intervals (low, high) to consider.
    freq_thresh, vel_thresh : float
        Maximum allowed jumps in frequency and velocity between connected points.
    color, linewidth : matplotlib parameters.
    smooth : bool
        If True, draw smooth curves using cubic spline with optional smoothing.
    num_points : int
        Number of points for interpolation when smooth=True.
    spline_s : float or None
        Smoothing factor for UnivariateSpline. If None, it is set to 0.01 * len(x_unique).
    median_filter : bool
        If True, apply median filter to the selected points before connection.
    """
    for low, high in bands:
        # Select points in current frequency band
        mask = (x >= low) & (x <= high)
        x_band = x[mask]
        y_band = y[mask]
        if len(x_band) < 2:
            continue

        # Sort by frequency
        idx_sorted = np.argsort(x_band)
        x_sorted = x_band[idx_sorted]
        y_sorted = y_band[idx_sorted]

        # Optional median filtering to remove local outliers
        if median_filter:
            window = max(3, len(y_sorted) // 10)   # base window size
            # Ensure kernel size is odd (required by medfilt)
            if window % 2 == 0:
                window += 1
            y_sorted = medfilt(y_sorted, kernel_size=window)

        # Greedy connection: keep points that are close to the last kept one
        kept = [0]  # indices in sorted array
        for i in range(1, len(x_sorted)):
            last = kept[-1]
            if (abs(x_sorted[i] - x_sorted[last]) <= freq_thresh) and \
               (abs(y_sorted[i] - y_sorted[last]) <= vel_thresh):
                kept.append(i)

        if len(kept) < 2:
            continue

        # Extract kept points
        x_kept = x_sorted[kept]
        y_kept = y_sorted[kept]

        # Enforce monotonic decreasing (velocity decreases with frequency)
        # Since x_kept is increasing, we need y_kept to be non‑increasing.
        y_mono = y_kept.copy()
        for i in range(1, len(y_mono)):
            if y_mono[i] > y_mono[i-1]:
                y_mono[i] = y_mono[i-1]   # force decreasing

        if smooth and len(kept) >= 3:
            # Remove duplicate x values (if any)
            x_unique, indices = np.unique(x_kept, return_index=True)
            y_unique = y_mono[indices]
            if len(x_unique) >= 3:
                # Use cubic spline with smoothing
                if spline_s is None:
                    spline_s = 0.01 * len(x_unique)   # default smoothing factor
                spl = UnivariateSpline(x_unique, y_unique, s=spline_s, k=3)
                x_fine = np.linspace(x_unique.min(), x_unique.max(), num_points)
                y_fine = spl(x_fine)
                ax.plot(x_fine, y_fine, color=color, linewidth=linewidth,
                        linestyle='-', marker='')
            else:
                # Not enough points for cubic, fall back to straight line
                ax.plot(x_kept, y_mono, color=color, linewidth=linewidth,
                        linestyle='-', marker='')
        else:
            # Straight line segments (already forced monotonic)
            ax.plot(x_kept, y_mono, color=color, linewidth=linewidth,
                    linestyle='-', marker='')

# ============================================
# Main: create 2x2 subplots
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
ax1, ax2, ax3, ax4 = axes.flatten()

# Compute data for both methods
freq1, V1, disp1, x_hl1, y_hl1, amp1 = compute_das_das()
freq2, V2, disp2, x_hl2, y_hl2, amp2 = compute_das_transverse()

# ========== (a) DAS-DAS f-v plot ==========
mesh1 = ax1.pcolormesh(freq1, V1, disp1, shading='auto', cmap='jet')
ax1.set_ylim(0.5, 2.0)
ax1.set_xlim(1, 60)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Raleigh wave phase velocity (km/s)')
ax1.set_title('(a)')
fig.colorbar(mesh1, ax=ax1, label='Normalized amplitude')

# ========== (b) Rayleigh wave dispersion curve ==========
# Scatter plot of highlighted points
ax2.scatter(x_hl1, y_hl1, s=10, marker='o', facecolors='none',
            edgecolors='red', linewidth=0.5)
# Smooth line segments connecting points (monotonic enforced)
bands = [(10, 25), (32, 50), (55.2, 60)]
plot_dispersion_segments(ax2, x_hl1, y_hl1, bands,
                         freq_thresh=10, vel_thresh=0.1,
                         color='red', linewidth=1.5, smooth=True, num_points=300,
                         spline_s=0.05,          # adjust for smoothness
                         median_filter=True)     # enable median filtering

# Add English legend: red circle = dispersion points; red line = fitted curve
legend_ax2 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='red', markersize=7, markeredgewidth=0.5,
           label='Rayleigh dispersion point'),
    Line2D([0], [0], color='red', linewidth=1.5,
           label='Rayleigh dispersion curve')
]
ax2.legend(handles=legend_ax2, loc='upper left', framealpha=0.9)

ax2.set_xlim(15, 60)
ax2.set_ylim(0.75, 1.35)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Raleigh wave phase velocity (km/s)')
ax2.set_title('(b)')

# ========== (c) DAS-transverse f-v plot ==========
mesh2 = ax3.pcolormesh(freq2, V2, disp2, shading='auto', cmap='jet')
ax3.set_ylim(0.5, 2.0)
ax3.set_xlim(1, 60)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Love wave phase velocity (km/s)')
ax3.set_title('(c)')
fig.colorbar(mesh2, ax=ax3, label='Normalized amplitude')

# ========== (d) Love wave dispersion curve ==========
ax4.scatter(x_hl2, y_hl2, s=10, marker='x', color='blue', linewidth=0.5)
plot_dispersion_segments(ax4, x_hl2, y_hl2, bands,
                         freq_thresh=10, vel_thresh=0.1,
                         color='blue', linewidth=1.5, smooth=True, num_points=300,
                         spline_s=0.05,          # adjust for smoothness
                         median_filter=True)     # enable median filtering

legend_ax4 = [
    Line2D([0], [0], marker='x', color='blue', markersize=7, linewidth=0.5,linestyle='None',
           label='Love dispersion point'),
    Line2D([0], [0], color='blue', linewidth=1.5,
           label='Love dispersion curve')
]
ax4.legend(handles=legend_ax4, loc='upper left', framealpha=0.9)

ax4.set_xlim(15, 60)
ax4.set_ylim(0.75, 1.35)
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Love wave phase velocity (km/s)')
ax4.set_title('(d)')

plt.show()