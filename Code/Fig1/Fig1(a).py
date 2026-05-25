import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw={'projection': ccrs.SouthPolarStereo()})
ax.set_extent([-170, 45, -89.94, -90], crs=ccrs.PlateCarree())

# Gridlines
lon_ticks = np.arange(-180, 180, 60)
lat_ticks = np.arange(-90, -89.9, 0.02)
gl = ax.gridlines(draw_labels=True, xlocs=lon_ticks, ylocs=lat_ticks,
                  x_inline=False, y_inline=False,
                  linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Coordinate transformations
proj = ccrs.SouthPolarStereo()
geodetic = ccrs.PlateCarree()

# Endpoint coordinates
south_pole_lonlat = (0, -90)
seismometer_lonlat = (145, -89.93)

south_pole_xy = proj.transform_point(*south_pole_lonlat, geodetic)
seismometer_xy = proj.transform_point(*seismometer_lonlat, geodetic)

# Plot South Pole and Seismometer
ax.plot(*south_pole_lonlat, 'ko', markersize=10, transform=geodetic, label='South Pole')
ax.plot(*seismometer_lonlat, 'kv', markersize=10, transform=geodetic, label='QSPA seismometer')

# DAS (straight line in projection coordinates)
x0, y0 = south_pole_xy
x1, y1 = seismometer_xy
ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2, label='DAS cable')

# Geophones (12 internal equally spaced points)
n_segments = 13
t = np.linspace(0, 1, n_segments + 1)
t_interp = t[1:-1]
x_interp = x0 + t_interp * (x1 - x0)
y_interp = y0 + t_interp * (y1 - y0)
ax.plot(x_interp, y_interp, 'bv', markersize=10, linestyle='none', label='Fairfield ZLand geophones')

ax.plot(142, -89.997, 'r*', markersize=14, transform=geodetic, label='Active source')

# SPICE Core
ax.plot(-105, -89.992, 'gs', markersize=10, transform=geodetic, label='SPICE ice core')

# Legend with larger font
ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8, labelspacing=1.0)

ax.set_title('(a)')
plt.show()