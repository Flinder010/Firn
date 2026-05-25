import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

depth_obs = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,54,64,74,84,94,114,144,174,200])
density_obs = np.array([0.3631,0.3836,0.4112,0.4325,0.4615,0.4722,0.4968,0.5067,0.5151,0.5284,0.5396,0.5478,0.5537,0.5632,0.5733,0.5864,0.5916,0.5968,0.6035,0.6116,0.6190,0.6259,0.6292,0.6348,0.6431,0.6526,0.6816,0.7062,0.7328,0.7605,0.7796,0.8489,0.9085,0.9170])

depth_full = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,54,64,74,84,94,114,144,174,200])

kohnen_full = np.array([0.3466,0.4253,0.4520,0.4748,0.4951,0.5137,0.5303,0.5452,0.5588,0.5711,0.5822,0.5923,0.6018,0.6103,0.6183,0.6260,0.6329,0.6395,0.6459,0.6518,0.6576,0.6625,0.6676,0.6727,0.6776,0.6915,0.7121,0.7305,0.7482,0.7644,0.7948,0.8357,0.8703,0.8960])

diez_full = np.array([0.3380,0.4717,0.4982,0.5168,0.5318,0.5445,0.5557,0.5655,0.5745,0.5829,0.5906,0.5980,0.6050,0.6117,0.6184,0.6249,0.6310,0.6372,0.6431,0.6488,0.6545,0.6600,0.6654,0.6707,0.6762,0.6918,0.7158,0.7382,0.7598,0.7801,0.8190,0.8711,0.9119,0.9170])

yang_p_full = np.array([0.3628,0.4284,0.4503,0.4691,0.4858,0.5010,0.5147,0.5271,0.5382,0.5485,0.5577,0.5662,0.5739,0.5812,0.5881,0.5945,0.6003,0.6061,0.6113,0.6164,0.6213,0.6257,0.6303,0.6346,0.6387,0.6509,0.6690,0.6856,0.7020,0.7169,0.7463,0.7884,0.8270,0.8600])

yang_s_full = np.array([0.3614,0.4708,0.4917,0.5062,0.5179,0.5278,0.5364,0.5441,0.5511,0.5577,0.5636,0.5691,0.5745,0.5797,0.5848,0.5898,0.5946,0.5994,0.6038,0.6083,0.6126,0.6169,0.6212,0.6252,0.6293,0.6412,0.6599,0.6775,0.6947,0.7111,0.7428,0.7885,0.8305,0.8664])

rho_ice = 0.917

def porosity(rho_bulk):
    phi = 1 - rho_bulk / rho_ice
    phi[phi < 0] = 0
    return phi

porosity_obs = porosity(density_obs)
porosity_kohnen = porosity(kohnen_full)
porosity_diez = porosity(diez_full)
porosity_yang_p = porosity(yang_p_full)
porosity_yang_s = porosity(yang_s_full)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

ax1.plot(density_obs, depth_obs, 'k-o', lw=1.5, ms=3, label='Observed (SPICEcore)')
ax1.plot(kohnen_full, depth_full, 'b-s', lw=1.5, ms=3, label='Kohnen et al. (1973, P-wave)')
ax1.plot(diez_full, depth_full, 'r-^', lw=1.5, ms=3, label='Diez et al. (2015, S-wave)')
ax1.plot(yang_p_full, depth_full, 'g-d', lw=1.5, ms=3, label='Yang et al. (2024, P-wave)')
ax1.plot(yang_s_full, depth_full, 'm-p', lw=1.5, ms=3, label='Yang et al. (2024, S-wave)')
ax1.set_xlabel('Density ($\mathrm{g}/\mathrm{cm}^{3}$)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('(a)')
ax1.legend(loc='lower left')

ax1.axhline(y=10, color='gray', ls='--', lw=1.2, alpha=0.7)
ax1.axhline(y=54, color='gray', ls='--', lw=1.2, alpha=0.7)
ax1.axhline(y=140, color='gray', ls='--', lw=1.2, alpha=0.7)

ax1.text(0.33, 5, 'Snow layer', ha='left', va='center')
ax1.text(0.33, 32, 'Firn layer', ha='left', va='center')
ax1.text(0.33, 97, 'Mixture of firn and ice', ha='left', va='center')
ax1.text(0.33, 150, 'Ice layer', ha='left', va='center')

ax1.set_ylim(200, 0)
ax1.set_yticks(np.arange(0, 210, 50))

ax2.plot(porosity_obs, depth_obs, 'k-o', lw=1.5, ms=3, label='Observed (SPICEcore)')
ax2.plot(porosity_kohnen, depth_full, 'b-s', lw=1.5, ms=3, label='Kohnen et al. (1973, P-wave)')
ax2.plot(porosity_diez, depth_full, 'r-^', lw=1.5, ms=3, label='Diez et al. (2015, S-wave)')
ax2.plot(porosity_yang_p, depth_full, 'g-d', lw=1.5, ms=3, label='Yang et al. (2024, P-wave)')
ax2.plot(porosity_yang_s, depth_full, 'm-p', lw=1.5, ms=3, label='Yang et al. (2024, S-wave)')
ax2.set_xlabel('Porosity')
ax2.set_ylabel('Depth (m)')
ax2.set_title('(b)')
ax2.legend(loc='lower right')

ax2.axhline(y=10, color='gray', ls='--', lw=1.2, alpha=0.7)
ax2.axhline(y=54, color='gray', ls='--', lw=1.2, alpha=0.7)
ax2.axhline(y=140, color='gray', ls='--', lw=1.2, alpha=0.7)

ax2.text(-0.06, 5, 'Snow layer', ha='left', va='center', transform=ax2.transData)
ax2.text(-0.06, 32, 'Firn layer', ha='left', va='center', transform=ax2.transData)
ax2.text(-0.06, 97, 'Mixture of firn and ice', ha='left', va='center', transform=ax2.transData)
ax2.text(-0.06, 150, 'Ice layer', ha='left', va='center', transform=ax2.transData)

ax2.set_ylim(200, 0)
ax2.set_yticks(np.arange(0, 210, 50))
ax2.set_xlim(-0.08, 0.7)

plt.tight_layout()
plt.show()