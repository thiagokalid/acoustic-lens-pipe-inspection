import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from numpy import pi
from matplotlib import pyplot as plt
from pipe_lens.acoustic_lens import AcousticLens
linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})


#%%
c1 = 6332.93 # in (m/s)
c2 = 1430.00 # in (m/s)
d = 170e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 91.03e-3 # in (m)

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0)

alpha = np.arange(0, alpha_max, 1e-3)
h = acoustic_lens.h(alpha)
theta = acoustic_lens.pipe2steering_angle(alpha)


#%% Figure 1:

plt.figure(figsize=(linewidth * .45, 2.5))
plt.plot(np.rad2deg(theta), np.rad2deg(alpha), color='k', linewidth=1.5)
mask = (np.deg2rad(4) <= theta) * (theta <= np.deg2rad(5))
plt.fill_between(np.rad2deg(theta[mask]),
                 np.rad2deg(alpha[mask]), color='lightgray')
plt.fill_betweenx(np.rad2deg(alpha[mask]),
                 np.rad2deg(theta[mask]), color='lightgray')

mask = (np.deg2rad(25.4) <= theta) * (theta <= np.deg2rad(26.4))
plt.fill_between(np.rad2deg(theta[mask]),
                 np.rad2deg(alpha[mask]), color='lightgray')
plt.fill_betweenx(np.rad2deg(alpha[mask]),
                 np.rad2deg(theta[mask]), color='lightgray')

plt.xlabel(r'$\theta$ / (degrees)')
plt.ylabel(r'$\alpha$ / (degrees)')
plt.axis([0, plt.axis()[1], 0, plt.axis()[3]])
plt.xticks(np.arange(0, 30, 5))
plt.yticks(np.arange(0, 50, 10))
plt.ylim([0, 45])
plt.grid(alpha=.2)

plt.tight_layout()
plt.savefig('../figures/theta_alpha.pdf')
plt.show()

#%% Figure 2:

dalpha = np.diff(alpha) / np.diff(theta)
plt.figure(figsize=(linewidth * .45, 2.5))
plt.plot(np.rad2deg(alpha[:-1]), dalpha, color='k', linewidth=1.5)

plt.xlabel(r'$\alpha$ / (degrees)')
plt.ylabel(r'$\mathrm{d\alpha/d\theta}$')
plt.grid(alpha=.2)
plt.xticks(np.arange(0, 60, 15))
plt.xlim([0, 45])

plt.tight_layout()
plt.savefig('../figures/dalpha_dtheta.pdf')
plt.show()
