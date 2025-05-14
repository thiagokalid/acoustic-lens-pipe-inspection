import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from framework.post_proc import envelope
from framework import file_m2k

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

path = "../data/echoes/"

data = file_m2k.read(path + "amplitude_profile.m2k", type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian', sel_shots=3)

time_grid = data.time_grid
ang_span = np.linspace(-45, 45, 181)

# Constants
log_cte = 1e-6
c = 5893.426

channels = data.ascan_data
sscan = np.sum(channels, axis=2)
sscan_envelope = envelope(sscan, axis=0)
sscan_envelope = (sscan_envelope - sscan_envelope.min()) / (sscan_envelope.max() - sscan_envelope.min())
sscan_log = np.log10(sscan_envelope + log_cte)

#%% Figure 1:

fig, ax = plt.subplots(figsize=(linewidth * .5, 2.5))
plt.imshow(sscan_log, aspect='auto', cmap='YlGnBu', vmin=-6, vmax=0,
                   extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]])

ytemp = np.arange(-1, 101)
plt.ylim([80, 0])
plt.plot(ytemp * 0, ytemp, "--", color='k', linewidth=1)
plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
plt.xlabel(r"$\alpha$-axis / (degrees)", labelpad=0)
plt.colorbar()

# Annotations:
ax.annotate(r'${1\tau_{\mathrm{lens}}(\alpha)}$',
            xy=(5.9, 25),
            xytext=(25, 15),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1.5),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'${2\tau_{\mathrm{lens}}(\alpha)}$',
            xy=(5.9, 49),
            xytext=(25, 39),
    color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1.5),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate(r'${\tau_{\mathrm{outer}}}$',
            xy=(-40, 55),
            xytext=(-40 + 11, 45),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1.5),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'${\tau_{\mathrm{inner}}}$',
            xy=(-40, 63),
            xytext=(-40 + 11, 78),
            color='black',
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1.5),
            ha="center",  # Center text horizontally
            va="bottom",  # Position text below arrow
            weight='bold',
            style='italic'
            )

plt.tight_layout()
plt.savefig("../figures/amplitude_profile_sscan.pdf")
plt.show()

#%%

sscan_col = sscan_envelope[:, 91] * 100

fig, ax = plt.subplots(figsize=(linewidth*.5, 2.5))
plt.plot(time_grid, sscan_col, color='k', linewidth=1.5)
plt.ylabel(r"Amplitude / (\%)", labelpad=0)
plt.xlabel(r"Time / $\mathrm{(\mu s)}$")
plt.grid(alpha=0.25)
plt.xlim([20, 65])
plt.ylim([-5, 105])


ax.annotate(r'$1\tau_{\mathrm{lens}}(0^\circ)$',
            xy=(25.6, 75),
            xytext=(35, 85),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$2\tau_{\mathrm{lens}}(0^\circ)$',
            xy=(48.1, 13),
            xytext=(43, 35),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{\mathrm{outer}}$',
            xy=(56, 13),
            xytext=(53, 45),
    color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r'$\tau_{\mathrm{inner}}$',
            xy=(62.53, 7.2),
            xytext=(59.53, 39),
    color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

plt.tight_layout()
plt.savefig("../figures/amplitude_profile_signal.pdf")
plt.show()
