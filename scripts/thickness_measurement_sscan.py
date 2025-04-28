import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt    
from pipe_lens.imaging_utils import convert_time2radius
from framework import file_m2k
from framework.post_proc import envelope
linewidth = 6.3091141732 # LaTeX linewidth

matplotlib.use('TkAgg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "font.weight": "normal",
})

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/echoes/'
data = file_m2k.read(root + f"pit_inspection.m2k",
                     freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

log_cte = 1e-6
theoretical_maximum = 2 ** 16 * data.probe_params.num_elem
time_grid = data.time_grid[:, 0]
ang_span = np.linspace(-45, 45, 181)

# It was manually identified where  the front and back wall was (in microseconds):
t_outer, t_inner = 55.27, 60.75
rtop, rbottom = 62.0, 58.0

r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)


channels = data.ascan_data[..., 0]
sscan = np.sum(channels, axis=2)
sscan_envelope = envelope(sscan, axis=0)
sscan_envelope = (sscan_envelope - sscan_envelope.min()) / (sscan_envelope.max() - sscan_envelope.min())
sscan_db = np.log10(sscan_envelope + log_cte)


#%%

plt.figure(figsize=(linewidth*.48, 2.5))
plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='YlGnBu', aspect='auto', interpolation='None', vmin=-6, vmax=0)
ytemp = np.arange(40, 65, 1)
plt.plot(np.ones_like(ytemp) * 7, ytemp, "--", color='black', linewidth=1)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / $(\mathrm{\mu s})$")
plt.ylim([63.45, 54.3])
plt.colorbar()

plt.tight_layout()
plt.savefig("../figures/sscan_backwall.pdf")
plt.show()


#%%

fig, ax = plt.subplots(figsize=(linewidth*.5, 2.5))
plt.plot(time_grid, sscan_envelope[:, 104] * 100, color='k', linewidth=1.5)
plt.xlim([54.3, 63.45])
plt.ylim([-.5, 13])
plt.xlabel(r"Time / $(\mathbf{\mu s})$")
plt.ylabel(r"Amplitude / (\%)", labelpad=0)
plt.grid(alpha=.5)

ax.annotate("", xy=(55.7, 10), xytext=(59.3, 10),
            arrowprops=dict(arrowstyle="|-|", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r"$\tau_w$", xy=(55.7,10.5), xytext=(57.5, 10.5),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

ax.annotate("Pit", xy=(58.9, 1.5), xytext=(58.5, 3),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

plt.tight_layout()
plt.savefig("../figures/ascan_backwall.pdf")
plt.show()