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
from bisect import bisect
import scipy

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

tbeg, tend = bisect(time_grid, 58.5), bisect(time_grid, 60.5)
ang_beg, ang_end = bisect(ang_span, -10), bisect(ang_span, 20)
sscan_windowed = sscan_envelope[tbeg:tend, ang_beg:ang_end]

vmin = np.log10(sscan_windowed.max())
vmax = np.log10(sscan_windowed.min())
#%%

plt.figure(figsize=(linewidth*.48, 2.5))
plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='YlGnBu', aspect='auto', interpolation='None', vmin=vmin, vmax=vmax)
ytemp = np.arange(40, 65, 1)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
plt.ylim([60.5, 58.5])
plt.xlim([-10, 20])
plt.colorbar()

plt.tight_layout()
plt.savefig("../figures/fwhm_sscan.pdf")
plt.show()


#%%
max_per_col = np.max(sscan_windowed, axis=1)
max_per_col_normalized = (max_per_col - max_per_col.min()) / (max_per_col.max() - max_per_col.min())

max_loc = float(time_grid[tbeg + np.argmax(max_per_col_normalized)])

peaks_interp = lambda x: np.interp(x, time_grid[tbeg:tend], max_per_col_normalized)
cost_fun = lambda x: np.power(peaks_interp(x) - .5, 2)

half_peak_loc_left = scipy.optimize.minimize(cost_fun, time_grid[tbeg], bounds=[(time_grid[tbeg], max_loc)]).x
half_peak_loc_right = scipy.optimize.minimize(cost_fun, ang_span[ang_beg], bounds=[(max_loc, time_grid[tend])]).x
fwhm_col = half_peak_loc_right[0] - half_peak_loc_left[0]

fwhm_col = fwhm_col / 2 * 5900 * 1e-6

fig, ax = plt.subplots(figsize=(linewidth*.5, 2.5))
plt.plot(time_grid[tbeg:tend], max_per_col_normalized * 100, "o:", color='k', linewidth=1.5, markersize=1)
plt.xlim([58.5, 60])
plt.ylim([-5, 110])
plt.yticks(np.arange(0, 125, 25))

plt.xlabel(r"Time / ($\mathbf{\mu s}$)")
plt.ylabel(r"Normalized amplitude / (\%)", labelpad=0)
plt.grid(alpha=.5)

ytemp = np.arange(20, 60, 1)
plt.grid(alpha=.5)
plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

ax.annotate("", xy=(half_peak_loc_left, 25), xytext=(half_peak_loc_right, 25),
            arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
mid = (half_peak_loc_left + half_peak_loc_right)/2
ax.annotate(rf"${fwhm_col * 1e3:.2f}$\,mm", xy=(mid, 25), xytext=(mid, 30),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )



plt.tight_layout()
plt.savefig("../figures/fwhm_radial_axis.pdf")
plt.show()

#%%
max_per_row = np.max(sscan_windowed, axis=0)
max_per_row_normalized = (max_per_row - max_per_row.min()) / (max_per_row.max() - max_per_row.min())

max_loc = float(ang_span[ang_beg + np.argmax(max_per_row_normalized)])

peaks_interp = lambda x: np.interp(x, ang_span[ang_beg:ang_end], max_per_row_normalized)
cost_fun = lambda x: np.power(peaks_interp(x) - .5, 2)

half_peak_loc_left = scipy.optimize.minimize(cost_fun, ang_span[ang_beg], bounds=[(ang_span[ang_beg], max_loc)]).x
half_peak_loc_right = scipy.optimize.minimize(cost_fun, ang_span[ang_beg], bounds=[(max_loc, 10)]).x
fwhm_row = half_peak_loc_right[0] - half_peak_loc_left[0]

fig, ax = plt.subplots(figsize=(linewidth*.5, 2.5))
plt.plot(ang_span[ang_beg:ang_end], max_per_row_normalized * 100, "o:", color='k', linewidth=1.5, markersize=4)
plt.xlim([-0, 12.5  ])
plt.ylim([-5, 110])
plt.yticks(np.arange(0, 125, 25))

plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel(r"Normalized amplitude / (\%)", labelpad=0)
plt.grid(alpha=.5)

ytemp = np.arange(20, 60, 1)
plt.grid(alpha=.5)
plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

ax.annotate("", xy=(half_peak_loc_left, 25), xytext=(half_peak_loc_right, 25),
            arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
mid = (half_peak_loc_left + half_peak_loc_right)/2
ax.annotate(rf"${fwhm_row:.2f}^\circ$", xy=(mid, 25), xytext=(mid, 30),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )


plt.tight_layout()
plt.savefig("../figures/fwhm_alpha_axis.pdf")
plt.show()