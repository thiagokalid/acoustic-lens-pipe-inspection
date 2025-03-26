from framework import file_m2k
import numpy as np
from numpy import power
import matplotlib, scipy
import matplotlib.pyplot as plt
import gc
from matplotlib.ticker import FuncFormatter
import time
matplotlib.use('TkAgg')
font = {
    'weight' : 'normal',
    'size'   : 9
}
linewidth = 6.3091141732 # LaTeX linewidth

# Set the default font to DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rc('font', **font)

from framework.post_proc import envelope
from pipe_lens.imaging_utils import fwhm, convert_time2radius
from tqdm import tqdm

#%% Chooses which acoustic lens geoemtry to use:
root = '../data/resolution/passive_dir/'

# ii = 2

angular_location = [
    0, # degree
    10, # degree
    20
]

num_versions = 3

passive_flaw_widths = np.zeros(shape=(len(angular_location), num_versions), dtype=float)

#%%
for ii in range(len(angular_location)):
    for vv in range(num_versions):
        version = f"v{vv+1}"
        current_ang_location = angular_location[ii]


        data = file_m2k.read(root + f"passive_dir_{current_ang_location}degree_{version}.m2k",
                             freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
        data_ref = file_m2k.read(root + f"ref.m2k",
                                 freq_transd=5, bw_transd=0.5, tp_transd='gaussian')


        n_shots = data.ascan_data.shape[-1]
        log_cte = 1e-6
        time_grid = data.time_grid[:, 0]
        ang_span = np.linspace(-45, 45, 181)

        # It was manually identified where the outer and inner surface was (in microseconds):
        t_outer, t_inner = 56.13, 63.22
        r_span = convert_time2radius(time_grid, t_outer, t_inner, 5.9, 1.483, 1.483)

        # Seta os vetores dos dos dados
        widths, heights, peaks, = np.zeros(n_shots), np.zeros(n_shots), np.zeros(n_shots)


        for i in tqdm(range(n_shots)):
            channels = data.ascan_data[..., i]
            channels_ref = np.mean(data_ref.ascan_data, axis=3)
            sscan = np.sum(channels - channels_ref, axis=2)
            sscan_db = 20 * np.log10(envelope(sscan / sscan.max(), axis=0) + log_cte)

            # Aplica API para descobrir a área acima de -6 dB
            corners = [(64, -8 + current_ang_location), (55, +8 + current_ang_location)]

            widths[i], heights[i], peaks[i], pixels_above_threshold, = fwhm(sscan, r_span, ang_span, corners)

            if False:
                plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], time_grid[-1], time_grid[0]], cmap='inferno', aspect='auto', interpolation="none")
                plt.imshow(pixels_above_threshold, extent=[ang_span[0], ang_span[-1], r_span[-1], r_span[0]], cmap='inferno', aspect='auto', interpolation="none")

        # Finding FWHM
        xspan = np.arange(0, n_shots) - np.where(peaks == peaks.max())[0]

        minimum = np.min([peaks[np.where(xspan == -5)], peaks[np.where(xspan == 5)]])

        normalized_peaks = (peaks - minimum) / (peaks.max() - minimum)
        peaks_percentage = normalized_peaks * 100

        peaks_interp = lambda x: np.interp(x, xspan, normalized_peaks)

        cost_fun = lambda x: power(peaks_interp(x) - .5, 2)

        half_peak_loc_left = scipy.optimize.minimize(cost_fun, 0, bounds=[(xspan[0], 0)]).x
        half_peak_loc_right = scipy.optimize.minimize(cost_fun, 0, bounds=[(0, xspan[-1])]).x
        passive_flaw_width = half_peak_loc_right[0] - half_peak_loc_left[0]
        passive_flaw_widths[ii, vv] = passive_flaw_width
        print(f"FWHM of {current_ang_location} degree ({version}) = {passive_flaw_width:.2f}")

        if angular_location == 0 and version == "v1":
            # %% Plots results:

            fig, ax = plt.subplots(figsize=(linewidth * .49, linewidth * .4))
            plt.plot(xspan, peaks_percentage, ":o", color="k")
            plt.xlabel("Passive direction movement / (mm)")
            plt.ylabel("Relative amplitude / (%)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
            plt.ylim([-10, 110])
            plt.yticks(np.arange(0, 125, 25))
            plt.xlim([-4.5, 4.5])
            ytemp = np.arange(20, 60, 1)
            plt.xticks(np.linspace(-4, 4, 5))
            plt.grid(alpha=.5)
            plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)
            plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=2)

            ax.annotate("", xy=(half_peak_loc_left, 25), xytext=(half_peak_loc_right, 25),
                        arrowprops=dict(arrowstyle="<->", color='red', alpha=.8, linewidth=2),
                        ha="center",  # Center text horizontally
                        va="bottom"  # Position text below arrow
                        )
            ax.annotate(rf"${passive_flaw_width:.2f}$ mm", xy=(0.22, 25), xytext=(0.22, 30),
                        ha="center",  # Center text horizontally
                        va="bottom"  # Position text below arrow
                        )

            plt.tight_layout()
            plt.savefig("../figures/passive_dir_resolution.pdf")
            plt.show()

        del data, data_ref, sscan, sscan_db, pixels_above_threshold, channels, channels_ref
        gc.collect()
        time.sleep(15)
        gc.collect()

#%%


fig, ax = plt.subplots(figsize=(linewidth * .49, linewidth * .4))
plt.errorbar(angular_location, np.mean(passive_flaw_widths, axis=1), np.std(passive_flaw_widths, axis=1), color='red', ls='None', marker='o', capsize=5, capthick=1, ecolor='black', markersize=3)
plt.xlabel(r"$\alpha$-axis / (degrees)")
plt.ylabel("FWHM / (mm)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.1f}"))
plt.ylim([1, 3.5])
plt.yticks(np.arange(1, 4, .5))
plt.xticks(np.linspace(0, 30, 4))
plt.xlim([-5, 25])
plt.grid(alpha=.5)
plt.plot(half_peak_loc_left * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=1)
plt.plot(half_peak_loc_right * np.ones_like(ytemp), ytemp, 'r', alpha=.8, linewidth=1)


plt.tight_layout()
plt.savefig("../figures/passive_dir_resolution_different_angles.pdf")
plt.show()

