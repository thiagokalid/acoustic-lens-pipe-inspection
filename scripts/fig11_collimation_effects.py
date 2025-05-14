import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
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

#%%

root = "../data/collimation/"
filename = ["no_collimation", "20mm_collimation", "5mm_collimation"]

# Read the ultrassound data:
data = [file_m2k.read(root + filename + '.m2k', freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0) for filename in filename]

# Define the wedge aperture of 90 degrees:
delta_alpha = .1 # degree
alpha_max = 45
angle_span = np.arange(-alpha_max, alpha_max + delta_alpha, delta_alpha)

# Minor adjusment for enhanced plotting:
roll_idx = [20, 0, 0]

# Estimation of hardware maximum amplitude:
hardware_ceiling = 2**16 # its a 16bit channel
maximum_amplitude = hardware_ceiling * data[0].probe_params.num_elem

# Extract relevant data from ultrasound file:
time_span = data[0].time_grid
sscans = [np.sum(data.ascan_data[..., 0], axis=2) for data in data] # sum all channels to obtain S-Scan
ascans = [np.roll(sscan[:, 92], shift=roll, axis=0) for sscan, roll in zip(sscans, roll_idx)] # extract the a-scan allign with alpha = 0 degrees.
envelopes = [envelope(ascan, axis=0) for ascan in ascans]
relative_amplitudes = [envelope / maximum_amplitude * 100 for envelope in envelopes]

#%% Plots the results:

fig, ax = plt.subplots(figsize=(linewidth * .6 , 2.5))
plt.plot(time_span, relative_amplitudes[2], linestyle="-", color="#FF1F5B", label=r"5\,mm", linewidth=2)
plt.plot(time_span, relative_amplitudes[1], linestyle="-", color="#00CD6C", label=r"20\,mm", linewidth=1)
plt.plot(time_span, relative_amplitudes[0], linestyle="--", color='k', label="No collimation", linewidth=.5)

plt.xlim([52.5, 62.5])
plt.xlabel(r"Time / $(\mathbf{\mu s})$")
plt.ylabel(r"Amplitude / (\%)", labelpad=0)
plt.grid(axis='x', alpha=.25)
plt.grid(axis='y', alpha=.75)
ax.annotate('Top of the flaw', xy=(59.6, 3.5), xytext=(57.5, 7.4),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1))
ax.annotate('Inner surface', xy=(60.2, 11.6), xytext=(57.8, 15),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1))
ax.annotate('Outer surface', xy=(54.86, 10), xytext=(55.59, 11.5),
            arrowprops=dict(arrowstyle="-|>", color='black', alpha=1, linewidth=1))
ax.legend(loc='upper center', ncol=3, fancybox=False, shadow=False, columnspacing=1, framealpha=.5)
plt.xticks(np.arange(52, 64, 2))
plt.yticks(np.arange(0, 27.5, 5))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"{x:.0f}"))
plt.tight_layout()

plt.savefig("../figures/collimation_effects.pdf")
plt.show()