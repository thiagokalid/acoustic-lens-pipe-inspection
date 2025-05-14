import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from framework.post_proc import envelope
from framework import file_m2k
from bisect import bisect
from tqdm import tqdm
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

path = '../data/remaining_thickness/'
# Constants
log_cte = 1e-6
c_steel = 5893.426

# Define a color bar
vmin_sscan = -6
vmax_sscan = 0


acoustic_lens_results = {
    "2nd": {"0deg": [], "-45deg": []},
    "3rd": {"0deg": [], "-45deg": []},
    "4th": {"0deg": [], "-45deg": []},
}

data_ref = file_m2k.read(path + 'ref2.m2k', type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
channels_ref = data_ref.ascan_data

t_span = data_ref.time_grid[..., 0]
ang_span = np.linspace(-45, 45, 181)


ang_range = [
    [-38, -20], # First CBH
    [15, 30]    # Second CBH
]

times_range = [
    [60.3, 61.80], # 2nd
    [58.18, 61.20], # 3rd
    [57.5, 60]  # 4th
]

ang_range_flaw_1 = ang_span[bisect(ang_span, ang_range[0][0]): bisect(ang_span, ang_range[0][1])]
ang_range_flaw_2 = ang_span[bisect(ang_span, ang_range[1][0]): bisect(ang_span, ang_range[1][1])]

ext_env = envelope(np.sum(data_ref.ascan_data[:, ...], axis=2), axis=0)
ext_log = np.log10(ext_env + log_cte)
ext_norm = np.log10(ext_env / ext_env.max() + log_cte)

rows = ["2nd", "3rd", "4th"]
versions = ["v1", "v2", "v3"]

for i in tqdm(range(len(rows) * len(versions))):
    row_idx = i // len(versions)  # Integer division to get the row index
    version_idx = i % len(versions)  # Modulo to get the version index


    row = rows[row_idx]  # Get the current row based on row index
    version = versions[version_idx]  # Get the current version based on version index

    filename = f"{row}_row_{version}.m2k"
    data = file_m2k.read(path + filename, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

    channels = data.ascan_data
    sscan = np.sum(channels - channels_ref, axis=2)
    sscan_env = envelope(sscan, axis=0)
    sscan_log = np.log10(sscan_env + log_cte)


    t0 = np.argmin(np.power(t_span - times_range[row_idx][0], 2))
    tf = np.argmin(np.power(t_span - times_range[row_idx][1], 2))
    t_span_croped = t_span[t0:tf]

    # Cortando a regiao de ambas as falhas

    a, b = bisect(ang_span, ang_range[0][0]), bisect(ang_span, ang_range[0][1])
    c, d = bisect(ang_span, ang_range[1][0]), bisect(ang_span, ang_range[1][1])

    flaw_1 = sscan_env[t0:tf, a:b, 0]
    flaw_2 = sscan_env[t0:tf, c:d, 0]

    # Obtendo posição de ambas as falhas
    max_pos_flaw_1 = np.unravel_index(np.argmax(flaw_1), shape=flaw_1.shape)
    max_pos_flaw_2 = np.unravel_index(np.argmax(flaw_2), shape=flaw_2.shape)

    # Tempo correspondente a cada falha
    time_flaw_1 = t_span_croped[max_pos_flaw_1[0]]
    time_flaw_2 = t_span_croped[max_pos_flaw_2[0]]

    # Ângulo correspondente a cada falha
    ang_flaw_1 = ang_range_flaw_1[max_pos_flaw_1[1] - 1]
    ang_flaw_2 = ang_range_flaw_2[max_pos_flaw_2[1] - 1]

    # A-scans utilizados para retirar a posição da externa de cada falha
    a_scan_ext_1 = sscan_env[:, int(bisect(ang_span, ang_flaw_1))]
    a_scan_ext_2 = sscan_env[:, int(bisect(ang_span, ang_flaw_2))]
    a_scan_flaw_1 = sscan_env[t0:tf, int(bisect(ang_span, ang_flaw_1))]
    a_scan_flaw_2 = sscan_env[t0:tf, int(bisect(ang_span, ang_flaw_2))]

    if True:
        # Plotting S-scans with identified scatterers:
        plt.figure(figsize=(8,6))
        plt.subplot(2, 1, 1)
        plt.suptitle(filename)

        sscan_plot = (sscan_env - sscan_env.min()) / (sscan_env.max() - sscan_env.min())
        sscan_db = np.log10(sscan_plot + 1e-6)


        plt.title("S-scan (log-scale)")
        plt.imshow(sscan_db, extent=[ang_span[0], ang_span[-1], t_span[-1], t_span[0]], aspect='auto', vmin=-6, vmax=0, cmap='viridis')
        plt.plot(ang_flaw_1, time_flaw_1, 'xr', markersize=6)
        plt.plot(ang_flaw_2, time_flaw_2, 'ob', markersize=6)
        plt.ylabel(r"Time / ($\mathrm{\mu s}$)")
        plt.xlabel(r"$\alpha$-axis / (degrees)")
        plt.grid(alpha=.5)

        plt.subplot(2, 1, 2)
        plt.title("S-scan column and it's peak.")
        plt.plot(t_span_croped, a_scan_flaw_1, color='r')
        plt.plot(time_flaw_1, a_scan_flaw_1.max(), 'xr')
        plt.plot(t_span_croped, a_scan_flaw_2, 'b')
        plt.plot(time_flaw_2, a_scan_flaw_2.max(), 'ob')
        plt.xlabel(r"Time / ($\mathrm{\mu s}$)")
        plt.grid(alpha=.75)
        plt.tight_layout()
        plt.show()

    # Obtendo o valor do maximo da externa apenas da primeira falha para plot exemplo do A-scan
    max_value = np.max(a_scan_ext_1)

    # Calculando o tempo da externa para cada falha separadamente
    ext_time_1 = t_span[np.argmax(a_scan_ext_1)]
    ext_time_2 = t_span[np.argmax(a_scan_ext_2)]


    # Calculo da espessura remanescente para ambas as falhas
    remaining_thickness_f1 = ((time_flaw_1 - ext_time_1) * 1e-3 * c_steel) / 2
    remaining_thickness_f2 = ((time_flaw_2 - ext_time_2) * 1e-3 * c_steel) / 2

    acoustic_lens_results[row]["0deg"].append(remaining_thickness_f1)
    acoustic_lens_results[row]["-45deg"].append(remaining_thickness_f2)


#%%

degs = ["-45deg", "0deg"]

results = {row: {} for row in rows}
for row in rows:
    for deg in degs:
        results[row][deg] = {"mean": 0, "std": 0, "error": 0, "measured": 0}

# Measured values with analog thickness gauge:
results["2nd"]["-45deg"]["measured"] = 13.61
results["3rd"]["-45deg"]["measured"] = 9.22
results["4th"]["-45deg"]["measured"] = 4.87

results["2nd"]["0deg"]["measured"] = 13.73
results["3rd"]["0deg"]["measured"] = 9.28
results["4th"]["0deg"]["measured"] = 4.90

for deg in degs:
    for row in rows:
        results[row][deg]["mean"] = np.mean(acoustic_lens_results[row][deg])
        results[row][deg]["std"] = np.std(acoustic_lens_results[row][deg])
        results[row][deg]["error"] = results[row][deg]["mean"] - results[row][deg]["measured"]
        print(f"Row: {row}, Deg: {deg} ===> mean = {results[row][deg]['mean']:.2f}; std = {results[row][deg]['std']:.2f}; error = {results[row][deg]['error']:.2f}")

