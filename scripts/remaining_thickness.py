import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect

from framework import file_m2k
from framework.post_proc import envelope

# Constants
linewidth = 6.3091141732
log_cte = 1e-6
c = 5900

# Define a color bar
vmin_sscan = -120
vmax_sscan = 0

path = "../data/remaining_thickness/"

data_ref = file_m2k.read(path + 'ref2.m2k', type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian', sel_shots=3)[1]

t_span = data_ref.time_grid[..., 0]
ang_span = np.linspace(-45, 45, 181)
ang_range = [[bisect(ang_span, -38), bisect(ang_span, -20)], [bisect(ang_span, 15), bisect(ang_span, 31)]]

ang_range_flaw_1 = ang_span[ang_range[0][0]:ang_range[0][1]]
ang_range_flaw_2 = ang_span[ang_range[1][0]:ang_range[1][1]]

times_range = [[bisect(t_span, 57.62), bisect(t_span, 58.9)], [bisect(t_span, 60.43), bisect(t_span, 61.64)],
               [bisect(t_span, 58.92), bisect(t_span, 60.4)]]

ext_env = envelope(np.sum(data_ref.ascan_data[:, ...], axis=2), axis=0)
ext_log = 20 * np.log10(ext_env + log_cte)
ext_norm = 20 * np.log10(ext_env / ext_env.max() + log_cte)

for arq in os.listdir(path):
    if 'linha' in arq:
        print(arq.split('.')[0] + "\n")
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian', sel_shots=3)[1]

        # Método convencional fazendo a subtração com a referencia SEM tubo
        data_env = envelope(np.sum(data.ascan_data[..., 0] - data_ref.ascan_data[..., 0], axis=2), axis=0)

        if 'segunda' in arq:
            t0 = times_range[1][0]
            tf = times_range[1][1]
            t_span_croped = t_span[t0:tf]
        elif 'terceira' in arq:
            t0 = times_range[2][0]
            tf = times_range[2][1]
            t_span_croped = t_span[t0:tf]
        else:
            t0 = times_range[0][0]
            tf = times_range[0][1]
            t_span_croped = t_span[t0:tf]

        # Cortando a regiao de ambas as falhas
        flaw_1 = data_env[t0:tf, ang_range[0][0]:ang_range[0][1], ...]
        flaw_2 = data_env[t0:tf, ang_range[1][0]:ang_range[1][1], ...]

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
        a_scan_ext_1 = data_env[:, int(bisect(ang_span, ang_flaw_1))]
        a_scan_ext_2 = data_env[:, int(bisect(ang_span, ang_flaw_2))]
        a_scan_flaw_1 = data_env[t0:tf, int(bisect(ang_span, ang_flaw_1))]

        # Obtendo o valor do maximo da externa apenas da primeira falha para plot exemplo do A-scan
        max_value = np.max(a_scan_ext_1)

        # Calculando o tempo da externa para cada falha separadamente
        ext_time_1 = t_span[np.argmax(a_scan_ext_1)]
        ext_time_2 = t_span[np.argmax(a_scan_ext_2)]

        # Plot do A-scan que vamos utilizar no artigo
        if 'segunda' in arq:
            plt.figure(figsize=(linewidth * .5, 3))
            plt.title("A_scan não saturado")
            plt.plot(t_span, a_scan_ext_1)
            plt.plot(ext_time_1, max_value, '.r', label='Máximo da externa')
            plt.plot(time_flaw_1, np.max(a_scan_flaw_1), '.b', label='Máximo da externa')
            plt.ylabel(r"Amplitude")
            plt.xlabel(r"Ângulo de varredura da tubulação")
            plt.legend()
            plt.show()

        # Calculo da espessura remanescente para ambas as falhas
        remaining_thickness_f1 = ((time_flaw_1 - ext_time_1) * 1e-3 * c) / 2
        remaining_thickness_f2 = ((time_flaw_2 - ext_time_2) * 1e-3 * c) / 2

        print(f"Espessura remanescente falha 1: {remaining_thickness_f1:.2f}mm")
        print(f"Espessura remanescente falha 2: {remaining_thickness_f2:.2f}mm \n")

        # S-scan para plot
        img_norm = 20 * np.log10(data_env / data_env.max() + log_cte)


        plt.figure(figsize=(linewidth * .5, 3))
        plt.title(arq.split('.')[0])
        plt.imshow(img_norm, aspect='auto', cmap='inferno', vmin=vmin_sscan, vmax=vmax_sscan,
                   extent=[ang_span[0], ang_span[-1], t_span[-1], t_span[0]], interpolation='None')

        plt.plot(ang_flaw_1, time_flaw_1, '.b', label="Máximo Falha")
        plt.plot(ang_flaw_2, time_flaw_2, '.b')
        plt.plot(ang_flaw_1, ext_time_1, '.r', label="Máximo da ext")
        plt.plot(ang_flaw_2, ext_time_2, '.r')

        plt.ylabel(r"Tempo em $\mu s$")
        plt.xlabel(r"Ângulo de varredura da tubulação")
        plt.legend(loc="lower left")
plt.show()