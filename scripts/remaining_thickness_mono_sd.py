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

path = '../data/remaining_thickness/single_element/'
c = 5893.426

second_row_results_0deg = list()
third_row_results_0deg = list()
fourth_row_results_0deg = list()

second_row_results_45deg = list()
third_row_results_45deg = list()
fourth_row_results_45deg = list()

for arq in os.listdir(path):
    print(arq.split('.')[0])
    if '2nd' in arq and '0deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        second_row = [bisect(t_span, 20), bisect(t_span, 22.5)]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        a_scan_crop = a_scan_ext[second_row[0]:second_row[1]]
        t_span_crop = t_span[second_row[0]:second_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        second_row_results_0deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()

    elif '2nd' in arq and '45deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        a_scan_crop = a_scan_ext[second_row[0]:second_row[1]]
        t_span_crop = t_span[second_row[0]:second_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        second_row_results_45deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()

    elif '3rd' in arq and '0deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        third_row = [bisect(t_span, 19), bisect(t_span, 22)]

        a_scan_crop = a_scan_ext[third_row[0]:third_row[1]]
        t_span_crop = t_span[third_row[0]:third_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        third_row_results_0deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()

    elif '3rd' in arq and '45deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        third_row = [bisect(t_span, 19), bisect(t_span, 22)]

        a_scan_crop = a_scan_ext[third_row[0]:third_row[1]]
        t_span_crop = t_span[third_row[0]:third_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        third_row_results_45deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()

    elif '4th' in arq and '0deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        fourth_row = [bisect(t_span, 18), bisect(t_span, 19.4)]

        a_scan_crop = a_scan_ext[fourth_row[0]:fourth_row[1]]
        t_span_crop = t_span[fourth_row[0]:fourth_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        fourth_row_results_0deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()

    elif '4th' in arq and '45deg' in arq:
        data = file_m2k.read(path + arq, type_insp='contact', water_path=0, freq_transd=5,
                             bw_transd=0.5, tp_transd='gaussian', sel_shots=3)
        t_span = data.time_grid[:, 0]
        a_scan_ext = envelope(data.ascan_data[bisect(t_span, 4):, 0, 0], axis=0)
        t_span = t_span[bisect(t_span, 4):]

        ext_max_pos = np.unravel_index(np.argmax(a_scan_ext), shape=a_scan_ext.shape)
        ext_time = t_span[ext_max_pos[0]]
        ext_max = np.max(a_scan_ext)

        fourth_row = [bisect(t_span, 18), bisect(t_span, 19.4)]

        a_scan_crop = a_scan_ext[fourth_row[0]:fourth_row[1]]
        t_span_crop = t_span[fourth_row[0]:fourth_row[1]]

        flaw_max_pos = np.unravel_index(np.argmax(a_scan_crop), shape=a_scan_crop.shape)
        flaw_time = t_span_crop[flaw_max_pos[0]]
        flaw_max = np.max(a_scan_crop)

        remaining_thickness = ((flaw_time - ext_time) * 1e-3 * c) / 2
        fourth_row_results_45deg.append(remaining_thickness)

        plt.figure()
        plt.title(arq.split('.')[0])
        plt.plot(t_span, a_scan_ext)
        plt.plot(ext_time, ext_max, 'r.')
        plt.plot(flaw_time, flaw_max, 'b.')
        plt.show()


second_row_0deg_mean = np.mean(second_row_results_0deg)
second_row_45deg_mean = np.mean(second_row_results_45deg)
sd_second_row_0deg = np.std(second_row_results_0deg)
sd_second_row_45deg = np.std(second_row_results_45deg)

third_row_0deg_mean = np.mean(third_row_results_0deg)
third_row_45deg_mean = np.mean(third_row_results_45deg)
sd_third_row_0deg = np.std(third_row_results_0deg)
sd_third_row_45deg = np.std(third_row_results_45deg)

fourth_row_0deg_mean = np.mean(fourth_row_results_0deg)
fourth_row_45deg_mean = np.mean(fourth_row_results_45deg)
sd_fourth_row_0deg = np.std(fourth_row_results_0deg)
sd_fourth_row_45deg = np.std(fourth_row_results_45deg)

print(f"\n\nMédia segunda linha a 0º : {second_row_0deg_mean} && Desvio padrão: {sd_second_row_0deg}")
print(f"Média segunda linha a 45º : {second_row_45deg_mean} && Desvio padrão: {sd_second_row_45deg}")

print(f"\n\nMédia terceira linha a 0º : {third_row_0deg_mean} && Desvio padrão: {sd_third_row_0deg}")
print(f"Média terceira linha a 45º : {third_row_45deg_mean} && Desvio padrão: {sd_third_row_45deg}")

print(f"\n\nMédia quarta linha a 0º : {fourth_row_0deg_mean} && Desvio padrão: {sd_fourth_row_0deg}")
print(f"Média quarta linha a 45º : {fourth_row_45deg_mean} && Desvio padrão: {sd_fourth_row_45deg}")

