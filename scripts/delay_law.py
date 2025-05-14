import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from framework import file_m2k
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline
from pipe_lens.raytracer import RayTracer
from pipe_lens.transducer import Transducer
from numpy import pi, sin, cos
from bisect import bisect
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

def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

root = '../data/echoes/'
acoustic_lens_type = "xl"
plot_fig = True # Chooses to either plot or save the output.

#%%
data = file_m2k.read(root + f"only_tube.m2k", type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
time_grid = data.time_grid

# Acoustic lens parameters:
c1 = 6332.93 # in (m/s)
c2 = 1430.00 # in (m/s)q
d = 170e-3 # in (m)
alpha_max = pi/4 # in (rad)
alpha_0 = 0  # in (rad)
h0 = 91.03e-3 # in (m)

acoustic_lens = AcousticLens(c1, c2, d, alpha_max, alpha_0, h0)

# Pipeline-related parameters:
radius = 70e-3
wall_width = 20e-3
inner_radius = (radius - wall_width)
c3 = 5900
pipeline = Pipeline(radius, wall_width, c3)

# Ultrasound phased array transducer specs:
transducer = Transducer(pitch=.5e-3, bw=.4, num_elem=64, fc=5e6)
transducer.zt += acoustic_lens.d

# Raytracer engine to find time of flight between emitter and focus:
raytracer = RayTracer(acoustic_lens, pipeline, transducer)

#%% Plane wave delay law:

delta_alpha = np.deg2rad(.5)
alpha_max = pi/4
alpha_min = -pi/4
alpha_span = np.arange(alpha_min, alpha_max + delta_alpha, delta_alpha)

theta_span = acoustic_lens.pipe2steering_angle(alpha_span)

delay_law_pw = np.zeros([181,64]) # Transmission law is equal to reception law.
for i in range(181):
    if theta_span[i] >= 0:
        delay_law_pw[i, :] = transducer.pitch / acoustic_lens.c1 * sin(theta_span[i]) * (transducer.elements - 1)
    else:
        delay_law_pw[i, :] = transducer.pitch / acoustic_lens.c1 * abs(sin(theta_span[i])) * (transducer.num_elem - transducer.elements)

#%% Focused wave delay law:

focus_radius = inner_radius
focus_angle = np.copy(alpha_span)
xf, zf = focus_radius * sin(focus_angle), focus_radius * cos(focus_angle)

solutions = raytracer.solve(xf, zf)

delay_focused = np.zeros([181,64]) # Transmission law is equal to reception law.
for n in range(transducer.num_elem):
    sol = solutions[n]
    d1 = dist(transducer.xt[n], transducer.zt[n], sol['xlens'], sol['zlens'])
    d2 = dist(sol['xlens'], sol['zlens'], sol['xpipe'], sol['zpipe'])
    d3 = dist(sol['xpipe'], sol['zpipe'], xf, zf)
    tof = d1/c1 + d2/c2 + d3/c3
    delay_focused[:, n] = tof
delay_law_focused = delay_focused.max() - delay_focused
delay_law_focused = np.asarray([delay_law_focused[i, :] - delay_law_focused[i, :].min() for i in range(delay_law_focused.shape[0])])

#%%
fig, ax = plt.subplots(figsize=(linewidth * .45, 2.25))

# Curves:
i = bisect(alpha_span, np.deg2rad(2.5))

plt.plot(transducer.elements, delay_law_pw[i, :] * 1e6, 'o', color='k', linewidth=1.5, label="Plane waves", markersize=2)
plt.plot(transducer.elements, delay_law_focused[i, :] * 1e6, 'x', color='#FF1F5B', linewidth=1.5, label="Focused waves", markersize=4)
plt.legend(framealpha=0.5)

# Axes:
plt.xticks(np.arange(0, 64 + 8, 8))
plt.xlim([-8, 64 + 8])
plt.ylabel(r"$\Delta t$ / ($\mathrm{\mu s}$)")
plt.xlabel(r"Element index")
min_ytick = np.min([delay_law_pw[i, :], delay_law_focused[i, :]]) * 1e6
plt.yticks(np.arange(0, .3, .05))
plt.grid(alpha=.5)

# Formating axes:
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fr"${x:.2f}$ %"))

plt.tight_layout()
plt.show()
plt.savefig("../figures/delay_law.pdf", bbox_inches="tight")
