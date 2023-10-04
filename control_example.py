# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

icor_control = lambda t: min(3 * np.exp(-(t - 2023) / 50), 3)

world3 = World3(pyear=2020)
world3.set_world3_control(icor_control=icor_control)
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

plot_world_variables(
    world3.time,
    [world3.nrfr, world3.iopc, world3.fpc, world3.pop, world3.ppolx],
    ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
    [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    figsize=(7, 5),
    title="World3 standard run",
)
plt.savefig("fig_world3_standard_a.pdf")

plot_world_variables(
    world3.time,
    [world3.fcaor, world3.io, world3.tai, world3.aiph, world3.fioaa],
    ["FCAOR", "IO", "TAI", "AI", "FIOAA"],
    [[0, 1], [0, 4e12], [0, 4e12], [0, 2e2], [0, 0.201]],
    figsize=(7, 5),
    title="World3 standard run - Capital sector",
)
plt.savefig("fig_world3_standard_b.pdf")

plot_world_variables(
    world3.time,
    [world3.ly, world3.al, world3.fpc, world3.lmf, world3.pop],
    ["LY", "AL", "FPC", "LMF", "POP"],
    [[0, 4e3], [0, 4e9], [0, 8e2], [0, 1.6], [0, 16e9]],
    figsize=(7, 5),
    title="World3 standard run - Agriculture sector",
)
plt.savefig("fig_world3_standard_c.pdf")

plot_world_variables(
    world3.time,
    [world3.fioaa1],
    ["icor"],
    [None],
    figsize=(7, 5),
    title="World3 standard run - Population sector",
)
plt.savefig("fig_world3_standard_d.pdf")
