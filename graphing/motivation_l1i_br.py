import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from scipy import interpolate
import numpy as np
from matplotlib.backend_bases import MouseButton
import palettable

plt.style.use("grayscale")
ax = plt.subplot(111)

# ax.set_prop_cycle('color', palettable.scientific.sequential.LaPaz_7.mpl_colors)
ax.set_prop_cycle(
    'color', palettable.scientific.sequential.GrayC_20_r.mpl_colors)

gray_color_set = palettable.scientific.sequential.GrayC_5_r.hex_colors
light_color_set = palettable.cartocolors.qualitative.Vivid_5.hex_colors

# keep all the data here
# IPC                     Hit rate
# IdealBR   1.47             63.5
# L1IHits   1.038            92.6
# baselineEP  1              54.9
# baseline  1                52.8

fig, (ax1, ax2) = plt.subplots(1, 2,
                               sharey='row', figsize=(100,100))

# first subplot
ax1.scatter(1, 54,
            marker=".", label="EP-Baseline", color=gray_color_set[1],  s=30)
ax1.scatter(1.0038, 92.6,
            marker="*", label="EP-L1I-Hits", color=gray_color_set[1],  s=30)

# second subplot
ax2.scatter(1, 52.8,
            marker="x", label="NoL1IPref-Baseline", color=gray_color_set[1],  s=30)
ax2.scatter(1.047, 63.5,
            marker="+", label="8BB-\u03BC-opCache-Hit", color=gray_color_set[1],  s=30)

fig.text(0.5, -0.1, 'IPC improvement (%)', ha='center', fontsize="8")
fig.text(0.02, 0.5, '\u03BC-op cache hit rate (%)', va='center', rotation='vertical', fontsize="8")

# # x_ticks = np.arange(1, 1.51, 0.1)
x_ticks = np.arange(1, 1.0061, 0.002)
# # x_ticks = [0,2,4,8,16,32]
y_ticks = np.arange(50, 100.1, 10)

plt.sca(ax1)
plt.xticks(x_ticks, fontsize="8")
plt.yticks(y_ticks, fontsize="8")
plt.sca(ax2)
x_ticks = np.arange(1, 1.06, 0.02)
plt.xticks(x_ticks, fontsize="8")
plt.yticks(y_ticks, fontsize="8")

ax1.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
ax1.grid(which='minor', color='black', alpha=0.1, linestyle='-', linewidth=0.5)
ax1.minorticks_on()
ax2.grid(which='major', color='black', linewidth=0.5, alpha=0.3)
ax2.grid(which='minor', color='black', alpha=0.1, linestyle='-', linewidth=0.5)
ax2.minorticks_on()


# # Add legend
ax1.legend()
ax1.legend(loc='upper center', ncol=1, bbox_to_anchor=(
    0.4, 1.25), framealpha=1, frameon=False, edgecolor="black", fontsize="7", handletextpad=0)
ax2.legend()
ax2.legend(loc='upper center', ncol=1, bbox_to_anchor=(
    0.4, 1.25), framealpha=1, frameon=False, edgecolor="black", fontsize="7", handletextpad=0)



fig = plt.gcf()  # Get the current figure
fig.set_size_inches(4, 2)

plt.savefig("motivation_l1i_vs_br.pdf", bbox_inches='tight')
