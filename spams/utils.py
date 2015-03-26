import os
os.environ['MPLCONFIGDIR'] = "/local/.config/matplotlib"
print(os.environ.get('MPLCONFIGDIR'))

import matplotlib.pyplot as plt
import matplotlib
from pylab import xticks
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)


def autolabel(rects, name):
    for ii, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, '%s' % (name[ii]), ha='center', va='bottom')


def draw_barplot(values, x_ticks=None, xlabel="x", ylabel="y", title="plot", save_as=None, **kwargs):
    rects = plt.bar(range(len(values)), values, align='center', **kwargs)
    xticks(range(len(values)), x_ticks)
    plt.xlabel(xlabel, fontsize = 30, fontweight='bold')
    plt.ylabel(ylabel, fontsize = 30, fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=36)
    #autolabel(rects, values)
    fig = plt.gcf()
    fig.set_size_inches((20, 16))
    if save_as:
        fig.savefig(save_as, dpi=100)
        plt.close(fig)
    else:
        plt.show()
