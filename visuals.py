from matplotlib import pyplot as plt

from utils import intervals_where

# cada ponto: 10 ms => cada linha = 500 pontos = 5 segundos
# 12 linhas = 1 min


def plot_mask_multiline(mask, columns=500):
    '''Plots a sequence of boolean values spanning multiple lines.

    The contiguous True values are aggregated into blocks and the False values
    are not plotted.'''

    lines_of_intervals = (intervals_where(mask[x:x + columns])
                          for x in range(0, len(mask), columns))

    # lines as sequences of (xmin, xwidth)
    lines_of_xranges = [
        [(start, end - start) for start, end in line]
        for line in lines_of_intervals]

    plt.axis([0, columns, len(lines_of_xranges), 0])
    for offset, xranges in enumerate(lines_of_xranges):
        plt.broken_barh(
            xranges, (offset, 0.8),
            facecolors=('blue', 'red', 'yellow', 'green') * len(xranges))
