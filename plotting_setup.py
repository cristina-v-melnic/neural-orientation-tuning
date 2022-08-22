from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

cmap = cm.get_cmap('turbo_r')

# Parameters used for plotting.
# No need to change unless the plotting style is suboptimal.
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ]),
})

sns.set(font_scale = 2.5)
sns.set_style("white")

params = {'font.size': 21,
          'legend.handlelength': 2,
          'legend.frameon': False,
          'legend.framealpha': 0.5,
          'figure.figsize': [10.4, 8.8],
          'lines.markersize': 20.0,
          'lines.linewidth': 3.5,
          'axes.linewidth': 3.5,
          'xtick.major.width': 3.5,
          'xtick.minor.width': 3.5,
          'ytick.major.width': 3.5,
          'ytick.minor.width': 3.5,
          'axes.spines.top': False,
          'axes.spines.right': False,

          'font.family': ['sans-serif'],
          'figure.autolayout': True
          # 'lines.dashed_pattern': [3.7, 1.6]
          }
plt.rcParams.update(params)