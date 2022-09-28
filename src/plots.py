from plotting_setup import *
from parameters import *

# Stimulus_sweep plots
def plot_soma_response(x, y, err, name, name_file = "tuning_curve", PO = [], plots_directory = sweep_directory):

    plt.plot(x, y, marker='o', alpha=1.0, linewidth="2", markersize="10", label="$f$",
             color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    plt.fill_between(x, y - err, y + err, alpha=0.3, color='gray')
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Postsynaptic $f$ (Hz)")

    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    if svg_enable == True: plt.savefig(sweep_directory + "output_f_theta_{}.svg".format(name_file))
    plt.savefig(sweep_directory + "output_f_theta_{}.png".format(name_file))
    plt.figure()

# Plots that reproduce figure 3 from Scholl et al. 2021
def plot_fig_3a(x, y1, y2, y3, std1, std2, std3, plots_directory = sweep_directory):

    # The tuning curve
    plt.plot(x, y1, marker='o', alpha=1.0, linewidth="2", markersize="10", label="$f$",
             color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    plt.fill_between(x, y1 - std1, y1 + std1, alpha = 0.3, color='gray')
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Postsynaptic $f$ (Hz)")

    # The cumulative synaptic strength
    # Excitatory input.
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker='D' ,alpha=0.9, linewidth="2", markersize="10", label="$W_{E}$",
             color='tab:orange', markeredgewidth=1.5, markeredgecolor="tab:orange")
    ax2.fill_between(x, y2 - std2, y2 + std2, alpha=0.3, color='tab:orange')

    ax2.plot(x, y3, marker='D', alpha=0.9, linewidth="2", markersize="10", label="$W_{I}$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="tab:blue")
    ax2.fill_between(x, y3 - std3, y3 + std3, alpha=0.3, color='tab:blue')

    ax2.set_ylabel("Cumulative $W$ (a.u)")
    ax2.spines.right.set_visible(True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.legend()

    if svg_enable == True: plt.savefig(sweep_directory + "fig3a.svg")
    plt.savefig(sweep_directory + "fig3a.png")
    plt.figure()

def plot_fig_3b(x, y1, y2, std1, std2, y3, y4, std3, std4, plots_directory = sweep_directory):
    # Excitatory 3b.
    # Number of active synapses for a given stimulus.
    plt.plot(x, y1, marker='D', alpha=1.0, linewidth="2", markersize="20", label="$w$", color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    plt.fill_between(x, y1 - std1, y1 + std1, alpha=0.3, color='gray')
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Individual $\langle w \\rangle$ (a.u.)")
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    # Individual synaptic strength of a typical active afferent.
    ax = plt.gca()
    ax.yaxis.label.set_color('gray')
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker='H', alpha=0.9, linewidth="2", markersize="20", label="$N_{E}$", color='tab:orange', markeredgewidth=1.5)
    ax2.fill_between(x, y2 - std2, y2 + std2, alpha=0.3, color='tab:orange')
    ax2.set_ylabel("# Active synapses")
    ax2.yaxis.label.set_color('tab:orange')
    ax2.spines.right.set_visible(True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    if svg_enable == True: plt.savefig(sweep_directory + "fig3b_ex.svg")
    plt.savefig(sweep_directory + "fig3b_ex.png")
    plt.figure()

    # Inhibitory 3b.
    # Number of active synapses for a given stimulus.
    plt.plot(x, y3, marker='D', alpha=0.9, linewidth="2", markersize="20", label="$w$",color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    plt.fill_between(x, y3 - std3, y3 + std3, alpha=0.3, color='gray')
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Individual $\langle w \\rangle$ (a.u.)" )
    plt.locator_params(axis='y', nbins=5)

    # Individual synaptic strength of a typical active afferent.
    ax = plt.gca()
    ax.yaxis.label.set_color('gray')
    ax2 = ax.twinx()
    ax2.plot(x, y4, marker='H', alpha=0.9, linewidth="2", markersize="20", label="$N_{I}$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="tab:blue")
    ax2.fill_between(x, y4 - std4, y4 + std4, alpha=0.3, color='tab:blue')
    ax2.set_ylabel("# Active synapses")
    ax2.yaxis.label.set_color('tab:blue')
    ax2.spines.right.set_visible(True)
    plt.locator_params(axis='y', nbins=5)

    if svg_enable == True: plt.savefig(sweep_directory + "fig3b_inh.svg")
    plt.savefig(sweep_directory + "fig3b_inh.png")
    plt.figure()

# Plots that reproduce figure 2 from Scholl et al. 2021
def plot_PO_vs_weight(x, y, name = '', binary = False):

    # Get distinct plot parameters for excitatory/inhibitory cases.
    if name == 'exc':
       po_dif = np.sort(np.abs(x))

       # Get a different color map for binary/continuous cases.
       if binary == True:
           t = np.zeros(len(x))
           for l in range(len(x)):
               if po_dif[l] != 0: t[l] = 1.0
       else:
           t = np.linspace(0.0, 1.0, len(po_dif))

       plt.scatter(po_dif, np.log(y), marker = "^", s = (y*150)**2, alpha = 0.8, color = cmap(t), edgecolors = "white", label = "Excitatory")
       plt.xlim([-10,98])
    else:
        plt.scatter(np.abs(x), np.log(y), marker = 'o', s = 800, alpha = 0.8, color = 'tab:blue',  edgecolors = "white", label = "Inhibitory")

    plt.xlabel("PO difference (deg)")
    plt.ylabel("Synaptic weight (log a.u.)")
    plt.locator_params(axis='y', nbins=5)

    if svg_enable == True: plt.savefig(sweep_directory + "PO_vs_weight_{}.svg".format(name))
    plt.savefig(sweep_directory + "PO_vs_weight_{}.png".format(name))
    plt.figure()

# Postsynaptic_train plots
def plot_v_trace(t_f, v_series, name_V = "V(t)"):
    # Plotting the trace.
    t = np.linspace(0, t_f, len(v_series))

    plt.plot(t, v_series, color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane potential (mV)")
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    if svg_enable == True: plt.savefig(integration_directory + name_V + ".svg")
    plt.savefig(integration_directory + name_V + ".png")
    plt.figure()

def plot_current_zoomed(t_f_zoom, I_in, I_ex, name_i = "I(t)_zoomed"):
    # Plotting the currents.
    t_f_zoom = t_f_zoom * dt / 1000
    t = np.linspace(0, t_f_zoom, len(I_in))

    plt.plot(t, I_in, color="tab:blue", label="Inhibitory", alpha=0.9, linewidth=1)
    plt.plot(t, I_ex, color="tab:orange", label="Excitatory", alpha=0.9, linewidth=1)
    plt.plot(t, np.asarray(I_in) + np.asarray(I_ex), color="black", label="Net", alpha=1.0, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Membrane currents (nA)")
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    plt.legend(labelcolor='linecolor')
    if svg_enable == True: plt.savefig(integration_directory + name_i + ".svg")
    plt.savefig(integration_directory + name_i + ".png")
    plt.figure()

def plot_voltage_zoomed(t_f_zoom, v_zoom_series, name_V = "V(t)"):
    t = np.linspace(0, t_f_zoom, len(v_zoom_series))
    plt.plot(t * 0.0001, v_zoom_series, color="gray", linewidth=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane potential (mV)")
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    if svg_enable == True: plt.savefig(integration_directory + name_V + "zoom.svg")
    plt.savefig(integration_directory + name_V + "zoom.png")
    plt.figure()
