from plotting_setup import *
import numpy as np

def plot_soma_response(x, y, err, name, PO = []):
    if name == 'PO':
        plt.scatter([x[np.argmax(y)]], [np.min(y)], alpha=1.0, marker='x' , s=50, color = 'tab:red', label ="PO")
        if len(PO) != 0:
            plt.text(PO[0] + 2, np.min(y), s="{} $\pm$ {:.2f} deg".format(PO[0], PO[1]))
        else:
            plt.text(x[np.argmax(y)]+2, np.min(y), s = "{} deg".format(x[np.argmax(y)]))
        plt.xlabel("Stimulus $\\theta$ (deg)")
        plt.ylabel("Postsynaptic $f$ (Hz)")
        #ax = plt.gca()
        #ax2 = ax.twinx()
        #ax2.spines.right.set_visible(True)
        #ax2.set_ylabel("Individual trials $f$ (Hz)")
        #ax2.yaxis.label.set_color('gray')

    elif name == "delta_PO":
        #plt.xlabel("PO difference (deg)")
        plt.xlabel("Stimulus $\\theta$ (deg)")
        plt.ylabel("Postsynaptic $f$ (Hz)")
        #plt.plot(x, nr_active, label = "Number of active synapses")
    else:
        #plt.xlabel("PO difference (deg)")
        plt.xlabel("Stimulus $\\theta$ (deg)")
        plt.ylabel("Number of active {} synapses (Hz)".format(name), labelpad = 2)
    plt.plot(x, y, marker='o', alpha=0.9, linewidth="2", markersize="10", label="$\langle f \\rangle$",
             color='gray', markeredgewidth=1.5)
    #plt.errorbar(x, y, yerr=err, fmt='none', color='black')
    plt.fill_between(x, y - err, y + err, alpha=0.3, color='gray')

    #plt.legend()

    # plt.title("Showing a bar of $\\theta$ orientation, 100 tuned synapses at 0 rad with weights = 0.01")
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.savefig("output_f_theta_{}.svg".format(name))
    plt.figure()

def plot_fig_3a(x, y1, y2, y3, std1, std2, std3):
    #stop = int(len(x) / 2)
    #x = x[:stop]
    #y1 =y1[:stop]
    #y2 = y2[:stop]
    #y3 = y3[:stop]
    #std1 = std1[:stop]
    #std2 = std2[:stop]
    #std3 = std3[:stop]


    plt.plot(x, y1, marker='o', alpha=1.0, linewidth="2", markersize="10", label="$f$",
             color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    #plt.errorbar(x, y1, yerr=std1, fmt='none', color='black', barsabove = True)
    plt.fill_between(x, y1 - std1, y1 + std1, alpha = 0.3, color='gray')

    #plt.xlabel("PO difference (deg)")
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Postsynaptic $f$ (Hz)")

    ax = plt.gca()
    #ax.yaxis.label.set_color('gray')
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker='D' ,alpha=0.9, linewidth="2", markersize="10", label="$W_{E}$",
             color='tab:orange', markeredgewidth=1.5, markeredgecolor="tab:orange")
    ax2.fill_between(x, y2 - std2, y2 + std2, alpha=0.3, color='tab:orange')
    #ax2.errorbar(x, y2, yerr=std2, fmt='none', color='tab:red', barsabove = True)
    ax2.plot(x, y3, marker='D', alpha=0.9, linewidth="2", markersize="10", label="$W_{I}$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="tab:blue")
    ax2.fill_between(x, y3 - std3, y3 + std3, alpha=0.3, color='tab:blue')
    #ax2.errorbar(x, y3, yerr=std3, fmt='none', color='blue', barsabove = True)
    ax2.set_ylabel("Cumulative $W$ (a.u)")
    ax2.spines.right.set_visible(True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    #plt.xlim([-95,95])
    plt.legend()
    plt.savefig("fig3a.svg")
    plt.savefig("fig3a.png")
    plt.figure()

def plot_fig_3b(x, y1, y2, std1, std2, y3, y4, std3, std4):
    #plt.plot(x, np.log(y1), marker='D', alpha=0.9, linewidth="2", markersize="20", label="Median weight",
    #        color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    #plt.fill_between(x, np.log(y1 - std1), np.log(y1 + std1), alpha=0.3, color='gray')
    #stop = int(len(x) / 2)
    #x = x[:stop]
    #y1 = y1[:stop]
    #y2 = y2[:stop]
    #y3 = y3[:stop]
    #y4 = y4[:stop]
    #std1 = std1[:stop]
    #std2 = std2[:stop]
    #std3 = std3[:stop]
    #std4 = std4[:stop]


    plt.plot(x, y1, marker='D', alpha=1.0, linewidth="2", markersize="20", label="$w$", color='gray', markeredgewidth=1.5, markeredgecolor="gray")

    # plt.errorbar(x, np.log(y1), yerr=np.log(std1), fmt='none', color='gray', barsabove=True)
    plt.fill_between(x, y1 - std1, y1 + std1, alpha=0.3, color='gray')
    #plt.xlabel("PO difference (deg)")
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Individual $\langle w \\rangle$ (a.u.)")
    #plt.legend()
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    ax = plt.gca()
    #plt.yscale('log')
    #Axis.set_minor_formatter(ax.yaxis, ScalarFormatter())
    #ax.ticklabel_format(style='sci')
    ax.yaxis.label.set_color('gray')
    ax2 = ax.twinx()
    #ax.set_yscale('log')
    ax2.plot(x, y2, marker='H', alpha=0.9, linewidth="2", markersize="20", label="$N_{E}$", color='tab:orange', markeredgewidth=1.5)
    ax2.fill_between(x, y2 - std2, y2 + std2, alpha=0.3, color='tab:orange')
    #ax2.errorbar(x, y2, yerr=std2, fmt='none', color='tab:orange', barsabove=True)
    ax2.set_ylabel("# Active synapses")
    ax2.yaxis.label.set_color('tab:orange')
    ax2.spines.right.set_visible(True)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    #plt.xlim([-95, 95])
    #plt.legend()
    plt.savefig("fig3b_ex.svg")
    plt.savefig("fig3b_ex.png")
    plt.figure()

    #plt.plot(x, np.log(y3), marker='D', alpha=0.9, linewidth="2", markersize="20", label="Average weight", color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    #plt.errorbar(x, np.log(y3), yerr=np.log(std3), fmt='none', color='gray', barsabove=True)
    #plt.fill_between(x, np.log(y3 - std3), np.log(y3 + std3), alpha=0.3, color='gray')
    plt.plot(x, y3, marker='D', alpha=0.9, linewidth="2", markersize="20", label="$w$",color='gray', markeredgewidth=1.5, markeredgecolor="gray")
    #plt.errorbar(x, np.log(y3), yerr=np.log(std3), fmt='none', color='gray', barsabove=True)
    plt.fill_between(x, y3 - std3, y3 + std3, alpha=0.3, color='gray')

    #plt.xlabel("PO difference (deg)")
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Individual $\langle w \\rangle$ (a.u.)" )
    #plt.legend()
    plt.locator_params(axis='y', nbins=5)
    ax = plt.gca()
    ax.yaxis.label.set_color('gray')
    ax2 = ax.twinx()
    #ax.set_yscale('log')
    #Axis.set_minor_formatter(ax.yaxis, ScalarFormatter())
    #ax.ticklabel_format(style='sci')
    ax2.plot(x, y4, marker='H', alpha=0.9, linewidth="2", markersize="20", label="$N_{I}$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="tab:blue")
    #ax2.errorbar(x, y4, yerr=std4, fmt='none', color='blue', barsabove=True)
    ax2.fill_between(x, y4 - std4, y4 + std4, alpha=0.3, color='tab:blue')
    ax2.set_ylabel("# Active synapses")
    ax2.yaxis.label.set_color('tab:blue')
    ax2.spines.right.set_visible(True)
    #plt.xlim([-100, 100])
    plt.locator_params(axis='y', nbins=5)
    #plt.legend()
    plt.savefig("fig3b_inh.svg")
    plt.savefig("fig3b_inh.png")
    plt.figure()

def plot_PO_vs_weight(x, y, name = '', binary = False):
    if name == 'exc':
       # t = np.linspace(0.0,1.0,len(x))
       po_dif = np.sort(np.abs(x))
       if binary == True:
           t = np.zeros(len(x))
           for l in range(len(x)):
               if po_dif[l] != 0:
                   t[l] = 1.0
       else:
           t = np.linspace(0.0, 1.0, len(po_dif))
           #for l in range(len(x)):
           #    if po_dif[l] == 0:
           #        t[l] = 0.0
       #plt.scatter(po_dif, np.log(y), marker="^", s=20*(100-po_dif), alpha=0.8, color = cmap(t), edgecolors="white", label = "Excitatory")
       plt.scatter(po_dif, np.log(y), marker="^", s=(y*150)**2, alpha=0.8, color=cmap(t), edgecolors="white",
                   label="Excitatory")
       plt.xlim([-10,98])
    else:
        plt.scatter(np.abs(x), np.log(y), marker='o', s=800, alpha=0.8, color = 'tab:blue',  edgecolors="white", label = "Inhibitory")
    plt.xlabel("PO difference (deg)")

    #plt.yscale('log')
    plt.ylabel("Synaptic weight (log a.u.)")
    plt.locator_params(axis='y', nbins=5)
    #ax = plt.gca()
    #Axis.set_minor_formatter(ax.yaxis, ScalarFormatter())
    #ax.ticklabel_format(style='sci')
    #plt.xlim([-1, 95])

    #plt.legend()
    plt.savefig("PO_vs_weight_{}.svg".format(name))
    plt.savefig("PO_vs_weight_{}.png".format(name))
    plt.figure()

