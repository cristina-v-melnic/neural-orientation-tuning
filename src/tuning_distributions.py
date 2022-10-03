from parameters import *
import matplotlib.pyplot as plt
from utils import *



def tune_binary(N_excit_synapses = N_excit_synapses,
                N_inhib_synapses = N_inhib_synapses,
                theta1 = np.pi/4, theta2 = 3*np.pi/4,
                to_plot = False):

    tuning_angles_in = np.ones(N_inhib_synapses)*theta1
    tuning_angles_in[0:int(N_inhib_synapses / 2)] = theta2
    # OS from <w>
    tuning_angles_ex = np.ones(N_excit_synapses)*theta1
    tuning_angles_ex[0:int(N_excit_synapses/4)] = theta2

    if to_plot == True:
        plt.hist(tuning_angles_ex * 180 / np.pi, label="$N_{E}$", alpha=0.8, color='tab:orange')
        plt.hist(tuning_angles_in * 180 / np.pi, label="$N_{I}$", alpha=0.8, color='tab:blue')

        plt.xlabel("PO (deg)")
        # plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        # plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')
        plt.figure()
    return tuning_angles_ex, tuning_angles_in


def tune_binary_synaptic(N_excit_synapses = N_excit_synapses,
                         N_inhib_synapses = N_inhib_synapses,
                         theta1 = np.pi/4, theta2 = 3*np.pi/4,
                         to_plot = False):

    tuning_angles_in = np.ones(N_inhib_synapses)*theta1
    tuning_angles_in[0:int(N_inhib_synapses/2)] = theta2
    # OS from <w>
    tuning_angles_ex = np.ones(N_excit_synapses)*theta1


    nr_others = 0
    for i in range(N_excit_synapses):
        if weight_profiles[i]<0.225:
            #tuning_angles_ex[i] = (-1)**i * theta2
            tuning_angles_ex[i] = theta2
            nr_others += 1
        #if i<N_inhib_synapses and w_inh[i]>0.15:
        #    #tuning_angles_in[i] = (-1)**i * theta2
        #    tuning_angles_in[i] =  theta2

    #plt.hist(weight_profiles)
    #plt.show()

    if to_plot == True:
        plt.hist(tuning_angles_ex * 180 / np.pi, label="$N_{E}$", alpha=0.8, color='tab:orange')
        plt.hist(tuning_angles_in * 180 / np.pi, label="$N_{I}$", alpha=0.8, color='tab:blue')
        plt.xlabel("PO (deg)")
        # plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        # plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')
        plt.figure()
    return tuning_angles_ex, tuning_angles_in

def tune_cont_synaptic( N_excit_synapses = N_excit_synapses,
                        N_inhib_synapses = N_inhib_synapses,
                        theta1 = np.pi/4, theta2 = 3*np.pi/4,
                        weight_profiles = weight_profiles,
                        to_plot = False):
    tuning_angles_in = np.linspace(0, np.pi, N_inhib_synapses)

    # OS from <w>
    tuning_angles_ex = np.zeros_like(weight_profiles)
    new_weight_profiles = np.flip(np.sort(weight_profiles))

    bins = 5
    delta_angle = np.linspace(0, np.pi/4, bins)
    w = np.linspace(new_weight_profiles[0], new_weight_profiles[-1], bins)

    bin = 1
    for i in range(N_excit_synapses):
        if bin<bins:
            if new_weight_profiles[i]>w[bin]:
                tuning_angles_ex[i] = np.pi/4 + np.random.normal(loc = 0.0, scale=np.pi/18 * (bin), size = 1)
            else:
                bin = bin + 1

    if to_plot == True:
        plt.hist(tuning_angles_ex * 180 / np.pi, label="$N_{E}$", alpha=0.8, color="tab:orange")
        plt.hist(tuning_angles_in * 180 / np.pi, label="$N_{I}$", alpha=0.8, color='tab:blue')
        #plt.scatter(np.linspace(0,180,bins) * 180 / np.pi, ns, color=cmap(t))
        plt.xlabel("PO (deg)")
        # plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        # plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')
        plt.figure()
    return tuning_angles_ex, tuning_angles_in


def cont_tuning(N_excit_synapses = N_excit_synapses,
                N_inhib_synapses = N_inhib_synapses,
                theta1 = np.pi/4, theta_2 = 3*np.pi/4,
                to_plot = False):
    bins = 18
    x = np.linspace(0, np.pi, bins)
    ns = gaussian(x, np.pi/4, np.pi/10)*50
    ns = ns.astype(int)
    #tuning_angles_ex = np.ones(N_excit_synapses)*np.pi/4
    tuning_angles_ex = np.zeros(N_excit_synapses)
    bin = 1
    for i in range(N_excit_synapses):
        if i<bin*100:
            tuning_angles_ex[i] = np.pi / 4 + np.random.normal(loc=0.0, scale=np.pi / 10*bin , size=1)
        else:
            bin = bin+1

    tuning_angles_in = np.linspace(0, np.pi, num=N_inhib_synapses)

    #print(np.sum(ns))
    #print(len(tuning_angles_ex))

    t = np.ones(len(x))
    t[:5] = np.flip(np.linspace(0.0,0.5,5))
    t[5:15] = np.linspace(0.22,1.0, 10)

    if to_plot == True:
        #print(ns)
        #plt.hist(tuning_angles_in * 180 / np.pi, bins=15, color="tab:blue", alpha=0.6,  label= "$N_{I}$" )
        plt.hist(tuning_angles_ex * 180 / np.pi, color = "tab:orange", alpha = 0.8,  label= "$N_{E}$")
        plt.hist(tuning_angles_in * 180 / np.pi,  color="tab:blue", alpha=0.8, label="$N_{I}$")
        #plt.scatter(x*180/np.pi, ns, color = cmap(t))

        plt.xlabel("PO (deg)")
        # plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        # plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')
        plt.figure()
    return tuning_angles_ex, tuning_angles_in
