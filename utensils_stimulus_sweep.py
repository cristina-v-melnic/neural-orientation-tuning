import matplotlib.pyplot as plt

from parameters import *
from plotting_setup import *

# Set weights
def generate_weights(order = False):
    global weight_profiles, w_inh

    weight_profiles = np.log(X.rvs(N_excit_synapses))
    w_inh = np.log(X.rvs(N_inhib_synapses))

    if order == True:
        weight_profiles = np.flip(np.sort(weight_profiles))

    return weight_profiles, w_inh

def cut_neurons(PO_vector, weight_profiles, number = 50):
    PO_indexes = np.where((PO_vector - np.pi/4) < np.pi/20)

    for i in range(number):
        weight_profiles[PO_indexes[0][i]] = 0

    return weight_profiles


# Tune the afferents.
def tune_binary(theta1 = np.pi/4, theta2 = 3*np.pi/4):
    tuning_angles_in = np.ones(N_inhib_synapses)*theta1
    tuning_angles_in[0:int(N_inhib_synapses / 2)] = theta2
    # OS from <w>
    tuning_angles_ex = np.ones(N_excit_synapses)*theta1
    tuning_angles_ex[0:int(N_excit_synapses/4)] = theta2

    plt.figure()
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


def tune_binary_synaptic(theta1 = np.pi/4, theta2 = 3*np.pi/4):
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
    print(nr_others)

    #plt.hist(weight_profiles)
    #plt.show()

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

def tune_cont_synaptic(theta1 = np.pi/4, theta2 = 3*np.pi/4, weight_profiles = weight_profiles):
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


def cont_tuning(theta1 = np.pi/4, theta_2 = 3*np.pi/4):
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

    print(np.sum(ns))
    print(len(tuning_angles_ex))

    t = np.ones(len(x))
    t[:5] = np.flip(np.linspace(0.0,0.5,5))
    t[5:15] = np.linspace(0.22,1.0, 10)


    print(ns)
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

def gaussian(x, mu, sig):
        # return 40 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# Get analytic response value.
def get_fs(theta = 1.3, theta_synapse = [0], f_background = f_background, width = np.pi/6):
    #s = 0.4
    delta = np.asarray(theta_synapse) - theta
    #return f_background/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    #f = f_background + (f_max - f_background)/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))

    desmos_const = (-1) * 1.98902731655
    B = (f_background - f_max) / desmos_const
    A = f_max - B
    N_syn = len(delta)
    f = np.zeros(N_syn)

    for i in range(N_syn):
        if np.abs(delta[i]) <= width:
            f[i] = A + B * np.cos(2 * np.pi * delta[i])
        else:
            f[i] = f_background

    #plt.scatter(delta, f)
    #plt.show()
    return f

# Get response of afferents for a given stimulus (array f_response).
def generate_spikes_array(f_response = np.ones(N_excit_synapses) * f_background, f_background = f_background):
    """
    Generate a spike of length N with a frequency of firing_rate.

    :param N: length of spike evolution in time array.
    :param firing_rate: 5 Hz: the background firing rate.
    :return: [array of background rate firing of shape = (N_synapses, N)
              number of spikes in (t_final - t_0) time]
    """

    if len(f_response) > 0:
        for i in range(len(f_response)):
            if f_response[i] < f_background:
                f_response[i] = f_background

        spikes_stimulus = np.zeros((len(f_response), stimulus_time_steps))
        fs = np.zeros(len(f_response))

        for i in range(len(f_response)):
            train = np.random.poisson(lam=f_response[i] * 10e-4 * dt, size= (1, stimulus_time_steps))
            fs[i] = len(np.nonzero(train)[0]) / stimulus_seconds
            spikes_stimulus[i] = train

        return spikes_stimulus, fs
    else:
        raise ValueError("Each synapse must have a corresponding frequency. i.e. size(f_response) = Nr exitatory synapses")

def count_active_synapses(f_array = np.linspace(0,20,20), f_active = f_max, w = weight_profiles):
    n = np.argwhere(f_array > f_active).ravel()
    return len(n), np.sum(w[n]), np.average(w[n])

