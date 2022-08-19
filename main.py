import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import truncnorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.axis import Axis

import matplotlib.pyplot as plt

import seaborn as sns

#np.random.seed(0)

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

# Neuron Model parameters
V_th = -50.0 # mV
V_rest = -70.0 # mV
V_spike = 0 # mV
E_leak = - 70.0 # mV
E_synapse = 0.0 # mV

tau_membrane = 20.0 # ms
tau_synapse = 5.0 # ms: weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_synapses)
#tau_synapse = 2.0 # ms:  weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.75, size = N_synapses)
g_0 = 0.0

# inhibitory parameters
tau_i = 10 # ms
E_inh = -90 #  mV
g_0_i = 0.0

# Input parameters
N_synapses = 250
N_inhib_synapses = int(N_synapses/5)
N_excit_synapses = N_synapses - N_inhib_synapses

#mu = 0.5
#mu = 0.4


mu = 0.29
sigma = np.sqrt(mu)

lower, upper = 1.2, 1.5


normal_lower = np.log(lower)
normal_upper = np.log(upper)
normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
normal_mean = np.log(mu) - normal_std**2 / 2

X = truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc = mu, scale = sigma)

#weight_profiles = np.random.lognormal(mean = normal_mean, sigma = normal_std, size = N_excit_synapses)
#w_inh = np.random.lognormal(mean = normal_mean, sigma = normal_std, size = N_inhib_synapses)

weight_profiles = np.log(X.rvs(N_excit_synapses))
w_inh = np.log(X.rvs(N_inhib_synapses))

# uniform distrib for weights has worked
#mu = 0.25
#weight_profiles = np.ones(N_excit_synapses) * mu
#w_inh = np.ones(N_inhib_synapses) * mu

# gaussian distrib for weights has worked as well
#mu = 0.07
#sigma = np.sqrt(mu)
#weight_profiles = np.random.normal( loc = mu, scale = sigma, size = N_excit_synapses)
#w_inh = np.random.normal(loc = mu, scale = sigma, size = N_inhib_synapses)

# The above specific mean and sigma values are chosen s.t. a reasonable background
# firing rate of the output neuron is obtained. (more details search_bg_weights())
f_background = 5 # Hz
f_max = 10 # Hz
f_inhib = 5 # Hz


# Integration parameters
dt = 0.1 # ms
nr_seconds = 20
N_steps = int(nr_seconds * 1000 / dt) # integration time steps
t_start = 0 # ms
t_end = nr_seconds * 1000 # ms
#dt = (t_end - t_start)/N_steps # time step width
time_range = np.linspace(t_start, t_end, N_steps)
save_interval = 10 # for dt = 0.1 save every 100 ms
save_zoom_interval = 500 #
tau_ref = int(50/dt)


burn_seconds = int(nr_seconds/2)
burn_steps = int(burn_seconds * 1000 / dt) # steps or 20 seconds or 20000 ms
stimulus_seconds = 10
stimulus_time_steps = int(stimulus_seconds * 1000 / dt)
silence_seconds = nr_seconds - burn_seconds - stimulus_seconds
silence_time_steps = int(silence_seconds * 1000 / dt)

spikes_pre = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, burn_steps))
spikes_post = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, silence_time_steps))
spikes_inh_pre = np.random.poisson(lam = f_inhib * 10e-4 * dt, size =  (N_inhib_synapses, burn_steps))
spikes_inh_post = np.random.poisson(lam = f_inhib * 10e-4 * dt, size =  (N_inhib_synapses, silence_time_steps))


def get_spike_indexes(spike_train, f_array = False):
    '''
    Function that transforms the array with Poisson distributed spikes (0 or 1)
    into an array that only stores the indexes of spikes, thereby avoiding the
    unnecessary storage of data.

    :param spike_train: Array of shape (N_syn, N_steps) - containing Poisson
                        distributed 0s or 1s for each presynaptic input.
           f_array: 'False' by default - bool parameter to indicate the
                     output of the individual presynaptic firing rates.

    :return: Default:
                      all_indexes (N_syn, different lengths) - containing only the
                      indexes of spikes for each presynaptic input.
             Upon specification:
                      List of arrays [all_indexes, all_fs (N_syn)] - where
                      the additional array gives the firing frequency of each
                      presynaptic input.
    '''

    N = len(spike_train)
    all_indexes = []

    if (f_array == False):
        for i in range(N):
            indexes = np.nonzero(spike_train[i])[0]
            all_indexes.append(indexes)
        return all_indexes
    else:
        all_fs = np.zeros(N)
        nr_seconds = len(spike_train[0]) * dt / 1000
        for i in range(N):
            indexes = np.nonzero(spike_train[i])[0]
            all_indexes.append(indexes)
            all_fs[i] = len(indexes) / nr_seconds
        return all_indexes, all_fs

def get_spiked_neuron_indexes(spike_train, t = 398400):
    N_neurons = len(spike_train)

    spiked = []
    # Search over all indexes for values matching the timestep number
    # and save the spiking neurons at that timestep in a list.
    for n in range(N_neurons):
        if np.any(np.equal(spike_train[n], t)):
            spiked.append(n)

    if (len(spiked) != 0):
        return spiked
    else:
        return False

def optimize_conductance_comp(g, g_i):
    '''

    Didn't really work. It took too long to compute.
    Try later.

    :param g:
    :param g_i:
    :return:
    '''
    g = g - g * dt / tau_synapse
    g_i = g_i - g_i * dt / tau_i

    ind_ex = get_spiked_neuron_indexes(all_trains)
    ind_in = get_spiked_neuron_indexes(spikes_inh)

    if ind_ex != False:
        g = g + dt * np.sum(np.take(weight_profiles, ind_ex))
        print(g)

    if ind_in != False:
        g_i = g_i + dt * np.sum(np.take(w_inh, ind_in))
        print(g_i)

    return g, g_i

def get_input_tuning(theta_i = 0, A = 1, N_orientations = 100):
    theta = np.linspace(theta_i - np.pi/2, theta_i + np.pi/2, N_orientations)
    c = f_response / (f_background + 2 * A)
    f = c*(f_background + A + A * np.cos((theta-theta_i)*np.pi/2))
    plt.plot(theta, f)
    plt.title("$\Theta =$ {} rad".format(theta_i))
    plt.xlabel("$\\theta$ (rad)")
    plt.ylabel("$f$ (Hz)")
    plt.show()
    return f

def get_f_result(theta, theta_synapse = [0], f_background = f_background):
    #c = f_response / (f_background + 2 * A)
    #f = c * (f_background + A + A * np.cos((theta - theta_synapse) * np.pi*3 / 4))
    #f = 2*A + A * np.cos((theta - theta_synapse) * np.pi * 2 / 3)
    #print(max(theta_synapse))
    #print(min(theta_synapse))
    #print(max(theta - theta_synapse))
    #print(min(theta - theta_synapse))
    B = (f_background-f_max)/(np.cos(1/3 * np.pi * np.pi) - 1)
    A = f_max - B
    f = A + B * np.cos((theta - theta_synapse) * np.pi * 2/3)
    #print(max(f))
    #print(min(f))
    return f

def get_fs_past(theta = 1.3, theta_synapse = [0], f_background = f_background):
    s = 0.4
    delta = theta - np.asarray(theta_synapse)
    #return f_background/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    f = f_background + (f_max - f_background)/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    #plt.scatter(delta, f)
    #plt.show()
    return f

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


def try_multiple_neurons(n=2):
    bars = 11
    f = np.zeros((n, bars))
    f_std = np.zeros((n, bars))
    PO = np.zeros(n)
    viridis = cm.get_cmap('viridis', n)


    for j in range(n):
        print("Neuron nr: {}.".format(j))
        x_axis, f[j], f_std[j], PO[j], all_spikes = get_response_for_bar(trials=1, to_plot=False, mean = j % 2 * np.pi/2)
        print(f[j])
        numpy.savetxt("neuron{}".format(j), f[j])

    #plot_soma_response(x=x_axis, y=np.mean(f,axis=0), err=np.sqrt(np.sum(np.square(f_std),axis=0))/n, name="PO", PO = [np.average(PO),np.std(PO)])
    #print(str(np.average(PO))+" "+str(np.std(PO)))

def generate_weights(order = False):
    global weight_profiles, w_inh

    # weight_profiles = np.random.lognormal(mean=normal_mean, sigma=normal_std, size=N_excit_synapses)
    # w_inh = np.random.lognormal(mean=normal_mean, sigma=normal_std, size=N_inhib_synapses)

    # X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    # plt.hist(weight_profiles)
    # plt.show()

    weight_profiles = np.log(X.rvs(N_excit_synapses))
   # weight_profiles = 2*gaussian(np.linspace(-1.5, 1.5, num=N_excit_synapses), mu=0.0, sig=0.5)
    w_inh = np.log(X.rvs(N_inhib_synapses))

    if order == True:
        weight_profiles = np.flip(np.sort(weight_profiles))


    return weight_profiles, w_inh


def get_response_for_bar(trials = 1, to_plot = True, color_neuron = 'gray', test_EI = False, std_ex = np.pi / 9, std_in = np.pi / 5, homogeneous = False, mean = np.pi/2):
    bars = 21
    fs_out = np.zeros((trials, bars))
    #mean = np.pi / 4
    #tuned_in_synapses = tune_all_synapses(N=N_inhib_synapses)
    #bars_range = np.linspace(mean*180/np.pi-90, mean*180/np.pi+90, bars)
    bars_range = np.linspace(0, 180, bars)

    nr_active_ex = np.zeros((trials, bars))
    w_active_ex = np.zeros((trials, bars))
    w_avg_ex = np.zeros((trials, bars))

    nr_active_in = np.zeros((trials, bars))
    w_active_in = np.zeros((trials, bars))
    w_avg_in = np.zeros((trials, bars))

    weight_profiles, w_inh = generate_weights(order=False)


    #if True:
    #    if test_EI == False:
    #        tuned_ex_synapses, tuned_in_synapses = tune_all_synapses(mean_ex=mean, mean_in=mean)
    #    else:
    #        tuned_ex_synapses, tuned_in_synapses = tune_all_synapses(mean_ex=mean, mean_in=mean, std_ex=std_ex, std_in=std_in)

    tuned_ex_synapses, tuned_in_synapses = tune_binary()

    #mu = 0.12
    #sigma = np.sqrt(mu)

    #normal_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
    #normal_mean = np.log(mu) - normal_std ** 2 / 2



    all_spikes = []
    #plt.hist(weight_profiles)
    #plt.show()
    #print(nr_spikes)


    for j in range(trials):
        print("trial {}".format(j))

        #print(np.shape(weight_profiles))
        #weight_profiles= np.sort(weight_profiles)
        #w1 = weight_profiles[:int(N_excit_synapses/2)]
        #w2 = np.flip(weight_profiles[int(N_excit_synapses/2):])
        #weight_profiles = np.block([w1,w2])
        #print(np.shape(weight_profiles))



        v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = evolve_potential_with_inhibition(
            spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
            to_plot=False, only_f=False, parameter_pass=True)
        print("after int = {}".format(v_spont))
        for i in range(bars):

            #theta = - np.pi/2 + i * np.pi / (bars)
            theta = i * np.pi / (bars-1)
            #f_response = get_f_result(theta, theta_synapse=0)
            #spikes_ex = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuned_ex_synapses))
            #spikes_in = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib), f_background=f_inhib)

            fs_ex = get_fs(theta=theta, theta_synapse=tuned_ex_synapses)
            fs_in = get_fs(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib)

            if homogeneous == False:
                spikes_ex, fs_res_ex = generate_spikes_array(fs_ex)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)
            else:
                fs_ex = get_fs(theta=theta, theta_synapse=mean * np.ones(N_excit_synapses), f_background=f_background)
                fs_in = get_fs(theta=theta, theta_synapse=mean * np.ones(N_inhib_synapses), f_background=f_inhib)
                spikes_ex, fs_res_ex = generate_spikes_array(f_response=fs_ex, f_background=f_background)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)

            nr_active_ex[j][i], w_active_ex[j][i], w_avg_ex[j][i] = count_active_synapses(fs_res_ex, f_active=f_max-1, w=weight_profiles)
            nr_active_in[j][i], w_active_in[j][i], w_avg_in[j][i] = count_active_synapses(fs_res_in, f_active=f_max-1, w=w_inh)

            if (j == 0) and (i == 0 or i == 5 or i == 10 or i == 15) and (to_plot == True):
                fs_out[j, i], spike_times = evolve_potential_with_inhibition(
                spikes_ex, spikes_in,
                v = v_spont, g_e = g_e_spont, g_i = g_i_spont, tau = tau_ref, nr_spikes=0, v_series=[], I_ex = [], I_in=[],
                name_i="i for bar: {}".format(i), name_V= "V for bar: {}".format(i), only_f=False, to_plot=True, parameter_pass=False)

            else:
                fs_out[j, i], spike_times = evolve_potential_with_inhibition(
                    spikes_ex, spikes_in,
                    v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[], I_in=[],
                    name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                    parameter_pass=False)
            all_spikes.append(spike_times)


            print("f_out = {}".format(fs_out[j,i]))
        plt.scatter(bars_range, fs_out[j,:], alpha=0.2, color = color_neuron)
        #plt.savefig("output trial {}".format(j))
    #plt.savefig("output_f_theta_all")

    print(spike_times)
    avg = np.mean(fs_out, axis=0)
    std = np.std(fs_out, axis=0)
    PO_index = np.argmax(avg)
    PO = bars_range[PO_index]

    if to_plot == True:
        avg_nr_ex = np.mean(nr_active_ex, axis=0)
        std_nr_ex = np.std(nr_active_ex, axis=0)
        avg_nr_in = np.mean(nr_active_in, axis=0)
        std_nr_in = np.std(nr_active_in, axis=0)

        avg_w_ex = np.mean(w_active_ex, axis=0)
        std_w_ex = np.std(w_active_ex, axis=0)
        avg_w_in = np.mean(w_active_in, axis=0)
        std_w_in = np.std(w_active_in, axis=0)

        avg_w_avg_ex = np.mean(w_avg_ex, axis=0)
        std_w_avg_ex = np.std(w_avg_ex, axis=0)
        avg_w_avg_in = np.mean(w_avg_in, axis=0)
        std_w_avg_in = np.std(w_avg_in, axis=0)

        plot_soma_response(bars_range, avg, std, name="PO")

        plot_PO_vs_weight(np.abs(tuned_ex_synapses - np.pi/4) * 180 / np.pi, weight_profiles, name='exc', binary=True)
        plot_PO_vs_weight(np.abs(tuned_in_synapses - np.pi/4) * 180 / np.pi, w_inh, name='inh', binary=True)

        plot_soma_response(bars_range, avg, std, name="delta_PO")
        #plot_soma_response(PO - bars_range, avg, std, name="delta_PO")

        plot_fig_3a(bars_range, avg, avg_w_ex, avg_w_in, std, std_w_ex, std_w_in)
        #plot_fig_3a(PO-bars_range, avg, avg_w_ex, avg_w_in, std, std_w_ex, std_w_in)
        plot_fig_3b(bars_range,
                    avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                    avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)
        #plot_fig_3b(PO-bars_range,
        #            avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
        #            avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)


    return bars_range, avg, std, PO, all_spikes

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

def count_active_synapses(f_array = np.linspace(0,20,20), f_active = f_max, w = weight_profiles):
    n = np.argwhere(f_array > f_active).ravel()
    return len(n), np.sum(w[n]), np.average(w[n])


def gaussian(x, mu, sig):
    #return 40 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def tune_binary(theta1 = np.pi/4, theta2 = 3*np.pi/4):
    tuning_angles_in = np.ones(N_inhib_synapses)*theta1
    tuning_angles_in[0:int(N_inhib_synapses / 2)] = theta2
    # OS from <w>
    tuning_angles_ex = np.ones(N_excit_synapses)*theta1
    tuning_angles_ex[0:int(N_excit_synapses/4)] = theta2

    p_inh = 0.5
    p_exc = 0.2

    #for i in range(N_excit_synapses):
    #    x = np.random.uniform(0,1)
    #    if x < p_exc:
    #        tuning_angles_ex[i] = theta2

        #if x > p_inh and i < N_inhib_synapses :
        #    tuning_angles_in[i] = theta2

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

    #stop = np.max(weight_profiles)
    #print(stop)
    #start = np.min(weight_profiles)
    #print(start)
    #range = np.max(weight_profiles) - np.min(weight_profiles)


    #print(range)

    new_weight_profiles = np.flip(np.sort(weight_profiles))

    bins = 5
    delta_angle = np.linspace(0, np.pi/4, bins)
    w = np.linspace(new_weight_profiles[0], new_weight_profiles[-1], bins)

    print(w)

    bin = 1
    for i in range(N_excit_synapses):
        if bin<bins:
            if new_weight_profiles[i]>w[bin]:
                tuning_angles_ex[i] = np.pi/4 + np.random.normal(loc = 0.0, scale=np.pi/18 * (bin), size = 1)
            else:
                bin = bin + 1

    #t = np.ones(bins)
    #t[:5] = np.flip(np.linspace(0.0,0.5,5))
    #t[5:15] = np.linspace(0.22,1.0, 10)



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





def tune_all_synapses(mean_ex = np.pi/4, mean_in = 0.0, std_ex = np.pi/10 , std_in = np.pi / 5,  to_plot = True):

    # OS from W
    tuning_angles_ex = np.abs(np.random.normal(loc=mean_ex, scale=std_ex, size=N_excit_synapses))
    #tuning_angles_in = np.random.normal(loc=mean_in, scale=std_in, size=N_inhib_synapses)
    #tuning_angles_in = np.random.uniform(mean_in - np.pi/2, mean_in + np.pi/2, size=N_inhib_synapses)
    for i in range(len(tuning_angles_ex)):
        if tuning_angles_ex[i] > np.pi or tuning_angles_ex[i]<0:
            if i%3 == 0:
                tuning_angles_ex[i] = np.random.uniform(mean_ex-std_ex/2, mean_ex+std_ex/2)
            else:
                tuning_angles_ex[i] = mean_ex

    tuning_angles_in = np.linspace(0, np.pi, num=N_inhib_synapses)
    # OS from <w>
    #tuning_angles_ex = np.linspace(mean_ex - np.pi / 2, mean_ex + np.pi / 2, num=N_excit_synapses)
    #range = np.linspace(0, np.pi, num=N_excit_synapses)
    #tuning_angles_ex = gaussian(range, mean_ex, std_ex)


    if (to_plot == True):
        #diff = np.histogram(tuning_angles_ex)[0]- np.histogram(tuning_angles_in)[0]
        #print(diff)

        plt.hist(tuning_angles_ex * 180/np.pi, label= "$N_{E}$", alpha = 0.8, color = 'tab:orange')
        #plt.hist((tuning_angles_ex - tuning_angles_in) * 180 / np.pi, label="$N_{E}-N_{I}$", alpha=0.8, color='tab:green')

        plt.hist(tuning_angles_in * 180/np.pi, label="$N_{I}$", alpha = 0.8, color = 'tab:blue')

        #bins = 15
        #t = np.linspace(0.0,1.0, bins)

        #plt.hist(tuning_angles_ex * 180 / np.pi, label="$N_{E}$", alpha=0.8, color=cmap(t), bins =bins)

        low_bound = np.min([np.min(tuning_angles_ex) * 180/np.pi, np.min(tuning_angles_in)]) * 180/np.pi
        up_bound = np.max([np.max(tuning_angles_ex) * 180/np.pi, np.max(tuning_angles_in)]) * 180/np.pi
        #shift = (up_bound-low_bound)/(2*len(diff))
        #plt.plot(np.linspace(low_bound-shift, up_bound-shift, len(diff)),
        #            diff, marker = 'o', markersize = 30, color = 'grey', label = "$N_{E}-N_{I}$" )
        #plt.xlabel("Tuning orientation $\\theta_{j}$ (deg)")
        plt.xlabel("PO (deg)")
        #plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        #plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')


        plt.figure()
    return tuning_angles_ex, tuning_angles_in


def get_input_response_for_bar():
    bars = 10
    fs_out = np.zeros(bars)
    all_fs = np.zeros((bars, N_excit_synapses))
    tuning_angles = tune_all_synapses()

    w_a = np.zeros(bars)
    f_a = np.zeros(bars)

    for i in range(bars):
        theta = -np.pi / 2 + i * np.pi / (bars)
        all_fs[i,:] = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuning_angles))[1]
        fs_out[i] = np.mean(all_fs[i,:])
        nr = 0
        for j, v in enumerate(all_fs[i,:]):
            if v > f_background+1:
                nr += 1
                w_a[i] += weight_profiles[j]
                f_a[i] += v
            if j == (N_excit_synapses-1) and f_a[i] != 0:
                f_a[i] = f_a[i] / (nr)
        #print(fs_out)
    theta_range = np.linspace(-90, 90, bars)

    for i in range(N_excit_synapses):
        plt.scatter(theta_range, all_fs[:,i], alpha=0.3, color = 'lightsteelblue')

    plt.plot(theta_range, fs_out, alpha=1, marker = "o", linewidth = "2", markersize = "12", label = "$\langle f \\rangle$")
    plt.plot(theta_range, all_fs[:,0], alpha=0.7, marker="o", linewidth="2", markersize="12", label = "$f_{i}$")
    plt.plot(theta_range, all_fs[:, 10], alpha=0.7, marker="o", linewidth="2", markersize="12", label="$f_{j}$")
    #plt.plot(theta_range, all_fs[:, 100], alpha=0.7, marker="o", linewidth="2", markersize="12", label="$f_{l}$")

    plt.xlabel("$\\theta$ (deg)")
    plt.ylabel("$f$ (Hz)")
    plt.legend()
    #plt.title("Showing a bar of $\\theta$ orientation, 100 tuned synapses at 0 rad")
    plt.show()
    plt.plot(theta_range, w_a/np.sum(w_a), alpha=0.7, marker="o", linewidth="2", markersize="12", label="$\sum_{j} w$")
    plt.plot(theta_range, f_a/np.sum(f_a), alpha=0.7, marker="o", linewidth="2", markersize="12", label="$\langle f \\rangle_{j}$")
    plt.xlabel("$\\theta$ (deg)")
    plt.ylabel("Normalized quantity (percent)", labelpad = 2)
    plt.legend()
    plt.savefig("cumulative_w.svg")
    plt.savefig("cumulative_w.png")



    #with open("presynaptic_f(theta)_tuned_at_0.txt", "w+") as f:
    #    f.write(str(f_response))



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


        #f_spikes_stimulus = np.sum(spikes_stimulus, axis=1) / (stimulus_seconds)


        #for i in range(stimulus_time_steps):
        #    for j in range(100):
        #        spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[0] * 10e-4 * dt)
        #        f_spikes[j]
            #print("f_synaptic_response = {} Hz".format(np.sum(spikes[0,burn_steps : burn_steps + stimulus_time_steps]) / stimulus_seconds))
            #for j in range(100, 150):
             #   spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[1] * 10e-4 * dt)
            #for j in range(150, N_synapses):
               # spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[2] * 10e-4 * dt)
    #return spikes, np.sum(spikes[0, burn_steps:burn_steps + stimulus_time_steps]) / (stimulus_seconds)
        #return spikes_stimulus, f_spikes_stimulus
        return spikes_stimulus, fs
    else:
        raise ValueError("Each synapse must have a corresponding frequency. i.e. size(f_response) = Nr exitatory synapses")


# To delete.
def generate_tuned_array(N = N_steps, firing_rate = 500, f_response = 0, N_syn = N_synapses):
    spikes = np.random.poisson(lam=firing_rate * 10e-4 * (t_end - t_start) / N, size=(1, N))
    return spikes, np.sum(spikes)/(nr_seconds)


def test_background_input_f(spikes_array):
    f = spikes_array[1]
    print(f)
    assert f < f_background + 1, "It should be around {} Hz.".format(f_background)

def test_background_input_cv(spikes_array):
    input_spikes = spikes_array[0]
    sigma = np.std(input_spikes)
    cv = sigma/np.mean(input_spikes)
    assert  cv < 1/sigma+1, "Not a Poisson distribution."

def test_visualise_background_input(spikes_array, spikes_to_see=10, name="excitatory", f = 5):
    all_trains = spikes_array
    N_synapses = len(spikes_array)

    #interval = int(N_synapses/spikes_to_see)-1
    interval = 1
    spikes_to_see = N_synapses
    nr_s = 1
    time_range = np.linspace(t_start, 1000, len(all_trains[0,:nr_s*10000]))

    #print(np.shape(all_trains))
    for i in range(2, N_synapses, interval):
        plt.scatter(time_range, (i+1)/int(N_synapses/spikes_to_see) * all_trains[i, :nr_s * 10000], label="synapse nr: {}".format(i), s = 10, marker= "D", color = "white")



    margin = 2/spikes_to_see
    plt.ylim([1 - margin, spikes_to_see + 2])
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell number")
    #plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.svg".format(spikes_to_see,name,N_synapses,f))
    plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.png".format(spikes_to_see, name, N_synapses, f))
    plt.figure()
    #plt.show()

def test_input_spikes(spikes_array, name = "excitatory", f = 5):
    #test_background_input_f(spikes_array)
    #test_background_input_cv(spikes_array)
    test_visualise_background_input(spikes_array, name = name, f = f)
    print("Passed all tests for Poisson input spikes!")

def get_f_isi_and_cv(spikes_arrays, to_plot = True, nr_seconds = stimulus_seconds, name = "", color=cmap(0.0)):
    """

    :param spikes_arrays: spikes_array = [Excitatory spike train array, Inhibitory spike trains]
                          - list of arrays of arbitrary dimensions
    :return: Plots firing rates, ISI and CV of ISI for both E/I populations.
    """

    if to_plot == True: both_isi = []
    both_fs = []
    spikes_array = spikes_arrays

    #for spikes_array in spikes_arrays:
    if True:
        N = len(spikes_array)
        if to_plot == True:
            all_isi = []
            all_indexes = []
        all_fs = np.zeros(N)
        for i in range(N):
            indexes = np.nonzero(spikes_array[i])
            all_fs[i] = len(indexes[0])/nr_seconds
            if to_plot == True:
                all_isi.append(np.diff(indexes, axis=1) * dt)
                if i%5 == 0:
                    plt.eventplot(np.asarray(indexes)*0.0001, color=color, lineoffsets=i, linelengths = 3)


        # The None term is added because the arrays in the list
        # have different sizes.
        if to_plot == True:
            all_isi = np.concatenate(all_isi, axis=None).ravel()
            both_isi.append(all_isi)
            all_indexes.append(indexes)
        both_fs.append(all_fs)

    if to_plot == True:
        plt.xlabel("Time (s)")
        #plt.ylabel("Cell number")

        ax = plt.gca()
        #ax.set_figwidth(4)
        #ax.set_figheight(1)
        ax.spines.left.set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.svg".format(spikes_to_see, name, N_synapses, f))
        plt.savefig("Raster plot = {}.svg".format(name))
        plt.savefig("Raster plot = {}.png".format(name))
        plt.figure()


        #labels = ['Excitatory', 'Inhibitory']
        #colors = ['tab:orange', 'tab:blue']
        labels = ['Input']
        colors = ['green']

        #for i, fs in enumerate(both_fs):
        fs = both_fs[0]
        if True:
            plt.hist(fs, weights=np.ones_like(fs) / len(fs), label=labels[0], color=color, alpha = 0.9)
            plt.xlabel("f (Hz)")
            plt.ylabel("Percent")

            #plt.legend()
            plt.savefig("Averate firing rates trial = {}.svg".format(name))
            plt.savefig("Averate firing rates trial = {}.png".format(name))
        plt.figure()

        #for i, isi in enumerate(both_isi):
        isi = all_isi
        if True:
            plt.hist(isi, weights=np.ones_like(isi) / len(isi), label=labels[0], color=color, alpha = 0.8)
            plt.xlabel("ISI (ms)")
            plt.ylabel("Percent")
            #plt.legend()
            plt.savefig("Interspike intervals trial = {}.svg".format(name))
            plt.savefig("Interspike intervals trial = {}.png".format(name))
        plt.figure()

        #for i, isi in enumerate(both_isi):
        if True:
            mean = np.mean(isi)
            cv = np.sqrt((isi - mean) ** 2) / mean
            plt.hist(cv, weights=np.ones_like(cv) / len(cv), label=labels[0], color=color, alpha = 0.8)
            plt.xlabel("CV of ISI")
            plt.ylabel("Percent")
            #plt.legend()
            plt.savefig("CV of ISI trial = {}.svg".format(name))
            plt.savefig("CV of ISI trial = {}.png".format(name))
        plt.figure()


        #plt.xlabel("Time (ms)")
        # plt.legend()
        # Draw a spike raster plot


        #interval = 1
        #spikes_to_see = 5
        #nr_s = 1
        #time_range = np.linspace(t_start, 1000, len(spikes_array[0, :nr_s * 10000]))

        #for i in range(2, 50, interval):
        #    plt.scatter(time_range, i/10*spikes_array[i, :nr_s * 10000], label="synapse nr: {}".format(i), s=20, marker="|", color=color)

        #margin = 2 / spikes_to_see
        #plt.ylim([1 - margin, spikes_to_see + 2])



def evolve_potential_with_inhibition(spikes_ex, spikes_in,
                                     v = V_rest, g_e = g_0, g_i = g_0_i,
                                     tau = tau_ref, nr_spikes = 0,
                                     v_series = [], v_zoom_series = [],  I_ex = [], I_in = [],
                                     w_ex = weight_profiles, w_in = w_inh,
                                     name_V = "V(t)", name_i = "i(t)",
                                     to_plot = True, only_f = True, parameter_pass = False):

    #print("V_in_funct = {}".format(v))

    initial = len(v_series)
    t_ex_trains = np.transpose(spikes_ex)
    t_in_trains = np.transpose(spikes_in)

    time_steps = len(spikes_ex[0])
    spike_times = []

    v_zoom_series = []
    I_ex = []
    I_in = []
    t_f_zoom = 1.5 #s


    for i in range(time_steps):
        tau += 1
        g_e = g_e - g_e * dt / tau_synapse + dt * (np.sum(np.multiply(t_ex_trains[i], w_ex)))
        g_i = g_i - g_i * dt / tau_i + dt * (np.sum(np.multiply(t_in_trains[i], w_in)))

        E_eff = (E_leak + g_e * E_synapse + g_i * E_inh)/(1 + g_e + g_i)
        tau_eff = tau_membrane/(1 + g_e + g_i)
        v = E_eff - (v - E_eff) * np.exp(-dt/tau_eff)

        if (v >= V_th) and (tau > tau_ref):

            nr_spikes = nr_spikes + 1
            spike_times.append(i)
            v = V_spike
            v_series.append(v)
            #if  (i % save_zoom_interval == 0) and (i * dt / 1000  < t_f_zoom):
            #if (i % save_zoom_interval == 0) and (i * dt / 1000 < t_f_zoom):
            if nr_spikes<3:
                v_zoom_series.append(v)
                I_ex.append(g_e * (E_synapse - V_rest))
                I_in.append(g_i * (E_inh - V_rest) + (E_leak - V_rest))
                t_f_zoom = i

            v = V_rest
            #if (i % save_zoom_interval == 0) and (i* dt / 1000 < t_f_zoom):
            #if (i % save_zoom_interval == 0) and (i * dt / 1000 < t_f_zoom):
            #if nr_spikes < 3:

            tau = 0
        else:
            #if (i % save_zoom_interval == 0) and  (i * dt / 1000  < 500):
            if nr_spikes < 3:
                I_ex.append(g_e * (E_synapse - v) )
                I_in.append(g_i * (E_inh - v)+ (E_leak-v))
                v_zoom_series.append(v)
                t_f_zoom = i
            if (i % save_interval == 0):
                v_series.append(v)


    #print("V_series after = {}".format(len(v_series)))
    if (len(v_series) == initial):
         t_f = time_steps * dt / 1000
         print("only stimulus")
    else:
        t_f = (len(v_series) + time_steps)* 0.1 / 1000

    print("Neuron output f = {} Hz".format(nr_spikes/t_f))

    if (only_f == True):
        return nr_spikes / (time_steps * 0.1 / 1000), spike_times
    elif (to_plot == True):
        # Plotting the trace.
        t = np.linspace(0, t_f, len(v_series))

        plt.plot(t, v_series, color = "gray")
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane potential (mV)")
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.savefig(name_V+".svg")
        plt.savefig(name_V + ".png")
        plt.figure()

        # Plotting the currents.

        t_f_zoom = t_f_zoom * dt / 1000
        t = np.linspace(0, t_f_zoom, len(I_in))
        plt.plot(t, I_in, color = "tab:blue", label="Inhibitory", alpha=0.9, linewidth=1)
        plt.plot(t, I_ex, color = "tab:orange", label="Excitatory", alpha = 0.9, linewidth=1)
        plt.plot(t, np.asarray(I_in) + np.asarray(I_ex), color = "black", label = "Net", alpha = 1.0, linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane currents (nA)")
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)

        plt.legend(labelcolor='linecolor')
        plt.savefig(name_i+".svg")
        plt.savefig(name_i + ".png")
        plt.figure()

        t = np.linspace(0, t_f_zoom, len(v_zoom_series))
        plt.plot(t, v_zoom_series, color="gray", linewidth=3)
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane potential (mV)")
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.savefig(name_V + "zoom.svg")
        plt.savefig(name_V + "zoom.png")
        plt.figure()


        return nr_spikes/(time_steps * 0.1 / 1000), spike_times
    elif (parameter_pass == True):
        return v, g_e, g_i, tau, nr_spikes, v_series, I_ex, I_in


def search_bg_weights():
    fs  = []
    # This one below was tested to work
    w_const = np.ones(N_synapses) * 0.18
    #fs.append(evolve_potential(w_ex = w_const))

    variance = 2.0/(N_synapses+1)
    for i in range(1,10):
        np.random.seed(i)
        weight_profiles = np.random.lognormal(mean=-2.0, sigma=0.5, size=N_synapses)
        #mu = 0.17 # mean range = [0.16 ,0.18] [2 ,10 Hz]
        #sigma = mu * 0.05 # sigma range = [0.04, 0.1] * mu
        #normal_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        #normal_mean = np.log(mu) - normal_std ** 2 / 2
        #weight_profiles = np.random.lognormal(mean=normal_mean, sigma=normal_std, size=N_synapses)
        #weight_profiles = np.random.normal(loc= 0.05, scale= np.sqrt(variance), size = N_synapses)
        #print(weight_profiles)
        plt.hist(weight_profiles)
        plt.savefig("weights rand seed = {}".format(i))
        plt.show()

        fs.append(evolve_potential(weight_profiles, f_response=0, name_f="f rand seed = {}".format(i), name_V="V rand seed = {}".format(i)))

        #if (i%10 == 0):
        #    print(i)

    #print(fs)
    print("The average firing rate = {} Hz and the std = {} ".format(np.mean(fs[1:]), np.std(fs[1:])))
    #plt.hist(fs[1:])
    #plt.figure()


def plot_f_analytic():
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    plt.figure()
    plt.plot(theta*180/np.pi, get_f_result(theta=theta, theta_synapse=[0]))
    plt.xlabel("$\Delta_{i} = (\\theta - \\theta_{i})$ (deg)")
    plt.ylabel("Firing rate $f_{i}$ (Hz)")
    plt.savefig("analytic_f")
    plt.xlim([-90,90])
    plt.show()

def get_hist_weights():
    plt.hist(weight_profiles, label= "$w_{E}$", alpha = 0.8, color = 'tab:orange')
    plt.hist(w_inh, label="$w_{I}$", alpha=0.8, color='tab:blue')
    plt.xlabel("$w_{X}$")
    plt.legend()
    plt.savefig("weight_distrib")
    plt.show()

def show_tuning_curve():
    bars = 100
    theta_range = np.linspace((-1) * np.pi/2, np.pi/2, bars)

    f_response = get_fs(theta = 0, theta_synapse = theta_range)
    plt.figure()
    plt.plot(theta_range*180/np.pi, f_response)
    plt.xlabel("$\Delta_{i}$ (deg)")
    plt.ylabel("Firing rate $f_{i}$ (Hz)")
    plt.savefig("analytic_f")
    plt.xlim([-90, 90])
    plt.show()

def get_output_bkg_stats(name = "0"):
    fs = []
    isi = []
    for i in range(50):
        print(i)
        spikes_pre = np.random.poisson(lam=f_background * 10e-4 * dt, size=(N_excit_synapses, burn_steps))
        spikes_inh_pre = np.random.poisson(lam=f_inhib * 10e-4 * dt, size=(N_inhib_synapses, burn_steps))
        f,spike_times = evolve_potential_with_inhibition(spikes_pre, spikes_inh_pre, to_plot=True)
        spike_times = np.asarray(spike_times)
        fs.append(f)
        isi.append((spike_times[1:]-spike_times[:-1]) * dt)

        # The None term is added because the arrays in the list
        # have different sizes
        all_isi = np.concatenate(isi, axis=None).ravel()


    plt.hist(fs, weights=np.ones_like(fs) / len(fs),  alpha = 0.8, color="gray")
    plt.xlabel("f (Hz)")
    plt.ylabel("Percent")
    #plt.legend()
    plt.savefig("Output average firing rates trial = {}.svg".format(name))
    plt.figure()

    plt.hist(all_isi, weights=np.ones_like(all_isi) / len(all_isi), alpha=0.8, color="gray")
    plt.xlabel("ISI (ms)")
    plt.ylabel("Percent")
    # plt.legend()
    plt.savefig("Output ISI trial = {}.svg".format(name))
    plt.figure()

    mean = np.mean(all_isi)
    cv = np.sqrt((all_isi - mean) ** 2) / mean
    plt.hist(cv, weights=np.ones_like(cv) / len(cv), color="gray", alpha=0.8)
    plt.xlabel("CV of ISI")
    plt.ylabel("Percent")
    # plt.legend()
    plt.savefig("Output CV of ISI trial = {}.svg".format(name))


def get_input_stats(PO = np.pi/4, trials = 50, show_trials = True, color = cmap(1.0)):
    bars = 21
    bars_range = np.linspace(0, 180, bars)
    fs = np.zeros_like(bars_range)
    stds = np.zeros_like(bars_range)
    f_bar = np.zeros((bars, trials))

    for i in range(bars):
        theta = i * np.pi / (bars-1)
        f_per_neuron = get_fs(theta=theta, theta_synapse= PO*np.ones(trials), f_background=f_background)
        spikes_arrays, f_bar[i] = generate_spikes_array(f_response=f_per_neuron, f_background=f_background)

        if i == 5 or i == 10 or i == 15:
            get_f_isi_and_cv(spikes_arrays, to_plot=True, name = str(i), color = color)
        #else:
         #   f_bar[i] = get_input_f_isi_and_cv(spikes_arrays, to_plot=False)[0]

        fs[i] = np.average(f_bar[i])
        stds[i] = np.std(f_bar[i])

    plt.figure()
    if show_trials == True:
        f_transpose = numpy.transpose(f_bar)
        #for j in range(trials):
         #plt.scatter(bars_range, f_transpose[j], marker='o', alpha=0.3, color = color)
    #plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
    #         color = color, markeredgewidth=1.5, markeredgecolor="black")
    plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color=color)
    #plt.errorbar(bars_range, fs, yerr=stds, fmt='none', color='black')
    plt.fill_between(bars_range, fs - stds, fs + stds, alpha=0.3, color=color)
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Presynaptic neuron $f$ (Hz)")
    #plt.legend()
    plt.savefig("Input response stats.svg")
    plt.savefig("Input response stats.png")
    plt.figure()

def get_output_stats_homogeneous(PO = 0, trials = 5, show_trials = True, color = "gray", to_plot = False):

    bars = 11
    bars_range = np.linspace(PO * 180 / np.pi - 90, PO * 180 / np.pi + 90, bars)
    fs = np.zeros_like(bars_range)
    stds = np.zeros_like(bars_range)
    f_bar = np.zeros((bars, trials))

    v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = evolve_potential_with_inhibition(
        spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
        to_plot=False, only_f=False, parameter_pass=True)
    print("after int = {}".format(v_spont))

    for i in range(bars):
        print("bar {}".format(i))
        theta = PO - np.pi / 2 + i * np.pi / (bars)
        fs_ex = get_fs(theta=theta, theta_synapse=PO * np.ones(N_excit_synapses), f_background=f_background)
        fs_in = get_fs(theta=theta, theta_synapse=PO * np.ones(N_inhib_synapses), f_background=f_background)

#from here
        all_spikes = []
        for j in range(trials):
            print("trial {}".format(j))

            if True:
                spikes_ex, fs_res_ex = generate_spikes_array(f_response=fs_ex, f_background=f_background)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_background)


                if (j == 0) and (i == 0 or i == 3 or i == 5) and (to_plot == True):
                    f_bar[i, j], spike_times = evolve_potential_with_inhibition(
                        spikes_ex, spikes_in,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=False, to_plot=True,
                        parameter_pass=False)

                else:
                    f_bar[i, j], spike_times = evolve_potential_with_inhibition(
                        spikes_ex, spikes_in,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                        parameter_pass=False)
                all_spikes.append(spike_times)
#until here
                print("f_out = {}".format(f_bar[i, j]))

        plt.figure()
        fs[i] = np.average(f_bar[i])
        stds[i] = np.std(f_bar[i])


    if show_trials == True:
        f_transpose = numpy.transpose(f_bar)
        for j in range(trials):
            plt.scatter(bars_range, f_transpose[j], marker='o', alpha=0.3, color=color)

        #if i == 0 or i == 3 or i == 5:
        #    get_f_isi_and_cv(spikes_arrays, to_plot=True, name=str(i), color=color)
        # else:
        #   f_bar[i] = get_input_f_isi_and_cv(spikes_arrays, to_plot=False)[0]


    plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color=color, markeredgewidth=1.5, markeredgecolor="black")
    plt.errorbar(bars_range, fs, yerr=stds, fmt='none', color='black')
    plt.xlabel("Stimulus $\\theta$")
    plt.ylabel("Response of the PS neuron $f$ (Hz)")
    plt.legend()
    plt.savefig("Output response stats")
    plt.figure()


def unpack_spike_times():
    list = [0, 1, 6]
    x = np.arange(10,20)
    print(x)
    print(x[list])
    x[list] = 1
    print(x)
    print(x[list])

def cont_tuning(theta1 = np.pi/4, theta_2 = 3*np.pi/4):
    bins = 18
    x = np.linspace(0, np.pi, bins)
    ns = gaussian(x, np.pi/4, np.pi/10)*50
    ns = ns.astype(int)
    #tuning_angles_ex = np.ones(N_excit_synapses)*np.pi/4
    tuning_angles_ex = np.zeros(N_excit_synapses)


    #last = 0
    #for i,val in enumerate(ns):
    #    tuning_angles_ex[last : last + val] = x[i]
    #    last = last + val

    bin = 1
    #for i in range(N_excit_synapses):
    for i in range(N_excit_synapses):
        if i<bin*100:
            tuning_angles_ex[i] = np.pi / 4 + np.random.normal(loc=0.0, scale=np.pi / 10*bin , size=1)
        else:
            bin = bin+1

    #if bin < bins:
    #    if new_weight_profiles[i] > w[bin]:
    #        tuning_angles_ex[i] = np.pi / 4 + np.random.normal(loc=0.0, scale=np.pi / 25 * (bin), size=1)
    #    else:
    #        bin = bin + 1





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


if __name__ == '__main__':
    #print(generate_spikes_array()[1])
    #evolve_potential()
    #evolve_potential(f_response=50)
    #search_bg_weights()
    #test_background_input_f()
    #test_background_input_cv()
    #get_input_tuning()
    #get_response_for_bar()
    #print(get_f_result(theta = 0, theta_synapse= 0))
    #evolve_potential_with_inhibition()
    #get_input_tuning(theta_i=0, A=1, N_orientations=100)
    #get_input_response_for_bar()
    #tune_all_synapses()
    #plot_f_analytic()
    #get_hist_weights()
    #test_input_spikes(spikes_pre, name = "excitatory", f = np.sum(spikes_pre)/(N_excit_synapses * burn_seconds))


    #get_input_f_isi_and_cv(spikes_arrays = [spikes_pre[:, :10000], spikes_inh_pre[:, :10000]])
    #get_isi(, population= "Inhibitory")
    #test_input_spikes(spikes_inh_pre, name = "inhibitory", f = np.sum(spikes_inh_pre)/(N_inhib_synapses * burn_seconds))





    #print(get_spike_indexes(spikes_pre))
    #get_spiked_neuron_indexes(get_spike_indexes(spikes_pre))
    #print(evolve_potential_with_inhibition(spikes_pre, spikes_inh_pre))
    #get_response_for_bar(trials=5)
    #get_fs(theta_synapse=tune_all_synapses(N_excit_synapses))
    #show_tuning_curve()
    #get_response_for_bar(trials=5)
    #tune_all_synapses(mean_ex=np.pi/4, mean_in=np.pi/4)
    #count_active_synapses(f_array=np.linspace(0, 20, 20), f_active=f_max)
    #show_tuning_curve()
    #try_multiple_neurons(n=5)
    #get_output_bkg_stats()

    get_response_for_bar(trials=5, to_plot=True, color_neuron='gray', test_EI=True, std_ex=np.pi/7, std_in=np.pi/4)
    #tune_binary_synaptic()

    #get_input_stats()

    #get_output_stats_homogeneous()
    #unpack_spike_times()

    #get_input_stats(show_trials = False)
    #get_output_stats_homogeneous()

    #try_multiple_neurons(n=20)

    #get_input_stats()

    #tune_all_synapses(mean_ex=np.pi/4, mean_in=0.0, std_ex=np.pi / 2, std_in=np.pi / 5, to_plot=True)
    #print(int(3.9))
    #cont_tuning()
    #tune_cont_synaptic()



