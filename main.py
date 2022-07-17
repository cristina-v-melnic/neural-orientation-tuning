import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


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

params = {'font.size': 21,
          'legend.handlelength': 2,
          'figure.figsize': [10.4, 8.8],
          'lines.markersize': 20.0,
          'lines.linewidth': 4.5,
          'axes.spines.top': False,
          'axes.spines.right': False
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
N_inhib_synapses = int(N_synapses/4)
N_excit_synapses = N_synapses - N_inhib_synapses

mu = 0.12
sigma = np.sqrt(mu)
#mu, sigma = 0.015, np.sqrt(0.015)
normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
normal_mean = np.log(mu) - normal_std**2 / 2

#weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_excit_synapses)
#w_inh = (-1) * np.ones(N_inhib_synapses)*(0.001)
#w_inh = (-1) * np.ones(N_inhib_synapses)*(0.015)
weight_profiles = np.random.lognormal(mean = normal_mean, sigma = normal_std, size = N_excit_synapses)
w_inh = np.random.lognormal(mean = normal_mean, sigma = normal_std, size = N_inhib_synapses)
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
tau_ref = int(5/dt)

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

def get_fs(theta = 1.3, theta_synapse = [0], f_background = f_background):
    s = 0.4
    delta = theta - np.asarray(theta_synapse)
    #return f_background/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    f = f_background + (f_max - f_background)/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    #plt.scatter(delta, f)
    #plt.show()
    return f

def get_response_for_bar(trials = 1):
    bars = 11
    fs_out = np.zeros((trials, bars))
    mean = np.pi/4
    tuned_ex_synapses, tuned_in_synapses = tune_all_synapses(mean_ex = mean, mean_in=mean)
    #tuned_in_synapses = tune_all_synapses(N=N_inhib_synapses)
    bars_range = np.linspace(mean*180/np.pi-90, mean*180/np.pi+90, bars)

    nr_active_ex = np.zeros((trials, bars))
    w_active_ex = np.zeros((trials, bars))
    w_avg_ex = np.zeros((trials, bars))

    nr_active_in = np.zeros((trials, bars))
    w_active_in = np.zeros((trials, bars))
    w_avg_in = np.zeros((trials, bars))

    #print(nr_spikes)

    for j in range(trials):
        print("trial {}".format(j))
        v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = evolve_potential_with_inhibition(
            spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
            to_plot=False, only_f=False, parameter_pass=True)
        print("after int = {}".format(v_spont))
        for i in range(bars):
            theta = mean - np.pi/2 + i * np.pi / (bars)
            #f_response = get_f_result(theta, theta_synapse=0)
            #spikes_ex = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuned_ex_synapses))
            #spikes_in = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib), f_background=f_inhib)

            fs_ex = get_fs(theta=theta, theta_synapse=tuned_ex_synapses)
            fs_in = get_fs(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib)

            spikes_ex, fs_res_ex = generate_spikes_array(fs_ex)
            spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)

            nr_active_ex[j][i], w_active_ex[j][i], w_avg_ex[j][i] = count_active_synapses(fs_res_ex, f_active=7.0, w=weight_profiles)
            nr_active_in[j][i], w_active_in[j][i], w_avg_in[j][i] = count_active_synapses(fs_res_in, f_active=7.0, w=w_inh)

            if (j == 0) and (i == 0 or i == 3 or i == 5):
                fs_out[j, i] = evolve_potential_with_inhibition(
                spikes_ex, spikes_in,
                v = v_spont, g_e = g_e_spont, g_i = g_i_spont, tau = tau_ref, nr_spikes=0, v_series=[], I_ex = [], I_in=[],
                name_i="i for bar: {}".format(i), name_V= "V for bar: {}".format(i), only_f=False, to_plot=True, parameter_pass=False)
            else:
                fs_out[j, i] = evolve_potential_with_inhibition(
                    spikes_ex, spikes_in,
                    v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[], I_in=[],
                    name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                    parameter_pass=False)

            print("f_out = {}".format(fs_out[j,i]))
        plt.scatter(bars_range, fs_out[j,:], alpha=0.4, color = 'gray')
        plt.savefig("output trial {}".format(j))
    plt.savefig("output_f_theta_all")

    avg = np.mean(fs_out, axis=0)
    std = np.std(fs_out, axis=0)

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


    PO_index = np.argmax(avg)
    PO = bars_range[PO_index]
    print(PO)

    plot_soma_response(bars_range, avg, std, name="PO")
    plot_soma_response(bars_range-PO, avg, std, name="delta_PO")

    plot_fig_3a(bars_range-PO, avg, avg_w_ex, avg_w_in, std, std_w_ex, std_w_in)
    plot_fig_3b(bars_range-PO,
                avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)

    #plot_soma_response(bars_range-PO, avg_nr_ex, std_nr_ex, name ="excitatory")
    #plot_soma_response(bars_range-PO, avg_nr_in, std_nr_in, name="inhibitory")
    #plot_soma_response(bars_range - PO, avg_w_ex, std_w_ex, name="w_excitatory")
    #plot_soma_response(bars_range - PO, avg_w_in, std_w_in, name="w_inhibitory")
    #plot_soma_response(bars_range - PO, avg_w_avg_ex, std_w_avg_ex, name="avg_w_excitatory")
    #plot_soma_response(bars_range - PO, avg_w_avg_in, std_w_avg_in, name="avg_w_inhibitory")

    plot_PO_vs_weight(tuned_ex_synapses * 180/np.pi - PO, weight_profiles, name = 'exc')
    plot_PO_vs_weight(tuned_in_synapses * 180/np.pi - PO, w_inh, name = 'inh')
    return avg, std, PO

def plot_soma_response(x, y, err, name):
    if name == 'PO':
        plt.scatter([x[np.argmax(y)]], [np.min(y)], alpha=1.0, marker='x' , s=50, color = 'tab:red', label ="PO")
        plt.text(x[np.argmax(y)]+2, np.min(y), s = "{} deg".format(x[np.argmax(y)]))
        plt.xlabel("Stimulus of orientation $\\theta$ (deg)")
        plt.ylabel("Post-synaptic neuron response firing rate $f$ (Hz)")
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.spines.right.set_visible(True)
        ax2.set_ylabel("Individual trials $f$ (Hz)")
        ax2.yaxis.label.set_color('gray')

    elif name == "delta_PO":
        plt.xlabel("Pre- and post-synaptic neuron PO difference (deg)")
        plt.ylabel("Post-synaptic neuron response firing rate $f$ (Hz)")
        #plt.plot(x, nr_active, label = "Number of active synapses")
    else:
        plt.xlabel("Pre- and post-synaptic neuron PO difference (deg)")
        plt.ylabel("Number of active {} synapses (Hz)".format(name))
    plt.plot(x, y, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color='gray', markeredgewidth=1.5, markeredgecolor = "black")
    plt.errorbar(x, y, yerr=err, fmt='none', color='black')

    plt.legend()

    # plt.title("Showing a bar of $\\theta$ orientation, 100 tuned synapses at 0 rad with weights = 0.01")
    plt.savefig("output_f_theta_{}".format(name))
    plt.figure()

def plot_fig_3a(x, y1, y2, y3, std1, std2, std3):
    plt.plot(x, y1, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color='gray', markeredgewidth=1.5, markeredgecolor="black")
    plt.errorbar(x, y1, yerr=std1, fmt='none', color='black', barsabove = True)
    plt.xlabel("Pre- and post-synaptic neuron PO difference (deg)")
    plt.ylabel("Post-synaptic neuron response firing rate $f$ (Hz)")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker='o' ,alpha=0.9, linewidth="2", markersize="20", label="Excitatory $W_{E}$",
             color='tab:orange', markeredgewidth=1.5, markeredgecolor="black")
    ax2.errorbar(x, y2, yerr=std2, fmt='none', color='tab:red', barsabove = True)
    ax2.plot(x, y3, marker='o', alpha=0.9, linewidth="2", markersize="20", label="Inhibitory $W_{I}$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="black")
    ax2.errorbar(x, y3, yerr=std3, fmt='none', color='blue', barsabove = True)
    ax2.set_ylabel("Cumulative weight of active synapses $W_{P}$")
    ax2.spines.right.set_visible(True)
    plt.legend()
    plt.savefig("fig3a")
    plt.figure()

def plot_fig_3b(x, y1, y2, std1, std2, y3, y4, std3, std4):
    plt.plot(x, np.log(y1), marker='o', alpha=0.9, linewidth="2", markersize="20", label="Average weight",
             color='tab:gray', markeredgewidth=1.5, markeredgecolor="red")
    plt.errorbar(x, np.log(y1), yerr=np.log(std1), fmt='none', color='gray', barsabove=True)
    plt.xlabel("Pre- and post-synaptic neuron PO difference (deg)")
    plt.ylabel("Average weight of active synapses (log)")
    ax = plt.gca()
    ax.yaxis.label.set_color('tab:gray')
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$N$",
            color='tab:orange', markeredgewidth=1.5, markeredgecolor="black")
    ax2.errorbar(x, y2, yerr=std2, fmt='none', color='tab:red', barsabove=True)
    ax2.set_ylabel("Number of active synapses")
    ax2.yaxis.label.set_color('tab:orange')
    ax2.spines.right.set_visible(True)
    plt.savefig("fig3b_ex")
    plt.figure()

    plt.plot(x, np.log(y3), marker='o', alpha=0.9, linewidth="2", markersize="20", label="Average weight",
             color='tab:gray', markeredgewidth=1.5, markeredgecolor="blue")
    plt.errorbar(x, np.log(y3), yerr=np.log(std3), fmt='none', color='gray', barsabove=True)
    plt.xlabel("Pre- and post-synaptic neuron PO difference (deg)")
    plt.ylabel("Average weight of active synapses (log)")
    ax = plt.gca()
    ax.yaxis.label.set_color('tab:gray')
    ax2 = ax.twinx()
    ax2.plot(x, y4, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$N$",
             color='tab:blue', markeredgewidth=1.5, markeredgecolor="black")
    ax2.errorbar(x, y4, yerr=std4, fmt='none', color='blue', barsabove=True)
    ax2.set_ylabel("Number of active synapses")
    ax2.yaxis.label.set_color('tab:blue')
    ax2.spines.right.set_visible(True)
    plt.savefig("fig3b_inh")
    plt.figure()

def plot_PO_vs_weight(x, y, name = ''):
    if name == 'exc':
        plt.scatter(np.abs(x), np.log(y), s = 80, marker='o', alpha=0.9, color = 'tab:orange', label = "Excitatory synapse")
    else:
        plt.scatter(np.abs(x), np.log(y), s = 80, marker='o', alpha=0.9, color = 'tab:blue',  label = "Inhibitory synapse")
    plt.xlabel("Pre- and post-synaptic neuron PO difference")
    plt.ylabel("Log(Synaptic weight)")
    plt.legend()
    plt.savefig("PO_vs_weight_{}".format(name))
    plt.figure()

def count_active_synapses(f_array = np.linspace(0,20,20), f_active = f_max, w = weight_profiles):
    n = np.argwhere(f_array > f_active).ravel()
    return len(n), np.sum(w[n]), np.average(w[n])


def tune_all_synapses(mean_ex = 0.0, mean_in = 0.0):
    #tuning_angles = np.random.normal(loc=0.0, scale = 0.55, size=N)
    #tuning_angles = np.random.normal(loc=0.0, scale=np.pi/5, size=N)

    import scipy.stats as stats

    #lower, upper = (-1) * np.pi/2, np.pi/2
    #mu_e, sigma_e = 0, np.pi/3
    #mu_i, sigma_i = 0, np.pi/3
    #tuning_angles_ex = stats.truncnorm(
    #    (lower - mu_e) / sigma_e, (upper - mu_e) / sigma_e, loc=mu_e, scale=sigma_e).rvs(N_excit_synapses)
    #N = stats.norm(loc=mu, scale=sigma)
    #tuning_angles_in = stats.truncnorm(
    #    (lower - mu_i) / sigma_i, (upper - mu_i) / sigma_i, loc=mu_i, scale=sigma_i).rvs(N_inhib_synapses)
   # print(tuning_angles_in.rvs(100))

    tuning_angles_ex = np.random.normal(loc=mean_ex, scale=np.pi / 7, size=N_excit_synapses)
    tuning_angles_in = np.random.normal(loc=mean_in, scale=np.pi / 5, size=N_inhib_synapses)
    #tuning_angles_in = np.ones(N_inhib_synapses) * 1.7

    #plt.ylabel("Log weights $w_{j}$ (a.u.)")
    #plt.xlabel("Tuning orientation $\\theta_{j}$ (deg)")
    #plt.scatter(tuning_angles * 180/np.pi, np.log(weight_profiles[:N_excit_synapses]), edgecolors = 'black')

    diff = np.histogram(tuning_angles_ex)[0]- np.histogram(tuning_angles_in)[0]
    print(diff)

    plt.hist(tuning_angles_ex * 180/np.pi, label= "$N_{E}$", alpha = 0.8, color = 'tab:orange')
    #plt.hist((tuning_angles_ex - tuning_angles_in) * 180 / np.pi, label="$N_{E}-N_{I}$", alpha=0.8, color='tab:green')
    plt.hist(tuning_angles_in * 180/np.pi, label="$N_{I}$", alpha = 0.8, color = 'tab:blue')

    low_bound = np.min([np.min(tuning_angles_ex), np.min(tuning_angles_in)]) * 180/np.pi
    up_bound = np.max([np.max(tuning_angles_ex),np.max(tuning_angles_in)]) * 180/np.pi
    shift = (up_bound-low_bound)/(2*len(diff))
    plt.plot(np.linspace(shift+low_bound, up_bound-shift, len(diff)),
                diff, marker = 'o', markersize = 12, color = 'grey', label = "$N_{E}-N_{I}$" )
    plt.xlabel("Tuning orientation $\\theta_{j}$ (deg)")
    #plt.xlim([-95,95])
    plt.legend()
    #plt.savefig("weights_per_tuned_synapse")
    plt.savefig('tuning_range')
    plt.show()
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
    plt.ylabel("Normalized quantity (percent)")
    plt.legend()
    plt.savefig("cumulative_w")
    plt.show()


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
        plt.scatter(time_range, (i+1)/int(N_synapses/spikes_to_see) * all_trains[i, :nr_s * 10000], label="synapse nr: {}".format(i), s = 10, marker= ".", color = "black")


    margin = 2/spikes_to_see
    plt.ylim([1 - margin, spikes_to_see + 2])
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell number")
    #plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.".format(spikes_to_see,name,N_synapses,f))
    plt.figure()
    #plt.show()

def test_input_spikes(spikes_array, name = "excitatory", f = 5):
    #test_background_input_f(spikes_array)
    #test_background_input_cv(spikes_array)
    test_visualise_background_input(spikes_array, name = name, f = f)
    print("Passed all tests for Poisson input spikes!")

def get_input_f_isi_and_cv(spikes_arrays):
    """

    :param spikes_arrays: spikes_array = [Excitatory spike train array, Inhibitory spike trains]
                          - list of arrays of arbitrary dimensions
    :return: Plots firing rates, ISI and CV of ISI for both E/I populations.
    """

    both_isi = []
    both_fs = []

    for spikes_array in spikes_arrays:
        N = len(spikes_array)
        all_isi = []
        all_fs = np.zeros(N)
        for i in range(N):
            indexes = np.nonzero(spikes_array[i])
            all_fs[i] = len(indexes[0])
            all_isi.append(np.diff(indexes, axis=1) * dt)

        # The None term is added because the arrays in the list
        # have different sizes.
        all_isi = np.concatenate(all_isi, axis=None).ravel()
        both_isi.append(all_isi)
        both_fs.append(all_fs)

    labels = ['Excitatory', 'Inhibitory']
    colors = ['tab:orange', 'tab:blue']

    for i, fs in enumerate(both_fs):
        plt.hist(fs, weights=np.ones_like(fs) / len(fs), label=labels[i], color=colors[i], alpha = 0.8)
        plt.xlabel("f (Hz)")
        plt.ylabel("Percent")
        plt.legend()
        plt.savefig("Averate firing rates")
    plt.figure()

    for i, isi in enumerate(both_isi):
        plt.hist(isi, weights=np.ones_like(isi) / len(isi), label=labels[i], color=colors[i], alpha = 0.8)
        plt.xlabel("Time (ms)")
        plt.ylabel("Percent")
        plt.legend()
        plt.savefig("Interspike intervals")
    plt.figure()

    for i, isi in enumerate(both_isi):
        mean = np.mean(isi)
        cv = np.sqrt((isi - mean) ** 2) / mean
        plt.hist(cv, weights=np.ones_like(cv) / len(cv), label=labels[i], color=colors[i], alpha = 0.8)
        plt.ylabel("Percent")
        plt.legend()
        plt.savefig("CV of ISI")
    plt.figure()


def evolve_potential_with_inhibition(spikes_ex, spikes_in,
                                     v = V_rest, g_e = g_0, g_i = g_0_i,
                                     tau = tau_ref, nr_spikes = 0,
                                     v_series = [],  I_ex = [], I_in = [],
                                     w_ex = weight_profiles, w_in = w_inh,
                                     name_V = "V(t)", name_i = "i(t)",
                                     to_plot = True, only_f = True, parameter_pass = False):

    #print("V_in_funct = {}".format(v))

    initial = len(v_series)
    t_ex_trains = np.transpose(spikes_ex)
    t_in_trains = np.transpose(spikes_in)

    time_steps = len(spikes_ex[0])

    for i in range(time_steps):
        tau += 1
        g_e = g_e - g_e * dt / tau_synapse + dt * (np.sum(np.multiply(t_ex_trains[i], w_ex)))
        g_i = g_i - g_i * dt / tau_i + dt * (np.sum(np.multiply(t_in_trains[i], w_in)))

        E_eff = (E_leak + g_e * E_synapse + g_i * E_inh)/(1 + g_e + g_i)
        tau_eff = tau_membrane/(1 + g_e + g_i)
        v = E_eff - (v - E_eff) * np.exp(-dt/tau_eff)

        if (v >= V_th) and (tau > tau_ref):
            nr_spikes = nr_spikes + 1
            v = V_spike

            if (i % save_interval == 0):
                v_series.append(v)
                I_ex.append(g_e * (E_synapse - v))
                I_in.append(g_i * (E_inh - v))

            v = V_rest
            tau = 0
        else:
            if (i % save_interval == 0):
                I_ex.append(g_e * (E_synapse - v))
                I_in.append(g_i * (E_inh - v))
                v_series.append(v)
    #print("V_series after = {}".format(len(v_series)))
    if (len(v_series) == initial):
         t_f = time_steps * 0.1 / 1000
         print("only stimulus")
    else:
        t_f = (len(v_series) + time_steps)* 0.1 / 1000

    print("Neuron output f = {} Hz".format(nr_spikes/t_f))

    if (only_f == True):
        return nr_spikes / (time_steps * 0.1 / 1000)
    elif (to_plot == True):
        # Plotting the trace.
        t = np.linspace(0, t_f, len(v_series))

        plt.plot(t, v_series)
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane potential (mV)")
        plt.savefig(name_V)
        plt.figure()

        # Plotting the currents.
        plt.plot(t, I_in, color = "tab:blue", label="Inhibitory current", alpha=0.9)
        plt.plot(t, I_ex, color = "tab:orange", label="Excitatory current", alpha = 0.9)
        plt.plot(t, np.asarray(I_in) + np.asarray(I_ex), color = "gray", label = "Net current", alpha = 1.0)
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane currents (nA)")
        plt.legend()
        plt.savefig(name_i)
        plt.figure()

        return nr_spikes/(time_steps * 0.1 / 1000)
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

def get_f(v_series):
    '''
    Doesn't work if you're not saving v at every time step -> REDO/FIX
    :param v_series:
    :return:
    '''
    step_box = 100 # time steps -> f(t in s ) -> 1 s
    time_box = step_box * dt
    len_boxes = int(len(v_series)/step_box)

    f_series = []
    spike_nr = 0
    for j in range(1,len_boxes+1):
        for i in range(step_box):
            if (v_series[ (j-1) * step_box + i] == 0):
                spike_nr += 1
        f_series.append(spike_nr*10000/(step_box*j))

    plt.plot(np.linspace(1,t_end/(1000),len_boxes),f_series, label=" $\langle f \ (T) \\rangle \ =$ {} Hz".format(spike_nr/1000))
    plt.ylabel("$\langle f \ (t) \\rangle$ (Hz)")
    plt.xlabel("t (s)")
    plt.legend()
    plt.show()


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

def get_output_bkg_stats():
    fs = []
    for i in range(50):
        print(i)
        spikes_pre = np.random.poisson(lam=f_background * 10e-4 * dt, size=(N_excit_synapses, burn_steps))
        spikes_inh_pre = np.random.poisson(lam=f_inhib * 10e-4 * dt, size=(N_inhib_synapses, burn_steps))
        f = evolve_potential_with_inhibition(spikes_pre, spikes_inh_pre)
        fs.append(f)


    # label=labels[i], color=colors[i],
    plt.hist(fs, weights=np.ones_like(fs) / len(fs),  alpha = 0.8)
    plt.xlabel("f (Hz)")
    plt.ylabel("Percent")
    #plt.legend()
    plt.savefig("Averate firing rates")
    plt.figure()

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
    get_response_for_bar(trials=5)
    #tune_all_synapses(mean_ex=np.pi/4, mean_in=np.pi/4)
    #count_active_synapses(f_array=np.linspace(0, 20, 20), f_active=f_max)


