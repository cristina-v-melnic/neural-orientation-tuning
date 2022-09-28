from COBA_integration import *
from utils_stimulus_sweep import *
from plots import *
import numpy as np


def get_response_for_bar(trials = 1, bars = 21, mean = np.pi/2,
                         syn = False, binary = False, to_plot = True,
                         color_neuron = 'gray', homogeneous = False,
                         cut_nr_neurons = 0, name_file = "post"):
    '''
    Get the tuning curve of a postsynaptic neuron with differently tuned presynaptic
    afferents by presenting it with a range of bars of various orientation angles.

    :param trials: (int) Nr of times to sweep the bar to get the tuning curve with mean and std.
    :param bars: (int) Number of bars shown (orientation angles) in each sweep.
    :param mean: (float) PO of most input in units of pi.
    :param binary: (bool) Afferents with 2 types of PO or continuous PO.
    :param syn: (bool) The synaptic (weight-based) or structural (number-based) scenario.

    :param to_plot: (bool) Generate tuning curve plots or only get the data.
    :param color_neuron: (str) Color of the postsynaptic tuning curve.

    :param homogeneous: (bool) Tune all afferents to the same PO.

    :param cut_nr_neurons: (int) Nr of neurons to delete for robustness tests.

    :return: Tuning curve of the postsynaptic neuron as a list and/or a plot.
    '''

    # Vectors to store the firing rate for every bar at each trial.
    fs_out = np.zeros((trials, bars))
    bars_range = np.linspace(0, 180, bars)

    # Vectors to store parameters of active synapses, i.e., f > 5 Hz.
    nr_active_ex = np.zeros((trials, bars))
    w_active_ex = np.zeros((trials, bars))
    w_avg_ex = np.zeros((trials, bars))

    nr_active_in = np.zeros((trials, bars))
    w_active_in = np.zeros((trials, bars))
    w_avg_in = np.zeros((trials, bars))

    # Set the weights according to connectivity, i.e., synaptic vs structural.
    if syn == True:
        w_ex, w_inh = generate_weights(order=True)
        # Set POs according to distribution, i.e., binary(two types of POs) vs continuous.
        if binary == True:
            tuned_ex_synapses, tuned_in_synapses = tune_binary_synaptic()
        else:
            tuned_ex_synapses, tuned_in_synapses = tune_cont_synaptic()
    else:
        w_ex, w_inh = generate_weights(order=False)
        if binary == True:
            tuned_ex_synapses, tuned_in_synapses = tune_binary()
        else:
            tuned_ex_synapses, tuned_in_synapses = cont_tuning()

    # Robustness experiment upon deleting tuned afferents in both scenarios.
    if cut_nr_neurons != 0:
        w_ex = cut_neurons(tuned_ex_synapses, w_ex, number = cut_nr_neurons)

    # List for storing postsynaptic spikes.
    all_spikes = []
    for j in range(trials):

        print("trial {}".format(j))

        # Get parameters in the spontaneous firing mode before showing the stimulus.
        v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = integrate_COBA(
                                                                                                                                        spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
                                                                                                                                        to_plot=False, only_f=False, parameter_pass=True, w_ex = w_ex)
        print("after int = {}".format(v_spont))

        # Loop over trials.
        for i in range(bars):
            theta = i * np.pi / (bars-1)

            # Get firing rates of all neurons for trial i (analytic).
            fs_ex = get_fs(theta=theta, theta_synapse=tuned_ex_synapses)
            fs_in = get_fs(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib)

            # Generate Poisson spikes with the analytic firing rate for every neuron.
            if homogeneous == False:
                spikes_ex, fs_res_ex = generate_spikes_array(fs_ex)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)
            else:
                fs_ex = get_fs(theta=theta, theta_synapse=mean * np.ones(N_excit_synapses), f_background=f_background)
                fs_in = get_fs(theta=theta, theta_synapse=mean * np.ones(N_inhib_synapses), f_background=f_inhib)
                spikes_ex, fs_res_ex = generate_spikes_array(f_response=fs_ex, f_background=f_background)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)

            # Get the number and the weights of active synapses,
            # defined as ones with firing rate larger than f_max - 0.1 * f_max.
            nr_active_ex[j][i], w_active_ex[j][i], w_avg_ex[j][i] = count_active_synapses(fs_res_ex, f_active=f_max-0.1*f_max, w=weight_profiles)
            nr_active_in[j][i], w_active_in[j][i], w_avg_in[j][i] = count_active_synapses(fs_res_in, f_active=f_max-0.1*f_max, w=w_inh)

            # Get the postsynaptic neuron voltage trace (i.e. spikes)
            # by COBA LIF integration. Get a few detailed plots for the 5th, 10th and 15th trials.
            if (j == 0) and (i == 0 or i == 5 or i == 10 or i == 15) and (to_plot == True):
                fs_out[j, i], spike_times = integrate_COBA(
                spikes_ex, spikes_in, w_ex = w_ex,
                v = v_spont, g_e = g_e_spont, g_i = g_i_spont, tau = tau_ref, nr_spikes=0, v_series=[], I_ex = [], I_in=[],
                name_i="i for bar: {}".format(i), name_V= "V for bar: {}".format(i), only_f=False, to_plot=True, parameter_pass=False)

            else:
                fs_out[j, i], spike_times = integrate_COBA(
                    spikes_ex, spikes_in, w_ex = w_ex,
                    v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[], I_in=[],
                    name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                    parameter_pass=False)

            # Save the spike times.
            all_spikes.append(spike_times)


            #print("f_out = {}".format(fs_out[j,i]))
        #plt.scatter(bars_range, fs_out[j,:], alpha=0.2, color = color_neuron)
        #plt.savefig("output trial {}".format(j))
    #plt.savefig("output_f_theta_all")


    avg = np.mean(fs_out, axis=0)
    std = np.std(fs_out, axis=0)

    # If specified, plot the response curve of the postsynaptic neuron and
    # additional properties.
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

        plot_soma_response(bars_range, avg, std, name="PO", name_file=name_file)

        plot_PO_vs_weight(np.abs(tuned_ex_synapses - np.pi/4) * 180 / np.pi, weight_profiles, name='exc', binary=True)
        plot_PO_vs_weight(np.abs(tuned_in_synapses - np.pi/4) * 180 / np.pi, w_inh, name='inh', binary=True)

        plot_fig_3a(bars_range, avg, avg_w_ex, avg_w_in, std, std_w_ex, std_w_in)
        plot_fig_3b(bars_range,
                    avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                    avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)

    return bars_range, avg, std, all_spikes

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
        f_transpose = np.transpose(f_bar)
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
    #plt.savefig("Input response stats.svg")
    plt.savefig(stats_directory + "Input response stats.png")
    plt.figure()

def get_output_bkg_stats(name = 0):
    fs = []
    isi = []
    for i in range(50):
        print(i)
        spikes_pre = np.random.poisson(lam=f_background * 10e-4 * dt, size=(N_excit_synapses, burn_steps))
        spikes_inh_pre = np.random.poisson(lam=f_inhib * 10e-4 * dt, size=(N_inhib_synapses, burn_steps))
        if i == 0:
            f,spike_times = integrate_COBA(spikes_pre, spikes_inh_pre, to_plot=True, only_f=False)
        else:
            f, spike_times = integrate_COBA(spikes_pre, spikes_inh_pre, only_f=True)
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
    #plt.savefig("Output average firing rates trial = {}.svg".format(name))
    plt.savefig(stats_directory + "Output average firing rates trial = {}.png".format(name))
    plt.figure()

    plt.hist(all_isi, weights=np.ones_like(all_isi) / len(all_isi), alpha=0.8, color="gray", bins=20)
    plt.xlim([0,1000])
    plt.xlabel("ISI (ms)")
    plt.ylabel("Percent")
    # plt.legend()
    #plt.savefig("Output ISI trial = {}.svg".format(name))
    plt.savefig(stats_directory+"Output ISI trial = {}.png".format(name))
    plt.figure()

    mean = np.mean(all_isi)
    cv = np.sqrt((all_isi - mean) ** 2) / mean
    plt.hist(cv, weights=np.ones_like(cv) / len(cv), color="gray", alpha=0.8, bins=40)
    plt.xlim([0,4])
    plt.xlabel("CV of ISI")
    plt.ylabel("Percent")
    # plt.legend()
    #plt.savefig("Output CV of ISI trial = {}.svg".format(name))
    plt.savefig(stats_directory+"Output CV of ISI trial = {}.png".format(name))
