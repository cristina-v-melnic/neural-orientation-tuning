from postsynaptic_train import *
from utensils_stimulus_sweep import *
from plots import *
import numpy as np


def get_response_for_bar(trials = 1, bars = 21, mean = np.pi/2,
                         syn = False, binary = True, to_plot = True,
                         color_neuron = 'gray', homogeneous = False,
                         cut_nr_neurons = 0):
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
        w_ex = cut_neurons(tuned_ex_synapses, w_ex, number=cut_nr_neurons)

    # List for storing postsynaptic spikes.
    all_spikes = []
    for j in range(trials):

        print("trial {}".format(j))

        # Get parameters in the spontaneous firing mode before showing the stimulus.
        v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = evolve_potential_with_inhibition(
                                                                                                                                        spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
                                                                                                                                        to_plot=False, only_f=False, parameter_pass=True, w_ex = w_ex)
        print("after int = {}".format(v_spont))
        for i in range(bars):
            theta = i * np.pi / (bars-1)
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
                spikes_ex, spikes_in, w_ex = w_ex,
                v = v_spont, g_e = g_e_spont, g_i = g_i_spont, tau = tau_ref, nr_spikes=0, v_series=[], I_ex = [], I_in=[],
                name_i="i for bar: {}".format(i), name_V= "V for bar: {}".format(i), only_f=False, to_plot=True, parameter_pass=False)

            else:
                fs_out[j, i], spike_times = evolve_potential_with_inhibition(
                    spikes_ex, spikes_in, w_ex = w_ex,
                    v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[], I_in=[],
                    name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                    parameter_pass=False)
            all_spikes.append(spike_times)


            #print("f_out = {}".format(fs_out[j,i]))
        #plt.scatter(bars_range, fs_out[j,:], alpha=0.2, color = color_neuron)
        #plt.savefig("output trial {}".format(j))
    #plt.savefig("output_f_theta_all")


    avg = np.mean(fs_out, axis=0)
    std = np.std(fs_out, axis=0)

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

        plot_fig_3a(bars_range, avg, avg_w_ex, avg_w_in, std, std_w_ex, std_w_in)
        plot_fig_3b(bars_range,
                    avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                    avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)

    return bars_range, avg, std, all_spikes