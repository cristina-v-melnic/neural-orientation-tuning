from parameters import *
from ConnectedNeuron import *
from SpikyLIF import *
from plots import *

class TunedResponse:
    '''
        Class containing orientation-selective response of
        pre- and post-synaptic neurons upon stimuli in form
        of bars or different orientation angles.
    '''

    def __init__(self, NeuronConnectivity, NeuronModel, nr_bars, trials, to_plot = "True"):
        '''
        :param NeuronConnectivity: (ConnectedNeuron) Object
                                    describing the connectivity
                                    and functionality of a single
                                    neron with converging afferents.
        :param NeuronModel: (SpikyLIF) Object containing the dynamics
                            of the neural activity in a circuit for
                            a certain period of time.
        :param bars: (int) Number of bars shown (orientation
                        angles) in each sweep.
        ::param trials: (int) Nr of times to sweep the bar
                        to get the tuning curve with mean
                        and std.
        :param to_plot: (bool) Generate tuning curve plots
                        or only get the data.
        '''
        # Stimulus details.
        self.nr_bars = nr_bars
        self.bars_range = np.linspace(0, 180, self.nr_bars)
        self.trials = trials

        # Neural network details.
        self.NeuronConnectivity = NeuronConnectivity
        self.NeuronModel = NeuronModel

        # Response details.
        # Vectors to store the firing rate for every bar at each trial.
        self.fs_out = np.zeros((self.trials, self.nr_bars))
        self.f_avg = 0
        self.f_std = 0


        # Vectors to store parameters of active synapses,
        # i.e., f > f_max - f_max * 0.1 Hz.
        self.nr_active_ex = np.zeros((self.trials, self.nr_bars))
        self.w_active_ex = np.zeros((self.trials, self.nr_bars))
        self.w_avg_ex = np.zeros((self.trials, self.nr_bars))

        self.nr_active_in = np.zeros((self.trials, self.nr_bars))
        self.w_active_in = np.zeros((self.trials, self.nr_bars))
        self.w_avg_in = np.zeros((self.trials, self.nr_bars))

        # Visualisation.
        self.to_plot = to_plot


    def get_response_for_bar(self, mean=np.pi / 2,
                             syn=False, binary=False, to_plot=True,
                             color_neuron='gray', homogeneous=False,
                             cut_nr_neurons=0, name_file="post"):
        '''
        Get the tuning curve of a postsynaptic neuron with
        differently tuned presynaptic afferents by presenting
        it with a range of bars of various orientation angles.

        :param mean: (float) PO of most input in units of pi.
        :param binary: (bool) Afferents with 2 types of PO or continuous PO.
        :param syn: (bool) The synaptic (weight-based) or structural (number-based) scenario.
        :param color_neuron: (str) Color of the postsynaptic tuning curve.
        :param homogeneous: (bool) Tune all afferents to the same PO.
        :param cut_nr_neurons: (int) Nr of neurons to delete for robustness tests.

        :return: Tuning curve of the postsynaptic neuron as a list and/or a plot.
        '''

        # Set the weights according to connectivity, i.e., synaptic vs structural.
        #if syn == True:
        #    w_ex, w_inh = generate_weights(order=True)
        #    # Set POs according to distribution, i.e., binary(two types of POs) vs continuous.
        #    if binary == True:
        #        tuned_ex_synapses, tuned_in_synapses = tune_binary_synaptic()
        #    else:
        #        tuned_ex_synapses, tuned_in_synapses = tune_cont_synaptic()
        #else:
        #   w_ex, w_inh = generate_weights(order=False)
        #    if binary == True:
        #        tuned_ex_synapses, tuned_in_synapses = tune_binary()
        #    else:
        #        tuned_ex_synapses, tuned_in_synapses = cont_tuning()

        # Robustness experiment upon deleting tuned afferents in both scenarios.
        #if cut_nr_neurons != 0:
        #    w_ex = cut_neurons(tuned_ex_synapses, w_ex, number=cut_nr_neurons)

        # List for storing postsynaptic spikes.
        all_spikes = []
        for j in range(self.trials):
            print("trial {}".format(j))



            # Get parameters in the spontaneous firing mode before showing the stimulus.
            v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = integrate_COBA(
                spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
                to_plot=False, only_f=False, parameter_pass=True, w_ex=w_ex)
            print("after int = {}".format(v_spont))

            # Loop over trials.
            for i in range(self.bars):
                theta = i * np.pi / (bars - 1)

                # Get firing rates of all neurons for trial i (analytic).
                fs_ex = get_fs(theta=theta, theta_synapse=tuned_ex_synapses)
                fs_in = get_fs(theta=theta, theta_synapse=tuned_in_synapses, f_background=f_inhib)

                # Generate Poisson spikes with the analytic firing rate for every neuron.
                if homogeneous == False:
                    spikes_ex, fs_res_ex = generate_spikes_array(fs_ex)
                    spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)
                else:
                    fs_ex = get_fs(theta=theta, theta_synapse=mean * np.ones(N_excit_synapses),
                                   f_background=f_background)
                    fs_in = get_fs(theta=theta, theta_synapse=mean * np.ones(N_inhib_synapses), f_background=f_inhib)
                    spikes_ex, fs_res_ex = generate_spikes_array(f_response=fs_ex, f_background=f_background)
                    spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_inhib)

                # Get the number and the weights of active synapses,
                # defined as ones with firing rate larger than f_max - 0.1 * f_max.
                nr_active_ex[j][i], w_active_ex[j][i], w_avg_ex[j][i] = count_active_synapses(fs_res_ex,
                                                                                              f_active=f_max - 0.1 * f_max,
                                                                                              w=weight_profiles)
                nr_active_in[j][i], w_active_in[j][i], w_avg_in[j][i] = count_active_synapses(fs_res_in,
                                                                                              f_active=f_max - 0.1 * f_max,
                                                                                              w=w_inh)

                # Get the postsynaptic neuron voltage trace (i.e. spikes)
                # by COBA LIF integration. Get a few detailed plots for the 5th, 10th and 15th trials.
                if (j == 0) and (i == 0 or i == 5 or i == 10 or i == 15) and (to_plot == True):
                    fs_out[j, i], spike_times = integrate_COBA(
                        spikes_ex, spikes_in, w_ex=w_ex,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=False, to_plot=True,
                        parameter_pass=False)

                else:
                    fs_out[j, i], spike_times = integrate_COBA(
                        spikes_ex, spikes_in, w_ex=w_ex,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                        parameter_pass=False)

                # Save the spike times.
                all_spikes.append(spike_times)

                # print("f_out = {}".format(fs_out[j,i]))
            # plt.scatter(bars_range, fs_out[j,:], alpha=0.2, color = color_neuron)
            # plt.savefig("output trial {}".format(j))
        # plt.savefig("output_f_theta_all")

        self.f_avg = np.mean(self.fs_out, axis=0)
        self.f_std = np.std(self.fs_out, axis=0)


        #return bars_range, avg, std, all_spikes

    def plot_figs(self):
        # If specified, plot the response curve of the postsynaptic neuron and
        # additional properties.
        avg_nr_ex = np.mean(self.nr_active_ex, axis=0)
        std_nr_ex = np.std(self.nr_active_ex, axis=0)
        avg_nr_in = np.mean(self.nr_active_in, axis=0)
        std_nr_in = np.std(self.nr_active_in, axis=0)

        avg_w_ex = np.mean(self.w_active_ex, axis=0)
        std_w_ex = np.std(self.w_active_ex, axis=0)
        avg_w_in = np.mean(self.w_active_in, axis=0)
        std_w_in = np.std(self.w_active_in, axis=0)

        avg_w_avg_ex = np.mean(self.w_avg_ex, axis=0)
        std_w_avg_ex = np.std(self.w_avg_ex, axis=0)
        avg_w_avg_in = np.mean(self.w_avg_in, axis=0)
        std_w_avg_in = np.std(self.w_avg_in, axis=0)

        plot_soma_response(self.bars_range, self.f_avg, self.f_std, name="PO", name_file=name_file)

        plot_PO_vs_weight(np.abs(self.NeuronConnectivity.PO_e - np.pi / 4) * 180 / np.pi, weight_profiles, name='exc',
                          binary=True)
        plot_PO_vs_weight(np.abs(self.NeuronConnectivity.PO_i - np.pi / 4) * 180 / np.pi, w_inh, name='inh', binary=True)

        plot_fig_3a(self.bars_range, self.f_avg, avg_w_ex, avg_w_in, self.f_std, std_w_ex, std_w_in)
        plot_fig_3b(self.bars_range,
                    avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                    avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)