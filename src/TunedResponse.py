import numpy as np

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

    def __init__(self, NeuronConnectivity, nr_bars, trials, to_plot = False, theta=[]):
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
        if theta==[]:
            self.theta = np.arange(1, self.nr_bars + 1, 1) * np.pi / (self.nr_bars)
        else:
            self.theta = theta
        #self.bars_range = np.linspace(0, 180, self.nr_bars)
        self.bars_range = 180/np.pi * np.asarray(self.theta)
        self.trials = trials

        # Neural connectivity details.
        self.NeuronConnectivity = NeuronConnectivity

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

        self.get_response_for_bar()

    def get_response_for_bar(self):
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

        # Get parameters in the spontaneous firing mode before showing the stimulus.
        Bkg_LIF = SpikyLIF(self.NeuronConnectivity)

        # Loop over trials. (1 trial = 1 sweep of bars
        # through the set of stimulus angles)
        for j in range(self.trials):
            print("Trial Nr:{}".format(j))

            # Loop over the stimulus angles:
            for i in range(self.nr_bars):

                # Get response firing rates of all neurons for trial i (analytic).
                fs_ex = self.get_fs(theta=self.theta[i], theta_synapse=self.NeuronConnectivity.PO_e)
                fs_in = self.get_fs(theta=self.theta[i], theta_synapse=self.NeuronConnectivity.PO_i, f_background=f_inhib)

                # Plot LIF dynamics for trial 0 and certain angles.
                plot_v_data = False
                if self.to_plot == True:
                    if (j == 0) and (i == 0 or i == 5 or i == 10 or i == 15):
                        plot_v_data = True

                # Obtain the postsynaptic activity.
                LIF_i = SpikyLIF(self.NeuronConnectivity, f_e = fs_ex, f_i = fs_in, mode="response",
                                         v = Bkg_LIF.v, g_e = Bkg_LIF.g_e, g_i = Bkg_LIF.g_i, to_plot = plot_v_data,
                                         name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i))

                # Save the postsynaptic firing rate for the response curve.
                self.fs_out[j, i] = LIF_i.f_post
                print("Bar {} deg: f_post  = {} mV".format(self.theta[i], LIF_i.f_post))

                # Get the number and the weights of active synapses,
                # defined as ones with firing rate larger than f_max - 0.1 * f_max.
                #self.nr_active_ex[j][i], self.w_active_ex[j][i], self.w_avg_ex[j][i] = count_active_synapses(LIF_i.f_spikes_e,
                #                                                                              f_active=f_max - 0.1 * f_max,
                #                                                                              w=self.NeuronConnectivity.W_e)
                #self.nr_active_in[j][i], self.w_active_in[j][i], self.w_avg_in[j][i] = count_active_synapses(LIF_i.f_spikes_i,
                #                                                                              f_active=f_max - 0.1 * f_max,
                #                                                                              w=self.NeuronConnectivity.W_e)

        self.f_avg = np.mean(self.fs_out, axis=0)
        self.f_std = np.std(self.fs_out, axis=0)


    def get_fs(self, theta=1.3, theta_synapse=[0], f_background=f_background,
               width=np.pi / 6):
        '''
        Generate the analytical individual tuned response of
        the afferents given the stimulus "theta" and their POs.

        :param theta: (double) The stimulus orientation angle (rad).
        :param theta_synapse: (array) PO of each afferent.
        :param f_background: Background firing rate.
        :param width: Range of angles to be tuned f > f_bkg.

        :return: f (array) Response firing rate of each
                    afferent to the stimulus "theta".
        '''
        delta = np.asarray(theta_synapse) - theta

        # Fitted parameters using desmos.
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

        return f


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

        plot_fig_3a(self.bars_range, self.f_avg, avg_w_ex, avg_w_in, self.f_std, std_w_ex, std_w_in)
        plot_fig_3b(self.bars_range,
                    avg_w_avg_ex, avg_nr_ex, std_w_avg_ex, std_nr_ex,
                    avg_w_avg_in, avg_nr_in, std_w_avg_in, std_nr_in)

    def plot_postsynaptic_curve(self):
        plot_soma_response(self.bars_range, self.f_avg, self.f_std, name="PO", name_file="post")
