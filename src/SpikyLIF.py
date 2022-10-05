import numpy as np

from ConnectedNeuron import *
from parameters import *
from plots import *


class SpikyLIF:
    '''
    Class describing the activity of the postsynaptic neuron
    given the circuit connectivity and spiking activity of the
    presynaptic neurons.
    '''

    def __init__(self, ConnectedNeuron, to_plot = False,
                 f_e = f_background, f_i = f_inhib, mode = "bkg",
                 name_i = "i(t)", name_V = "V(t)", show = False,
                 v = V_rest, g_e = g_0, g_i = g_0_i):
        '''

        :param ConnectedNeuron: Object with connectivity description.
        :param to_plot: [True, False] Plots the postsynaptic trace and
                        membrane currents upon activation and saves them
                        externally;
        :param spikes_e: (nr_excit_neurons x timesteps) Array with values 1/0
                            giving the spiking activity of excitatory afferents;
        :param spikes_i: (Nr_inhib_neurons x timesteps) array analogous to above;
        :param mode: ["bkg", "response"] Type of firing mode.
        :param name_i: Name of the membrane currents plot file;
        :param name_V: Name of the membrane potential plot file;
        :param show: [] If to_plot == True, returns the plots to console
                        in addition to saving them externally.
        '''

        # Afferents weights and firing rates of
        # spike trains.
        self.ConnectedNeuron = ConnectedNeuron
        self.W_e = self.ConnectedNeuron.W_e
        self.W_i = self.ConnectedNeuron.W_i
        self.mode = mode
        self.f_response_e = f_e # Analytic firing rates of
        self.f_response_i = f_i # afferents from each PO.

        # Integration details.
        self.dt = dt

        # Model parameters.
        self.tau = tau_ref

        # Key output for the response curve.
        self.nr_spikes = 0
        self.g_e = g_e
        self.g_i = g_i
        self.v = v

        # Visualisation variables.
        self.to_plot = to_plot
        self.show = show
        self.name_i = name_i
        self.name_V = name_V
        self.v_series = []
        self.v_zoom_series = []
        self.I_ex = []
        self.I_in = []
        self.t_f_zoom = 1.5  # s

        # List to store the spike times for storage optimality (not used yet).
        self.spike_times = []

        # Generate and assign the spike trains
        # of the afferents.
        self.spikes_e = [0]
        self.spikes_i = [0]

        self.fs_spikes_e = [0] # Firing rates of afferents from the
        self.fs_spikes_i = [0] # generated Poisson spike trains
        self.assign_spikes_arrays() # Populates the empty vars above.

        # Get the trial duration from the spike data.
        self.time_steps = len(self.spikes_e[0])
        self.t_f = self.time_steps * dt / 1000  # total time in seconds
        self.f_post = self.nr_spikes / (self.time_steps * 0.1 / 1000)

        # Get the postsynaptic spike train.
        self.integrate_COBA()

        # Plot parameters od postsynaptic activity
        # upon specification.
        if self.to_plot == True:
           self.visualise_COBA()


    def integrate_COBA(self):
        '''
        Conductance-based leaky integrate-and-fire integration
        to obtain the voltage traces/spike times of the post-
        synaptic neuron.
        '''

        # Transposing the spike train for
        # performing the dot product.
        t_ex_trains = np.transpose(self.spikes_e)
        t_in_trains = np.transpose(self.spikes_i)

        # Integration loop.
        for i in range(self.time_steps):
            self.tau += 1
            # Update conductances of both E and I populations
            # from the spikes of afferents.
            self.g_e = self.g_e - self.g_e * self.dt / tau_synapse + \
                  self.dt * (np.sum(np.multiply(t_ex_trains[i], self.W_e)))
            self.g_i = self.g_i - self.g_i * self.dt / tau_i + \
                  self.dt * (np.sum(np.multiply(t_in_trains[i], self.W_i)))

            # Use the analytic solution to update V_t+1.
            E_eff = (E_leak + self.g_e * E_synapse + self.g_i * E_inh) / (1 + self.g_e + self.g_i)
            tau_eff = tau_membrane / (1 + self.g_e + self.g_i)
            self.v = E_eff - (self.v - E_eff) * np.exp(-self.dt / tau_eff)

            # Leaky integrate-and-fire.
            if (self.v >= V_th) and (self.tau > tau_ref):
                self.nr_spikes = self.nr_spikes + 1
                self.spike_times.append(i)
                self.v = V_spike

                # Save data for visualisation.
                if self.to_plot == True:
                    self.save_dynamics(i)

                # Reset the potential after the spike and the refractory period.
                self.v = V_rest
                self.tau = 0
            else:
                # Save data for visualisation.
                if self.to_plot == True:
                    self.save_dynamics(i)
        self.f_post = self.nr_spikes / (self.time_steps * 0.1 / 1000)


    def visualise_COBA(self):
        '''
        Plots of the voltage traces and membrane currents.
        '''
        # Voltage trace during the whole integration period.
        plot_v_trace(self.t_f, self.v_series, name_V = self.name_V, show = self.show)

        # Voltage trace and membrane potential zoomed to the
        # time window of the first 3 spikes.
        plot_current_zoomed(self.t_f_zoom, self.I_in, self.I_ex, name_i = self.name_i, show = self.show)
        plot_voltage_zoomed(self.t_f_zoom, self.v_zoom_series, name_V = self.name_V, show = self.show)


    def save_dynamics(self, i):
        '''
        Save lists of voltage traces and membrane currents.

        :param i: the iteration of the last recorded spike
                    for the zoomed in visualisation.
        :return:
        '''
        self.v_series.append(self.v)
        # Save all v and I data up to the 3rd spike.
        if self.nr_spikes < 3:
            self.I_ex.append(self.g_e * (E_synapse - self.v))
            self.I_in.append(self.g_i * (E_inh - self.v) + (E_leak - self.v))
            self.v_zoom_series.append(self.v)
            self.t_f_zoom = i


    # Get response of afferents for a given stimulus (array f_response).
    def assign_spikes_arrays(self):
        '''
        Assign the right spike trains of afferents
        depending on whether the system is in the
        backroung firing mode or not.
        '''
        # If in the background firing mode:
        if self.mode == "bkg":
            self.spikes_e = spikes_pre
            self.spikes_i = spikes_inh_pre
        else:
            self.spikes_e, self.fs_spikes_e = self.get_poisson_spike_train(self.f_response_e)
            self.spikes_i, self.fs_spikes_i = self.get_poisson_spike_train(self.f_response_i)

    def get_poisson_spike_train(self, f_response):
        """
        Generate a spike of length N with a frequency of firing_rate.

        :param N: length of spike evolution in time array.
        :param firing_rate: 5 Hz: the background firing rate.
        :return: [array of background rate firing of shape = (N_synapses, N)
                  number of spikes in (t_final - t_0) time]
        """

        # Make sure no neuron is firing below the background rate.
        if len(f_response) > 0:
            for i in range(len(f_response)):
                if f_response[i] < f_background:
                    f_response[i] = f_background
        else:
            raise ValueError(
                "Each synapse must have a corresponding frequency. i.e. size(f_response) = Nr exitatory synapses")
        # Get the spike train for each afferent.
        spikes_stimulus = np.zeros((len(f_response), stimulus_time_steps))
        fs = np.zeros(len(f_response))

        for i in range(len(f_response)):
            train = np.random.poisson(lam=f_response[i] * 10e-4 * dt, size=(1, stimulus_time_steps))
            fs[i] = len(np.nonzero(train)[0]) / stimulus_seconds
            spikes_stimulus[i] = train
        return spikes_stimulus, fs
