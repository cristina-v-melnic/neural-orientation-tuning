import numpy as np
from parameters import *
from plots import *


class SpikyLIF:
    '''
    Class describing the activity of the postsynaptic neuron
    given the circuit connectivity and spiking activity of the
    presynaptic neurons.
    '''

    def __init__(self, ConnectedNeuron, to_plot = False,
                 spikes_e = spikes_pre, spikes_i = spikes_inh_pre,
                 name_i = "i(t)", name_V = "V(t)", show = False):
        '''

        :param ConnectedNeuron: Object with connectivity description.
        :param to_plot: [True, False] Plots the postsynaptic trace and
                        membrane currents upon activation and saves them
                        externally;
        :param spikes_e: (nr_excit_neurons x timesteps) Array with values 1/0
                            giving the spiking activity of excitatory afferents;
        :param spikes_i: (Nr_inhib_neurons x timesteps) array analogous to above;
        :param name_i: Name of the membrane currents plot file;
        :param name_V: Name of the membrane potential plot file;
        :param show: [] If to_plot == True, returns the plots to console
                        in addition to saving them externally.
        '''

        # Afferents weights and spike trains.
        self.ConnectedNeuron = ConnectedNeuron
        self.W_e = self.ConnectedNeuron.W_e
        self.W_i = self.ConnectedNeuron.W_i
        self.spikes_e = spikes_e
        self.spikes_i = spikes_i

        # Integration details.
        self.dt = dt
        # Get the trial duration from the spike data.
        self.time_steps = len(self.spikes_e[0])

        # Model parameters.
        self.tau = tau_ref

        # Key output for the response curve.
        self.nr_spikes = 0
        self.f_post = self.nr_spikes / (self.time_steps * 0.1 / 1000)
        self.g_e = g_0
        self.g_i = g_0_i
        self.v = V_rest

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
        self.t_f = self.time_steps * dt / 1000  # total time in seconds

        # List to store the spike times for storage optimality (not used yet).
        self.spike_times = []



        self.integrate_COBA()
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