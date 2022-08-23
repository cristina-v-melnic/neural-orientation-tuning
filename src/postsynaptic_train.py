from plots import *

def evolve_potential_with_inhibition(spikes_ex, spikes_in,
                                     v = V_rest, g_e = g_0, g_i = g_0_i,
                                     tau = tau_ref, nr_spikes = 0,
                                     v_series = [], I_ex = [], I_in=[],
                                     w_ex = weight_profiles, w_in = w_inh,
                                     name_V = "V(t)", name_i = "i(t)",
                                     to_plot = True, only_f = True, parameter_pass = False):
    '''
    Numerically integrate the conductance based leaky integrate-and-fire equations to get
    the postsynaptic firing rate from the presynaptic afferents upon stimulus presentation.

    :param spikes_ex: (array) Spike trains of excitatory afferents.
    :param spikes_in: (array) Spike trains of inhibitory afferents.

    :param v: (float) Initial value of membrane potential.
    :param g_e: (float) Excitatory conductance initial value.
    :param g_i: (float) Inhibitory conductance initial value.
    :param tau: (int) Refractory period in time steps.
    :param nr_spikes: (int) Container for nr of spikes.

    :param v_series: (list) Container for selected spikes.
    :param I_ex: (list) Container for excitatory current vals.
    :param I_in: (list) Container for inhibitory current vals.

    :param w_ex: (array) Weights of excitatory afferents.
    :param w_in: (array) Weights of inhibitory afferents.

    :param name_V: (str)
    :param name_i: (str)

    :param to_plot:
    :param only_f: (bool) Return only firing rate of postsynaptic neuron.
    :param parameter_pass: (bool) Only pass parameters for a subsequent computation.


    :return: Postsynaptic spike train parameters and/or plots, depending on specifications.
    '''

    # Record the initial lenth of the spike train.
    initial = len(v_series)

    # Get the trial duration from the spike data.
    time_steps = len(spikes_ex[0])

    # List to store the spike times for storage optimality (not used yet).
    spike_times = []

    # Transpose the spike trains from afferents to use for the dot product.
    t_ex_trains = np.transpose(spikes_ex)
    t_in_trains = np.transpose(spikes_in)

    # Vectors to store data for visualisation.
    if to_plot == True:
        v_zoom_series = []
        I_ex = []
        I_in = []
        t_f_zoom = 1.5 #s

    # Integration loop.
    for i in range(time_steps):
        tau += 1
        # Update conductances of both E and I populations from the spikes of afferents.
        g_e = g_e - g_e * dt / tau_synapse + dt * (np.sum(np.multiply(t_ex_trains[i], w_ex)))
        g_i = g_i - g_i * dt / tau_i + dt * (np.sum(np.multiply(t_in_trains[i], w_in)))

        # Use the analytic solution to update V_t+1.
        E_eff = (E_leak + g_e * E_synapse + g_i * E_inh)/(1 + g_e + g_i)
        tau_eff = tau_membrane/(1 + g_e + g_i)
        v = E_eff - (v - E_eff) * np.exp(-dt/tau_eff)

        # Leaky integrate-and-fire.
        if (v >= V_th) and (tau > tau_ref):

            nr_spikes = nr_spikes + 1
            spike_times.append(i)
            v = V_spike

            if to_plot == True:
                v_series.append(v)

                # Save all v and I data up to the 3rd spike.
                if nr_spikes < 3:
                    v_zoom_series.append(v)
                    I_ex.append(g_e * (E_synapse - V_rest))
                    I_in.append(g_i * (E_inh - V_rest) + (E_leak - V_rest))
                    t_f_zoom = i

            # Reset the potential after the spike and the refractory period.
            v = V_rest
            tau = 0
        else:
            if to_plot == True:
                # Save all v and I data up to the 3rd spike.
                if nr_spikes < 3:
                    I_ex.append(g_e * (E_synapse - v) )
                    I_in.append(g_i * (E_inh - v)+ (E_leak-v))
                    v_zoom_series.append(v)
                    t_f_zoom = i
                # Save only every other spike for visualisation.
                if (i % save_interval == 0):
                    v_series.append(v)

    # Check if the spike train is only from this simulation
    # or is a continuation of another trial.
    if (len(v_series) == initial):
         t_f = time_steps * dt / 1000
         print("only stimulus")
    else:
        t_f = (len(v_series) + time_steps)* 0.1 / 1000

    # Print the final firing rate after each integration.
    print("Neuron output f = {} Hz".format(nr_spikes/t_f))

    # Choose to return only the firing rate,
    # the firing rate and plots,
    # or to pass parameters and continue the integration.
    if (only_f == True):
        return nr_spikes / (time_steps * 0.1 / 1000), spike_times

    elif (to_plot == True):

        plot_v_trace(t_f, v_series, name_V=name_V)
        plot_current_zoomed(t_f_zoom, I_in, I_ex, name_i=name_i)
        plot_voltage_zoomed(t_f_zoom, v_zoom_series, name_V=name_V)

        return nr_spikes/(time_steps * 0.1 / 1000), spike_times

    elif (parameter_pass == True):
        return v, g_e, g_i, tau, nr_spikes, v_series, I_ex, I_in
