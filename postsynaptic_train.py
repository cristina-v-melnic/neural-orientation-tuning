from parameters import *
from plotting_setup import *

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
