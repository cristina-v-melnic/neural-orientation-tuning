import random

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(598)

# Neuron Model parameters
V_th = -50.0 # mV
V_rest = -70.0 # mV
V_spike = 0 # mV
E_leak = - 70.0 # mV
E_synapse = 0.0 # mV

tau_membrane = 30.0 # ms
tau_synapse = 5.0 # ms: weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_synapses)
#tau_synapse = 2.0 # ms:  weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.75, size = N_synapses)
g_0 = 0.0

# Input parameters
N_synapses = 200
weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_synapses)
# The above specific mean and sigma values are chosen s.t. a reasonable background
# firing rate of the output neuron is obtained. (more details search_bg_weights())
f_background = 5 # Hz
f_response = 500 # Hz

# Integration parameters
dt = 0.1 # ms
nr_seconds = 150
N_steps = int(nr_seconds * 1000 / dt) # integration time steps
t_start = 0 # ms
t_end = nr_seconds * 1000 # ms
#dt = (t_end - t_start)/N_steps # time step width
time_range = np.linspace(t_start, t_end, N_steps)



def generate_spikes_array(N = N_steps, firing_rate = f_background, f_response = 500, N_syn = N_synapses):
    """
    Generate a spike of length N with a frequency of firing_rate.

    :param N: length of spike evolution in time array.
    :param firing_rate: 5 Hz indicates background and 500 Hz activity rates.
    :return: [array of background rate firing of shape = (N_synapses, N)
              number of spikes in (t_final - t_0) time]
    """
    spikes = np.random.poisson(lam = firing_rate * 10e-4 * dt, size = (N_syn,N))
    print("f_spont = {} Hz".format(np.sum(spikes)/(N_syn*nr_seconds)))

    if f_response != 0:
        burn_steps = 500000 # steps or 50 seconds or 50000 ms
        stimulus_seconds = 50
        stimulus_time_steps = int(stimulus_seconds * 1000 / dt)

        for i in range(stimulus_time_steps):
            spikes[0, burn_steps + i] = np.random.poisson(lam=f_response * 10e-4 * dt)
        print("f_synaptic_response = {} Hz".format(np.sum(spikes[0,burn_steps : burn_steps + stimulus_time_steps]) / stimulus_seconds ))
        return spikes, np.sum(spikes[1:]) / ((N_syn-1) * nr_seconds)
    else:
        return spikes, np.sum(spikes)/(N_syn * nr_seconds)

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

def test_visualise_background_input(spikes_array, spikes_to_see = 10):
    all_trains, f = spikes_array
    interval = int(N_synapses/spikes_to_see)-1

    #print(np.shape(all_trains))
    for i in range(0, N_synapses, interval):
        plt.scatter(time_range/1000, (i +1)/int(N_synapses/spikes_to_see) * all_trains[i, :], label="synapse nr: {}".format(i), marker= "|")

    margin = 1/spikes_to_see
    plt.ylim([1 - margin, spikes_to_see + margin])
    plt.xlabel("time (s)")
    plt.title("Raster plots of {} synapses out of {}. Ensemble $\langle f \\rangle =$ {} Hz.".format(spikes_to_see,N_synapses,f))
    plt.show()

def test_input_spikes(spikes_array):
    test_background_input_f(spikes_array)
    test_background_input_cv(spikes_array)
    test_visualise_background_input(spikes_array)
    print("Passed all tests for Poisson input spikes!")


def evolve_potential(w_ex = weight_profiles, f_response = 0):

    #all_trains, frq = generate_tuned_array()
    all_trains, frq = generate_spikes_array(f_response=f_response)
    #test_input_spikes([all_trains, frq])
    #print("input spike f = {} Hz".format(frq))
    t_spike_trains = np.transpose(all_trains)

    g = g_0
    g_series =[]
    v_series = []
    v = V_rest
    w_ex[0] = 0.6
    nr_spikes = 0
    for i in range(N_steps):
        g = g - g*dt/tau_synapse + dt * (np.sum(np.multiply(t_spike_trains[i], w_ex)))
        g_series.append(g)
        E_eff = (E_leak + g*E_synapse)/(1 + g)
        tau_eff = tau_membrane/(1+g)
        v = E_eff - (v - E_eff) * np.exp(-dt/tau_eff)
        if (v >= V_th):
            nr_spikes = nr_spikes + 1
            v = V_spike
            if (i%100==0): v_series.append(v)
            v = V_rest
        else:
            if (i%100==0): v_series.append(v)

    #print("nr spikes = {} in {} s".format(nr_spikes, nr_seconds))
    print("Neuron output f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    #plt.plot(time_range/1000, v_series)
    plt.plot(np.linspace(0,nr_seconds,len(v_series)), v_series)
    plt.xlabel("time (s)")
    plt.ylabel("v(t)")
    plt.show()
    #plt.hist(v_series)
    #plt.show()
    #get_f(v_series=v_series)
    return nr_spikes*1000/(t_end-t_start)

def search_bg_weights():
    fs  = []
    # This one below was tested to work
    w_const = np.ones(N_synapses) * 0.18
    fs.append(evolve_potential(w_ex = w_const))
    np.random.seed(0)
    variance = 2.0/(N_synapses+1)
    for i in range(3):
        weight_profiles = np.random.lognormal(mean=-2.0, sigma=0.5, size=N_synapses)
        #weight_profiles = np.random.normal(loc= 0.05, scale= np.sqrt(variance), size = N_synapses)
        #print(weight_profiles)
        fs.append(evolve_potential(weight_profiles))
        #if (i%10 == 0):
        #    print(i)

    print(fs)
    print("The average firing rate = {} Hz and the std = {} ".format(np.mean(fs[1:]), np.std(fs[1:])))
    plt.hist(fs[1:])
    plt.show()

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


if __name__ == '__main__':
    #print(generate_spikes_array()[1])
    #evolve_potential()
    evolve_potential(f_response=50)
    #search_bg_weights()
    #test_background_input_f()
    #test_background_input_cv()