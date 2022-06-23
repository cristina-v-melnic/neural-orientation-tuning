import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Neuron Model parameters
V_th = -50.0 # mV
V_rest = -70.0 # mV
V_spike = 0 # mV
E_leak = - 70.0 # mV
E_synapse = 0.0 # mV

tau_membrane = 30.0 # ms
tau_synapse = 2.0 # ms

g_0 = 0.0

# Input parameters
N_synapses = 200
weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.75, size = N_synapses)
# The above specific mean and sigma values are chosen s.t. a reasonable background
# firing rate of the output neuron is obtained. (more details search_bg_weights())
f_background = 5 # Hz

# Integration parameters
nr_seconds = 10
N_steps = nr_seconds * 10000 # integration time steps
t_start = 0 # ms
t_end = nr_seconds * 1000 # ms
dt = (t_end - t_start)/N_steps # time step width
time_range = np.linspace(t_start, t_end, N_steps)



def generate_spikes_array(N = N_steps, firing_rate = f_background, N_syn = N_synapses):
    """
    Generate a spike of length N with a frequency of firing_rate.

    :param N: length of spike evolution in time array.
    :param firing_rate: 5 Hz indicates background and 500 Hz activity rates.
    :return: [array of background rate firing of shape = (N_synapses, N)
              number of spikes in (t_final - t_0) time]
    """
    spikes = np.random.poisson(lam = firing_rate * 10e-4 * (t_end-t_start)/N, size = (N_syn,N))
    return spikes, np.sum(spikes)/(N_syn * nr_seconds)

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


def evolve_potential(w_ex = weight_profiles):

    all_trains, frq = generate_spikes_array()
    test_input_spikes([all_trains, frq])
    print("avg nr spikes = {} in {} ms".format(frq, (t_end-t_start)))
    t_spike_trains = np.transpose(all_trains)

    g = g_0
    g_series =[]
    v_series = []
    v = V_rest
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
            v_series.append(v)
            v = V_rest
        else:
            v_series.append(v)

    print("nr spikes = {}".format(nr_spikes))
    print("Voltage f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    plt.plot(time_range/1000, v_series)
    plt.xlabel("time (s)")
    plt.ylabel("v(t)")
    plt.show()
    #plt.hist(v_series)
    #plt.show()
    get_f(v_series=v_series)
    return nr_spikes*1000/(t_end-t_start)

def search_bg_weights():
    fs  = []
    # This one below was tested to work
    w_const= np.ones(N_synapses) * 0.3
    fs.append(evolve_potential(w_ex = w_const))
    for i in range(10):
        weight_profiles = np.random.lognormal(mean=-2.0, sigma=0.75, size=N_synapses)
        fs.append(evolve_potential(weight_profiles))

    #print(fs)
    print("The average firing rate in 10 trials = {}".format(np.sum(fs[1:])/10))

def get_f(v_series):
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

    plt.plot(np.linspace(1,t_end/(1000),len_boxes),f_series, label=" $\langle f \ (T) \\rangle \ =$ {} Hz".format(spike_nr*1000/(t_end-t_start)))
    plt.ylabel("$\langle f \ (t) \\rangle$ (Hz)")
    plt.xlabel("t (s)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #print(generate_spikes_array()[1])
    evolve_potential()
    #search_bg_weights()
    #test_background_input_f()
    #test_background_input_cv()