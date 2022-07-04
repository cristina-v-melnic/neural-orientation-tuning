import random

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


# Neuron Model parameters
V_th = -50.0 # mV
V_rest = -70.0 # mV
V_spike = 0 # mV
E_leak = - 70.0 # mV
E_synapse = 0.0 # mV

tau_membrane = 20.0 # ms
tau_synapse = 5.0 # ms: weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_synapses)
#tau_synapse = 2.0 # ms:  weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.75, size = N_synapses)
g_0 = 0.0

# inhibitory parameters
tau_i = 10 # ms
E_inh = -90 #  mV
g_0_i = 0.0

# Input parameters
N_synapses = 250
N_inhib_synapses = int(N_synapses/5)
N_excit_synapses = N_synapses - N_inhib_synapses

mu, sigma = 0.08, np.sqrt(0.065)
#mu, sigma = 0.015, np.sqrt(0.015)
normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
normal_mean = np.log(mu) - normal_std**2 / 2

#weight_profiles = np.random.lognormal(mean = -2.0, sigma = 0.5, size = N_excit_synapses)
w_inh = (-1) * np.ones(N_inhib_synapses)*(1e-3)
#w_inh = (-1) * np.ones(N_inhib_synapses)*(0.015)
weight_profiles = np.random.lognormal(mean = normal_mean, sigma = normal_std, size = N_excit_synapses)
# The above specific mean and sigma values are chosen s.t. a reasonable background
# firing rate of the output neuron is obtained. (more details search_bg_weights())


f_background = 5 # Hz
f_max = 10 # Hz
f_inhib = 20 # Hz


# Integration parameters
dt = 0.1 # ms
nr_seconds = 80
N_steps = int(nr_seconds * 1000 / dt) # integration time steps
t_start = 0 # ms
t_end = nr_seconds * 1000 # ms
#dt = (t_end - t_start)/N_steps # time step width
time_range = np.linspace(t_start, t_end, N_steps)
save_interval = int(10/dt) # for dt = 0.1 save every 100 ms

burn_seconds = int(nr_seconds/2)
burn_steps = int(burn_seconds * 1000 / dt) # steps or 20 seconds or 20000 ms
stimulus_seconds = 5
stimulus_time_steps = int(stimulus_seconds * 1000 / dt)
silence_seconds = nr_seconds - burn_seconds - stimulus_seconds
silence_time_steps = int(silence_seconds * 1000 / dt)

spikes_pre = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, burn_steps))
spikes_post = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, silence_time_steps))
spikes_inh = np.random.poisson(lam = f_inhib * 10e-4 * dt, size =  (N_inhib_synapses, N_steps))

def get_input_tuning(theta_i = 0, A = 1, N_orientations = 100):
    theta = np.linspace(theta_i - np.pi/2, theta_i + np.pi/2, N_orientations)
    c = f_response / (f_background + 2 * A)
    f = c*(f_background + A + A * np.cos((theta-theta_i)*np.pi/2))
    plt.plot(theta, f)
    plt.title("$\Theta =$ {} rad".format(theta_i))
    plt.xlabel("$\\theta$ (rad)")
    plt.ylabel("$f$ (Hz)")
    plt.show()
    return f

def get_f_result(theta, theta_synapse = [0]):
    #c = f_response / (f_background + 2 * A)
    #f = c * (f_background + A + A * np.cos((theta - theta_synapse) * np.pi*3 / 4))
    #f = 2*A + A * np.cos((theta - theta_synapse) * np.pi * 2 / 3)
    #print(max(theta_synapse))
    #print(min(theta_synapse))
    #print(max(theta - theta_synapse))
    #print(min(theta - theta_synapse))
    B = (f_background-f_max)/(np.cos(1/3 * np.pi * np.pi) - 1)
    A = f_max - B
    f = A + B * np.cos((theta - theta_synapse) * np.pi * 3 /4)
    #print(max(f))
    #print(min(f))
    return f

def get_response_for_bar():
    bars = 20
    fs_out = np.zeros(bars)
    tuned_synapses = tune_all_synapses()
    for i in range(bars):
        theta = -np.pi/2 + i * np.pi / (bars)
        #f_response = get_f_result(theta, theta_synapse=0)
        print(i)
        fs_out[i] = evolve_potential_with_inhibition(theta=theta, name_f="f for bar: {}".format(i), name_V= "V for bar: {}".format(i), theta_synapse=tuned_synapses)
        print("f_out = {}".format(fs_out[i]))
    plt.scatter(np.linspace(-90, 90, bars), fs_out)
    plt.xlabel("$\\theta$")
    plt.ylabel("f")
    #plt.title("Showing a bar of $\\theta$ orientation, 100 tuned synapses at 0 rad with weights = 0.01")
    plt.savefig("output_f_theta")
    plt.show()

def tune_all_synapses():
    tuning_angles = np.random.normal(loc=0.0, scale = 0.55, size=N_excit_synapses)
    #plt.ylabel("Log weights $w_{j}$ (a.u.)")
    #plt.xlabel("Tuning orientation $\\theta_{j}$ (deg)")
    #plt.scatter(tuning_angles * 180/np.pi, np.log(weight_profiles[:N_excit_synapses]), edgecolors = 'black')
    #plt.savefig("weights_per_tuned_synapse")
    #plt.show()
    return tuning_angles


def get_input_response_for_bar():
    bars = 20
    fs_out = np.zeros(bars)
    all_fs = np.zeros((bars, N_excit_synapses))
    tuning_angles = tune_all_synapses()
    w_a = np.zeros(bars)
    f_a = np.zeros(bars)

    for i in range(bars):
        theta = -np.pi / 2 + i * np.pi / (bars)
        all_fs[i,:] = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=tuning_angles))[1]
        fs_out[i] = np.mean(all_fs[i,:])
        nr = 0
        for j, v in enumerate(all_fs[i,:]):
            if v > f_background+1:
                nr += 1
                w_a[i] += weight_profiles[j]
                f_a[i] += v
            if j == (N_excit_synapses-1) and f_a[i] != 0:
                f_a[i] = f_a[i] / (nr)
        #print(fs_out)
    theta_range = np.linspace(-90, 90, bars)

    for i in range(N_excit_synapses):
        plt.scatter(theta_range, all_fs[:,i], alpha=0.3, color = 'lightsteelblue')

    plt.plot(theta_range, fs_out, alpha=1, marker = "o", linewidth = "2", markersize = "12", label = "$\langle f \\rangle$")
    plt.plot(theta_range, all_fs[:,0], alpha=0.7, marker="o", linewidth="2", markersize="12", label = "$f_{i}$")
    plt.plot(theta_range, all_fs[:, 10], alpha=0.7, marker="o", linewidth="2", markersize="12", label="$f_{j}$")
    #plt.plot(theta_range, all_fs[:, 100], alpha=0.7, marker="o", linewidth="2", markersize="12", label="$f_{l}$")

    plt.xlabel("$\\theta$ (deg)")
    plt.ylabel("$f$ (Hz)")
    plt.legend()
    #plt.title("Showing a bar of $\\theta$ orientation, 100 tuned synapses at 0 rad")
    plt.show()
    plt.plot(theta_range, w_a/np.sum(w_a), alpha=0.7, marker="o", linewidth="2", markersize="12", label="$\sum_{j} w$")
    plt.plot(theta_range, f_a/np.sum(f_a), alpha=0.7, marker="o", linewidth="2", markersize="12", label="$\langle f \\rangle_{j}$")
    plt.xlabel("$\\theta$ (deg)")
    plt.ylabel("Normalized quantity (percent)")
    plt.legend()
    plt.savefig("cumulative_w")
    plt.show()


    #with open("presynaptic_f(theta)_tuned_at_0.txt", "w+") as f:
    #    f.write(str(f_response))



def generate_spikes_array(N = N_steps, firing_rate = f_background, f_response = np.ones(N_excit_synapses), N_syn = N_excit_synapses):
    """
    Generate a spike of length N with a frequency of firing_rate.

    :param N: length of spike evolution in time array.
    :param firing_rate: 5 Hz indicates background and 500 Hz activity rates.
    :return: [array of background rate firing of shape = (N_synapses, N)
              number of spikes in (t_final - t_0) time]
    """


    if len(f_response) > 0:
        for i in range(len(f_response)):
            if f_response[i] < f_background:
                f_response[i] = f_background

        spikes_stimulus = np.zeros((len(f_response), stimulus_time_steps))

        for i in range(len(f_response)):
           spikes_stimulus[i, :] = np.random.poisson(lam=f_response[i] * 10e-4 * dt, size= (1, stimulus_time_steps))

        f_spikes_stimulus = np.sum(spikes_stimulus, axis=1) / (stimulus_seconds)

        all_spikes = np.hstack([spikes_pre, spikes_stimulus, spikes_post])
        print(np.shape(all_spikes))

        #for i in range(stimulus_time_steps):
        #    for j in range(100):
        #        spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[0] * 10e-4 * dt)
        #        f_spikes[j]
            #print("f_synaptic_response = {} Hz".format(np.sum(spikes[0,burn_steps : burn_steps + stimulus_time_steps]) / stimulus_seconds))
            #for j in range(100, 150):
             #   spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[1] * 10e-4 * dt)
            #for j in range(150, N_synapses):
               # spikes[j, burn_steps + i] = np.random.poisson(lam=f_response[2] * 10e-4 * dt)
    #return spikes, np.sum(spikes[0, burn_steps:burn_steps + stimulus_time_steps]) / (stimulus_seconds)
        return all_spikes, f_spikes_stimulus
    else:
        raise ValueError("Each synapse must have a corresponding frequency. i.e. size(f_response) = Nr exitatory synapses")


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
        plt.scatter(time_range/1000, (i + 1)/int(N_synapses/spikes_to_see) * all_trains[i, :], label="synapse nr: {}".format(i), marker= "|")

    margin = 1/spikes_to_see
    plt.ylim([1 - margin, spikes_to_see + margin])
    plt.xlabel("time (s)")
    plt.title("Raster plots of {} synapses out of {}. Ensemble $\langle f \\rangle =$ {} Hz.".format(spikes_to_see,N_synapses,f))
    #plt.show()

def test_input_spikes(spikes_array):
    test_background_input_f(spikes_array)
    test_background_input_cv(spikes_array)
    test_visualise_background_input(spikes_array)
    print("Passed all tests for Poisson input spikes!")


def evolve_potential(w_ex = weight_profiles, theta = 0, name_V = "V(t)", name_f = "f(t)", theta_synapse = np.zeros(N_excit_synapses)):
    #theta_synapse = np.asarray([0, np.pi/2, np.pi/4])
    #f_response = get_f_result(theta, theta_synapse=theta_synapse)
    #all_trains, frq = generate_tuned_array()
    #all_trains, frq = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=theta_synapse))
    all_trains, frq = generate_spikes_array(f_response=np.ones(N_excit_synapses)*5)
    #test_input_spikes([all_trains, frq])
    #print("input spike f = {} Hz".format(frq))
    t_spike_trains = np.transpose(all_trains)

    g = g_0
    g_series = []
    v_series = []
    f_series = []
    f_showtime = []
    nr_burned_spikes = 0
    v = V_rest
    #for j in range(100):
    #    w_ex[j] = np.random.lognormal(mean = -2.0, sigma = 0.5, size = 1)


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
            if (i > burn_steps and i < burn_steps+stimulus_time_steps):
                nr_burned_spikes = nr_burned_spikes + 1
            if (i % save_interval == 0):
                v_series.append(v)
                f_series.append(nr_spikes*10000/(save_interval*(len(f_series)+1)))
                if (i > burn_steps and i< burn_steps+stimulus_time_steps):
                    f_showtime.append(nr_burned_spikes*10000/(save_interval*(len(f_showtime)+1)))
            v = V_rest
        else:
            if (i % save_interval == 0):
                v_series.append(v)
                f_series.append(nr_spikes*10000/(save_interval*(len(f_series)+1)))
                if (i>burn_steps and i< burn_steps+stimulus_time_steps):
                    f_showtime.append(nr_burned_spikes * 10000 / (save_interval * (len(f_showtime) + 1)))


    #print("nr spikes = {} in {} s".format(nr_spikes, nr_seconds))
    print("Neuron output f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    #plt.plot(time_range/1000, v_series)
    t = np.linspace(0, nr_seconds, len(v_series))
    plt.plot(t, v_series)
    plt.xlabel("time (s)")
    plt.ylabel("v(t)")
    plt.savefig(name_V)
    plt.figure()

    #plt.plot(t,f_series)
    #plt.xlabel("time (s)")
    #plt.ylabel("f(t)")
    #plt.title("Neuron output f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    #plt.savefig(name_f)
    #plt.figure()

    #plt.plot(np.linspace(80, 100, len(f_showtime)), f_showtime)
    #plt.xlabel("time (s)")
    #plt.ylabel("f(t) post burn")
    #f_post_burn = f_showtime[-1]
    #plt.title("Neuron output post burn f = {} Hz".format(f_post_burn))
    #plt.savefig(name_f+"showtime")
    #plt.figure()

    #f_stim = f_showtime[-1]
    #plt.hist(v_series)
    #plt.show()
    #get_f(v_series=v_series)
    return f_showtime[-1]



def evolve_potential_with_inhibition(w = weight_profiles, theta = 0, name_V = "V(t)", name_f = "f(t)", theta_synapse = np.zeros(N_excit_synapses)):
    all_trains, frq = generate_spikes_array(f_response=get_f_result(theta=theta, theta_synapse=theta_synapse))
    #all_trains, frq = generate_spikes_array(N_syn=N_excit_synapses, f_response=np.ones(N_excit_synapses) * 5)
    #inh_trains, frs_i = generate_spikes_array(N_syn = N_inhib_synapses, f_response=np.ones(N_inhib_synapses) * f_inhib)
    t_spike_trains = np.transpose(all_trains)
    t_inhib_trains = np.transpose(spikes_inh)

    g = g_0
    g_i = g_0_i
    g_series = []
    v_series = []
    f_series = []
    f_showtime = []
    nr_burned_spikes = 0
    v = V_rest
    #for j in range(100):
    #    w_ex[j] = np.random.lognormal(mean = -2.0, sigma = 0.5, size = 1)

    nr_spikes = 0
    for i in range(N_steps):
        g = g - g * dt / tau_synapse + dt * (np.sum(np.multiply(t_spike_trains[i], w)))
        g_i = g_i - g_i * dt / tau_i + dt * (np.sum(np.multiply(t_inhib_trains[i], w_inh)))

        E_eff = (E_leak + g*E_synapse + g_i * E_inh)/(1 + g + g_i)
        tau_eff = tau_membrane/(1 + g + g_i)
        v = E_eff - (v - E_eff) * np.exp(-dt/tau_eff)
        if (v >= V_th):
            nr_spikes = nr_spikes + 1
            v = V_spike
            if (burn_steps and i< burn_steps+stimulus_time_steps): nr_burned_spikes = nr_burned_spikes + 1
            if (i % save_interval == 0):
                v_series.append(v)
                f_series.append(nr_spikes*10000/(save_interval*(len(f_series)+1)))
                if (burn_steps and i< burn_steps+stimulus_time_steps):
                    f_showtime.append(nr_burned_spikes*10000/(save_interval*(len(f_showtime)+1)))
            v = V_rest
        else:
            if (i % save_interval == 0):
                v_series.append(v)
                f_series.append(nr_spikes*10000/(save_interval*(len(f_series)+1)))
                if (burn_steps and i< burn_steps+stimulus_time_steps): f_showtime.append(nr_burned_spikes * 10000 / (save_interval * (len(f_showtime) + 1)))


    #print("nr spikes = {} in {} s".format(nr_spikes, nr_seconds))
    print("Neuron output f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    #plt.plot(time_range/1000, v_series)
    t = np.linspace(0, nr_seconds, len(v_series))
    plt.plot(t, v_series)
    plt.xlabel("time (s)")
    plt.ylabel("v(t)")
    plt.savefig(name_V)
    plt.figure()

    #plt.plot(t,f_series)
    #plt.xlabel("time (s)")
    #plt.ylabel("f(t)")
    #plt.title("Neuron output f = {} Hz".format(nr_spikes*1000/(t_end-t_start)))
    #plt.savefig(name_f)
    #plt.figure()

    #plt.plot(np.linspace(80, 100, len(f_showtime)), f_showtime)
    #plt.xlabel("time (s)")
    #plt.ylabel("f(t) post burn")

    f_post_burn = f_showtime[-1]
    #plt.title("Neuron output post burn f = {} Hz".format(f_post_burn))
    #plt.savefig(name_f+"showtime")
    #plt.figure()
    #plt.hist(v_series)
    #plt.show()
    #get_f(v_series=v_series)
    #return nr_spikes*1000/(t_end-t_start)
    return nr_burned_spikes*1000/stimulus_time_steps

def search_bg_weights():
    fs  = []
    # This one below was tested to work
    w_const = np.ones(N_synapses) * 0.18
    #fs.append(evolve_potential(w_ex = w_const))

    variance = 2.0/(N_synapses+1)
    for i in range(1,10):
        np.random.seed(i)
        weight_profiles = np.random.lognormal(mean=-2.0, sigma=0.5, size=N_synapses)
        #mu = 0.17 # mean range = [0.16 ,0.18] [2 ,10 Hz]
        #sigma = mu * 0.05 # sigma range = [0.04, 0.1] * mu
        #normal_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        #normal_mean = np.log(mu) - normal_std ** 2 / 2
        #weight_profiles = np.random.lognormal(mean=normal_mean, sigma=normal_std, size=N_synapses)
        #weight_profiles = np.random.normal(loc= 0.05, scale= np.sqrt(variance), size = N_synapses)
        #print(weight_profiles)
        plt.hist(weight_profiles)
        plt.savefig("weights rand seed = {}".format(i))
        plt.show()

        fs.append(evolve_potential(weight_profiles, f_response=0, name_f="f rand seed = {}".format(i), name_V="V rand seed = {}".format(i)))

        #if (i%10 == 0):
        #    print(i)

    #print(fs)
    print("The average firing rate = {} Hz and the std = {} ".format(np.mean(fs[1:]), np.std(fs[1:])))
    #plt.hist(fs[1:])
    #plt.figure()

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
    #evolve_potential(f_response=50)
    #search_bg_weights()
    #test_background_input_f()
    #test_background_input_cv()
    #get_input_tuning()
    get_response_for_bar()
    #print(get_f_result(theta = 0, theta_synapse= 0))
    #evolve_potential_with_inhibition()
    #get_input_tuning(theta_i=0, A=1, N_orientations=100)
    #get_input_response_for_bar()
    #tune_all_synapses()


