from numpy import np
from plotting_setup import *
from utensils_stimulus_sweep import *

def get_spike_indexes(spike_train, f_array = False):
    '''
    Function that transforms the array with Poisson distributed spikes (0 or 1)
    into an array that only stores the indexes of spikes, thereby avoiding the
    unnecessary storage of data.

    :param spike_train: Array of shape (N_syn, N_steps) - containing Poisson
                        distributed 0s or 1s for each presynaptic input.
           f_array: 'False' by default - bool parameter to indicate the
                     output of the individual presynaptic firing rates.

    :return: Default:
                      all_indexes (N_syn, different lengths) - containing only the
                      indexes of spikes for each presynaptic input.
             Upon specification:
                      List of arrays [all_indexes, all_fs (N_syn)] - where
                      the additional array gives the firing frequency of each
                      presynaptic input.
    '''

    N = len(spike_train)
    all_indexes = []

    if (f_array == False):
        for i in range(N):
            indexes = np.nonzero(spike_train[i])[0]
            all_indexes.append(indexes)
        return all_indexes
    else:
        all_fs = np.zeros(N)
        nr_seconds = len(spike_train[0]) * dt / 1000
        for i in range(N):
            indexes = np.nonzero(spike_train[i])[0]
            all_indexes.append(indexes)
            all_fs[i] = len(indexes) / nr_seconds
        return all_indexes, all_fs

def get_spiked_neuron_indexes(spike_train, t = 398400):
    N_neurons = len(spike_train)

    spiked = []
    # Search over all indexes for values matching the timestep number
    # and save the spiking neurons at that timestep in a list.
    for n in range(N_neurons):
        if np.any(np.equal(spike_train[n], t)):
            spiked.append(n)

    if (len(spiked) != 0):
        return spiked
    else:
        return False

def optimize_conductance_comp(g, g_i):
    '''

    Didn't really work. It took too long to compute.
    Try later.

    :param g:
    :param g_i:
    :return:
    '''
    g = g - g * dt / tau_synapse
    g_i = g_i - g_i * dt / tau_i

    ind_ex = get_spiked_neuron_indexes(all_trains)
    ind_in = get_spiked_neuron_indexes(spikes_inh)

    if ind_ex != False:
        g = g + dt * np.sum(np.take(weight_profiles, ind_ex))
        print(g)

    if ind_in != False:
        g_i = g_i + dt * np.sum(np.take(w_inh, ind_in))
        print(g_i)

    return g, g_i

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


def get_input_response_for_bar():
    bars = 10
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
    plt.ylabel("Normalized quantity (percent)", labelpad = 2)
    plt.legend()
    plt.savefig("cumulative_w.svg")
    plt.savefig("cumulative_w.png")


def tune_all_synapses(mean_ex = np.pi/4, mean_in = 0.0, std_ex = np.pi/10 , std_in = np.pi / 5,  to_plot = True):

    # OS from W
    tuning_angles_ex = np.abs(np.random.normal(loc=mean_ex, scale=std_ex, size=N_excit_synapses))
    #tuning_angles_in = np.random.normal(loc=mean_in, scale=std_in, size=N_inhib_synapses)
    #tuning_angles_in = np.random.uniform(mean_in - np.pi/2, mean_in + np.pi/2, size=N_inhib_synapses)
    for i in range(len(tuning_angles_ex)):
        if tuning_angles_ex[i] > np.pi or tuning_angles_ex[i]<0:
            if i%3 == 0:
                tuning_angles_ex[i] = np.random.uniform(mean_ex-std_ex/2, mean_ex+std_ex/2)
            else:
                tuning_angles_ex[i] = mean_ex

    tuning_angles_in = np.linspace(0, np.pi, num=N_inhib_synapses)
    # OS from <w>
    #tuning_angles_ex = np.linspace(mean_ex - np.pi / 2, mean_ex + np.pi / 2, num=N_excit_synapses)
    #range = np.linspace(0, np.pi, num=N_excit_synapses)
    #tuning_angles_ex = gaussian(range, mean_ex, std_ex)


    if (to_plot == True):
        #diff = np.histogram(tuning_angles_ex)[0]- np.histogram(tuning_angles_in)[0]
        #print(diff)

        plt.hist(tuning_angles_ex * 180/np.pi, label= "$N_{E}$", alpha = 0.8, color = 'tab:orange')
        #plt.hist((tuning_angles_ex - tuning_angles_in) * 180 / np.pi, label="$N_{E}-N_{I}$", alpha=0.8, color='tab:green')

        plt.hist(tuning_angles_in * 180/np.pi, label="$N_{I}$", alpha = 0.8, color = 'tab:blue')

        #bins = 15
        #t = np.linspace(0.0,1.0, bins)

        #plt.hist(tuning_angles_ex * 180 / np.pi, label="$N_{E}$", alpha=0.8, color=cmap(t), bins =bins)

        low_bound = np.min([np.min(tuning_angles_ex) * 180/np.pi, np.min(tuning_angles_in)]) * 180/np.pi
        up_bound = np.max([np.max(tuning_angles_ex) * 180/np.pi, np.max(tuning_angles_in)]) * 180/np.pi
        #shift = (up_bound-low_bound)/(2*len(diff))
        #plt.plot(np.linspace(low_bound-shift, up_bound-shift, len(diff)),
        #            diff, marker = 'o', markersize = 30, color = 'grey', label = "$N_{E}-N_{I}$" )
        #plt.xlabel("Tuning orientation $\\theta_{j}$ (deg)")
        plt.xlabel("PO (deg)")
        #plt.xlim([-95,95])
        plt.locator_params(axis='y', nbins=5)
        plt.legend(labelcolor='linecolor')
        #plt.savefig("weights_per_tuned_synapse")
        plt.savefig('tuning_range.svg')
        plt.savefig('tuning_range.png')


        plt.figure()
    return tuning_angles_ex, tuning_angles_in


def get_f_result(theta, theta_synapse = [0], f_background = f_background):
    #c = f_response / (f_background + 2 * A)
    #f = c * (f_background + A + A * np.cos((theta - theta_synapse) * np.pi*3 / 4))
    #f = 2*A + A * np.cos((theta - theta_synapse) * np.pi * 2 / 3)
    #print(max(theta_synapse))
    #print(min(theta_synapse))
    #print(max(theta - theta_synapse))
    #print(min(theta - theta_synapse))
    B = (f_background-f_max)/(np.cos(1/3 * np.pi * np.pi) - 1)
    A = f_max - B
    f = A + B * np.cos((theta - theta_synapse) * np.pi * 2/3)
    #print(max(f))
    #print(min(f))
    return f

def get_fs_past(theta = 1.3, theta_synapse = [0], f_background = f_background):
    s = 0.4
    delta = theta - np.asarray(theta_synapse)
    #return f_background/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    f = f_background + (f_max - f_background)/(s * np.sqrt(2 * np.pi)) * np.exp(- delta**2/(2 * s**2))
    #plt.scatter(delta, f)
    #plt.show()
    return f


def try_multiple_neurons(n=2):
    bars = 11
    f = np.zeros((n, bars))
    f_std = np.zeros((n, bars))
    PO = np.zeros(n)
    viridis = cm.get_cmap('viridis', n)


    for j in range(n):
        print("Neuron nr: {}.".format(j))
        x_axis, f[j], f_std[j], PO[j], all_spikes = get_response_for_bar(trials=1, to_plot=False, mean = j % 2 * np.pi/2)
        print(f[j])
        numpy.savetxt("neuron{}".format(j), f[j])

    #plot_soma_response(x=x_axis, y=np.mean(f,axis=0), err=np.sqrt(np.sum(np.square(f_std),axis=0))/n, name="PO", PO = [np.average(PO),np.std(PO)])
    #print(str(np.average(PO))+" "+str(np.std(PO)))

def test_background_input_f(spikes_array):
    f = spikes_array[1]
    print(f)
    assert f < f_background + 1, "It should be around {} Hz.".format(f_background)

def test_background_input_cv(spikes_array):
    input_spikes = spikes_array[0]
    sigma = np.std(input_spikes)
    cv = sigma/np.mean(input_spikes)
    assert  cv < 1/sigma+1, "Not a Poisson distribution."

def test_visualise_background_input(spikes_array, spikes_to_see=10, name="excitatory", f = 5):
    all_trains = spikes_array
    N_synapses = len(spikes_array)

    #interval = int(N_synapses/spikes_to_see)-1
    interval = 1
    spikes_to_see = N_synapses
    nr_s = 1
    time_range = np.linspace(t_start, 1000, len(all_trains[0,:nr_s*10000]))

    #print(np.shape(all_trains))
    for i in range(2, N_synapses, interval):
        plt.scatter(time_range, (i+1)/int(N_synapses/spikes_to_see) * all_trains[i, :nr_s * 10000], label="synapse nr: {}".format(i), s = 10, marker= "D", color = "white")



    margin = 2/spikes_to_see
    plt.ylim([1 - margin, spikes_to_see + 2])
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell number")
    #plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.svg".format(spikes_to_see,name,N_synapses,f))
    plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.png".format(spikes_to_see, name, N_synapses, f))
    plt.figure()
    #plt.show()

def test_input_spikes(spikes_array, name = "excitatory", f = 5):
    #test_background_input_f(spikes_array)
    #test_background_input_cv(spikes_array)
    test_visualise_background_input(spikes_array, name = name, f = f)
    print("Passed all tests for Poisson input spikes!")

def get_f_isi_and_cv(spikes_arrays, to_plot = True, nr_seconds = stimulus_seconds, name = "", color=cmap(0.0)):
    """

    :param spikes_arrays: spikes_array = [Excitatory spike train array, Inhibitory spike trains]
                          - list of arrays of arbitrary dimensions
    :return: Plots firing rates, ISI and CV of ISI for both E/I populations.
    """

    if to_plot == True: both_isi = []
    both_fs = []
    spikes_array = spikes_arrays

    #for spikes_array in spikes_arrays:
    if True:
        N = len(spikes_array)
        if to_plot == True:
            all_isi = []
            all_indexes = []
        all_fs = np.zeros(N)
        for i in range(N):
            indexes = np.nonzero(spikes_array[i])
            all_fs[i] = len(indexes[0])/nr_seconds
            if to_plot == True:
                all_isi.append(np.diff(indexes, axis=1) * dt)
                if i%5 == 0:
                    plt.eventplot(np.asarray(indexes)*0.0001, color=color, lineoffsets=i, linelengths = 3)


        # The None term is added because the arrays in the list
        # have different sizes.
        if to_plot == True:
            all_isi = np.concatenate(all_isi, axis=None).ravel()
            both_isi.append(all_isi)
            all_indexes.append(indexes)
        both_fs.append(all_fs)

    if to_plot == True:
        plt.xlabel("Time (s)")
        #plt.ylabel("Cell number")

        ax = plt.gca()
        #ax.set_figwidth(4)
        #ax.set_figheight(1)
        ax.spines.left.set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.savefig("{} {} synapses out of {}. Ensemble f = {:.2f} Hz.svg".format(spikes_to_see, name, N_synapses, f))
        plt.savefig("Raster plot = {}.svg".format(name))
        plt.savefig("Raster plot = {}.png".format(name))
        plt.figure()


        #labels = ['Excitatory', 'Inhibitory']
        #colors = ['tab:orange', 'tab:blue']
        labels = ['Input']
        colors = ['green']

        #for i, fs in enumerate(both_fs):
        fs = both_fs[0]
        if True:
            plt.hist(fs, weights=np.ones_like(fs) / len(fs), label=labels[0], color=color, alpha = 0.9)
            plt.xlabel("f (Hz)")
            plt.ylabel("Percent")

            #plt.legend()
            plt.savefig("Averate firing rates trial = {}.svg".format(name))
            plt.savefig("Averate firing rates trial = {}.png".format(name))
        plt.figure()

        #for i, isi in enumerate(both_isi):
        isi = all_isi
        if True:
            plt.hist(isi, weights=np.ones_like(isi) / len(isi), label=labels[0], color=color, alpha = 0.8)
            plt.xlabel("ISI (ms)")
            plt.ylabel("Percent")
            #plt.legend()
            plt.savefig("Interspike intervals trial = {}.svg".format(name))
            plt.savefig("Interspike intervals trial = {}.png".format(name))
        plt.figure()

        #for i, isi in enumerate(both_isi):
        if True:
            mean = np.mean(isi)
            cv = np.sqrt((isi - mean) ** 2) / mean
            plt.hist(cv, weights=np.ones_like(cv) / len(cv), label=labels[0], color=color, alpha = 0.8)
            plt.xlabel("CV of ISI")
            plt.ylabel("Percent")
            #plt.legend()
            plt.savefig("CV of ISI trial = {}.svg".format(name))
            plt.savefig("CV of ISI trial = {}.png".format(name))
        plt.figure()


        #plt.xlabel("Time (ms)")
        # plt.legend()
        # Draw a spike raster plot


        #interval = 1
        #spikes_to_see = 5
        #nr_s = 1
        #time_range = np.linspace(t_start, 1000, len(spikes_array[0, :nr_s * 10000]))

        #for i in range(2, 50, interval):
        #    plt.scatter(time_range, i/10*spikes_array[i, :nr_s * 10000], label="synapse nr: {}".format(i), s=20, marker="|", color=color)

        #margin = 2 / spikes_to_see
        #plt.ylim([1 - margin, spikes_to_see + 2])


def plot_f_analytic():
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    plt.figure()
    plt.plot(theta*180/np.pi, get_f_result(theta=theta, theta_synapse=[0]))
    plt.xlabel("$\Delta_{i} = (\\theta - \\theta_{i})$ (deg)")
    plt.ylabel("Firing rate $f_{i}$ (Hz)")
    plt.savefig("analytic_f")
    plt.xlim([-90,90])
    plt.show()

def get_hist_weights():
    plt.hist(weight_profiles, label= "$w_{E}$", alpha = 0.8, color = 'tab:orange')
    plt.hist(w_inh, label="$w_{I}$", alpha=0.8, color='tab:blue')
    plt.xlabel("$w_{X}$")
    plt.legend()
    plt.savefig("weight_distrib")
    plt.show()

def show_tuning_curve():
    bars = 100
    theta_range = np.linspace((-1) * np.pi/2, np.pi/2, bars)

    f_response = get_fs(theta = 0, theta_synapse = theta_range)
    plt.figure()
    plt.plot(theta_range*180/np.pi, f_response)
    plt.xlabel("$\Delta_{i}$ (deg)")
    plt.ylabel("Firing rate $f_{i}$ (Hz)")
    plt.savefig("analytic_f")
    plt.xlim([-90, 90])
    plt.show()

def get_output_bkg_stats(name = "0"):
    fs = []
    isi = []
    for i in range(50):
        print(i)
        spikes_pre = np.random.poisson(lam=f_background * 10e-4 * dt, size=(N_excit_synapses, burn_steps))
        spikes_inh_pre = np.random.poisson(lam=f_inhib * 10e-4 * dt, size=(N_inhib_synapses, burn_steps))
        f,spike_times = evolve_potential_with_inhibition(spikes_pre, spikes_inh_pre, to_plot=True)
        spike_times = np.asarray(spike_times)
        fs.append(f)
        isi.append((spike_times[1:]-spike_times[:-1]) * dt)

        # The None term is added because the arrays in the list
        # have different sizes
        all_isi = np.concatenate(isi, axis=None).ravel()


    plt.hist(fs, weights=np.ones_like(fs) / len(fs),  alpha = 0.8, color="gray")
    plt.xlabel("f (Hz)")
    plt.ylabel("Percent")
    #plt.legend()
    plt.savefig("Output average firing rates trial = {}.svg".format(name))
    plt.figure()

    plt.hist(all_isi, weights=np.ones_like(all_isi) / len(all_isi), alpha=0.8, color="gray")
    plt.xlabel("ISI (ms)")
    plt.ylabel("Percent")
    # plt.legend()
    plt.savefig("Output ISI trial = {}.svg".format(name))
    plt.figure()

    mean = np.mean(all_isi)
    cv = np.sqrt((all_isi - mean) ** 2) / mean
    plt.hist(cv, weights=np.ones_like(cv) / len(cv), color="gray", alpha=0.8)
    plt.xlabel("CV of ISI")
    plt.ylabel("Percent")
    # plt.legend()
    plt.savefig("Output CV of ISI trial = {}.svg".format(name))


def get_input_stats(PO = np.pi/4, trials = 50, show_trials = True, color = cmap(1.0)):
    bars = 21
    bars_range = np.linspace(0, 180, bars)
    fs = np.zeros_like(bars_range)
    stds = np.zeros_like(bars_range)
    f_bar = np.zeros((bars, trials))

    for i in range(bars):
        theta = i * np.pi / (bars-1)
        f_per_neuron = get_fs(theta=theta, theta_synapse= PO*np.ones(trials), f_background=f_background)
        spikes_arrays, f_bar[i] = generate_spikes_array(f_response=f_per_neuron, f_background=f_background)

        if i == 5 or i == 10 or i == 15:
            get_f_isi_and_cv(spikes_arrays, to_plot=True, name = str(i), color = color)
        #else:
         #   f_bar[i] = get_input_f_isi_and_cv(spikes_arrays, to_plot=False)[0]

        fs[i] = np.average(f_bar[i])
        stds[i] = np.std(f_bar[i])

    plt.figure()
    if show_trials == True:
        f_transpose = numpy.transpose(f_bar)
        #for j in range(trials):
         #plt.scatter(bars_range, f_transpose[j], marker='o', alpha=0.3, color = color)
    #plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
    #         color = color, markeredgewidth=1.5, markeredgecolor="black")
    plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color=color)
    #plt.errorbar(bars_range, fs, yerr=stds, fmt='none', color='black')
    plt.fill_between(bars_range, fs - stds, fs + stds, alpha=0.3, color=color)
    plt.xlabel("Stimulus $\\theta$ (deg)")
    plt.ylabel("Presynaptic neuron $f$ (Hz)")
    #plt.legend()
    plt.savefig("Input response stats.svg")
    plt.savefig("Input response stats.png")
    plt.figure()

def get_output_stats_homogeneous(PO = 0, trials = 5, show_trials = True, color = "gray", to_plot = False):

    bars = 11
    bars_range = np.linspace(PO * 180 / np.pi - 90, PO * 180 / np.pi + 90, bars)
    fs = np.zeros_like(bars_range)
    stds = np.zeros_like(bars_range)
    f_bar = np.zeros((bars, trials))

    v_spont, g_e_spont, g_i_spont, tau_spont, nr_spikes_spont, v_series_spont, I_ex_spont, I_in_spont = evolve_potential_with_inhibition(
        spikes_ex=spikes_pre, spikes_in=spikes_inh_pre,
        to_plot=False, only_f=False, parameter_pass=True)
    print("after int = {}".format(v_spont))

    for i in range(bars):
        print("bar {}".format(i))
        theta = PO - np.pi / 2 + i * np.pi / (bars)
        fs_ex = get_fs(theta=theta, theta_synapse=PO * np.ones(N_excit_synapses), f_background=f_background)
        fs_in = get_fs(theta=theta, theta_synapse=PO * np.ones(N_inhib_synapses), f_background=f_background)

#from here
        all_spikes = []
        for j in range(trials):
            print("trial {}".format(j))

            if True:
                spikes_ex, fs_res_ex = generate_spikes_array(f_response=fs_ex, f_background=f_background)
                spikes_in, fs_res_in = generate_spikes_array(f_response=fs_in, f_background=f_background)


                if (j == 0) and (i == 0 or i == 3 or i == 5) and (to_plot == True):
                    f_bar[i, j], spike_times = evolve_potential_with_inhibition(
                        spikes_ex, spikes_in,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=False, to_plot=True,
                        parameter_pass=False)

                else:
                    f_bar[i, j], spike_times = evolve_potential_with_inhibition(
                        spikes_ex, spikes_in,
                        v=v_spont, g_e=g_e_spont, g_i=g_i_spont, tau=tau_ref, nr_spikes=0, v_series=[], I_ex=[],
                        I_in=[],
                        name_i="i for bar: {}".format(i), name_V="V for bar: {}".format(i), only_f=True, to_plot=False,
                        parameter_pass=False)
                all_spikes.append(spike_times)
#until here
                print("f_out = {}".format(f_bar[i, j]))

        plt.figure()
        fs[i] = np.average(f_bar[i])
        stds[i] = np.std(f_bar[i])


    if show_trials == True:
        f_transpose = numpy.transpose(f_bar)
        for j in range(trials):
            plt.scatter(bars_range, f_transpose[j], marker='o', alpha=0.3, color=color)

        #if i == 0 or i == 3 or i == 5:
        #    get_f_isi_and_cv(spikes_arrays, to_plot=True, name=str(i), color=color)
        # else:
        #   f_bar[i] = get_input_f_isi_and_cv(spikes_arrays, to_plot=False)[0]


    plt.plot(bars_range, fs, marker='o', alpha=0.9, linewidth="2", markersize="20", label="$\langle f \\rangle$",
             color=color, markeredgewidth=1.5, markeredgecolor="black")
    plt.errorbar(bars_range, fs, yerr=stds, fmt='none', color='black')
    plt.xlabel("Stimulus $\\theta$")
    plt.ylabel("Response of the PS neuron $f$ (Hz)")
    plt.legend()
    plt.savefig("Output response stats")
    plt.figure()


def unpack_spike_times():
    list = [0, 1, 6]
    x = np.arange(10,20)
    print(x)
    print(x[list])
    x[list] = 1
    print(x)
    print(x[list])


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
