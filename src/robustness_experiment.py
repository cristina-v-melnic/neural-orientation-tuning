import os
import pickle
from stimulus_sweep import *
from plotting_setup import *

def check_robustness(plot_directory = "../plots/robustness/"):
    '''

    :return: Plot of mean-squared-error (MSE) versus number of deleted synapses.
    '''
    # How many neurons to cut.
    #nr_to_cut = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    #nr_to_cut = np.arange(0, 36, 2)
    nr_to_cut = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    # The type of connectivity.
    types = ["syn", "struct"]

    # Where to save the resulting files.
    data_directory = '../data/robustness/'

    # Run simulations to get tuning curves and save them in files.
    run_check_robustness(nr_to_cut, types, data_directory)

    # Import the tuning curves from files.
    f_syn, f_struct = get_data_robustness(nr_to_cut, types, data_directory)

    # Find the MSE in both cases.
    syn_err = get_mse(f_syn[1:, 3:10], f_syn[0, 3:10])
    struct_err = get_mse(f_syn[1:, 3:10], f_struct[0, 3:10])


    plt.scatter(100/200*np.asarray(nr_to_cut[1:]), syn_err, label = "weight-based")
    plt.scatter(100/200*np.asarray(nr_to_cut[1:]), struct_err, label = "number-based")
    plt.xlabel("% deleted synapses")
    plt.ylabel("MSE$(f)$")
    plt.legend()
    if svg_enable == True: plt.savefig(plot_directory + "robustness.svg")
    plt.savefig(plot_directory + "robustness.png")
    plt.figure()


def run_check_robustness(nr_to_cut, types, directory, nr_trials = 5):
    '''
    :return: Files with tuning curves in both scenarios with
             different amounts of similar afferents being cut.
    '''

    # Get the tuning curves for all scenarios and save them in separate files.
    for type in types:
        for nr in nr_to_cut:
            name_file = "fs_" + type + "_" + str(nr) + '.pickle'

            if os.path.exists(directory + name_file) == False:
                if nr == 0 or nr == 60 or nr == 80 or nr == 100:
                    fs_syn = get_response_for_bar(trials = nr_trials, to_plot = True, syn = True, binary = True,
                                                  cut_nr_neurons = nr, name_file = str(type)+"_"+str(nr))[1]
                else:
                    fs_syn = get_response_for_bar(trials = nr_trials, to_plot = False, syn = True, binary = True,
                                                  cut_nr_neurons = nr, name_file = str(type) + "_" + str(nr))[1]
                with open(directory + name_file, 'wb') as fout:
                    pickle.dump(fs_syn, fout)
            else:
                print("Yeey! {} datafile already exists!".format(name_file))


def run_if_nonexistent(parameter_to_save, directory, nr_trials = 5):
    return 0

def get_data_robustness(nr_to_cut, types, directory):
    '''

    :return: Load existing tuning curves into 2 arrays.
    '''

    f_syn = np.zeros((len(nr_to_cut), 21))
    f_struct = np.zeros((len(nr_to_cut), 21))

    for i in range(len(types)):
        for j in range(len(nr_to_cut)):
            name_file = "fs_" + types[i] + "_" + str(nr_to_cut[j]) + '.pickle'
            with open(directory + name_file, 'rb') as fin:
                if i == 0:
                    f_syn[j] = pickle.load(fin)
                else:
                    f_struct[j] = pickle.load(fin)
    return f_syn, f_struct


def get_mse(x, x_bar):
    '''
    :return: Mean squared error.
    '''
    return np.mean(np.square(np.subtract(x, x_bar)), axis=1)