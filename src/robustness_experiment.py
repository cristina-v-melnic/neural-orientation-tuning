import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from stimulus_sweep import *
from plotting_setup import *

def check_robustness():
    '''

    :return: Plot of mean-squared-error (MSE) versus number of deleted synapses.
    '''
    # How many neurons to cut.
    nr_to_cut = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    #nr_to_cut = np.arange(0, 36, 2)

    # The type of connectivity.
    types = ["syn", "struct"]

    # Where to save the resulting files.
    directory = '../data/robustness/'

    # Run simulations to get tuning curves and save them in files.
    run_check_robustness(nr_to_cut, types, directory)

    # Import the tuning curves from files.
    f_syn, f_struct = get_data_robustness(nr_to_cut, types, directory)

    # Find the MSE in both cases.
    syn_err = get_mse(f_syn[1:], f_syn[0])
    struct_err = get_mse(f_syn[1:], f_struct[0])

    #plt.figure()
    plt.scatter(nr_to_cut[1:], syn_err, label = "weight-based")
    plt.scatter(nr_to_cut[1:], struct_err, label = "number-based")
    plt.xlabel("# deleted synapses")
    plt.ylabel("MSE$(f)$")
    plt.legend()
    plt.savefig("robustness.svg")
    plt.savefig("robustness.png")
    #plt.show()



def run_check_robustness(nr_to_cut, types, directory):
    '''
    :return: Files with tuning curves in both scenarios with
             different amounts of similar afferents being cut.
    '''

    # Get the tuning curves for all scenarios and save them in separate files.
    for type in types:
        for nr in nr_to_cut:
            name_file = "fs_" + type + "_" + str(nr) + '.pickle'
            if os.path.exists(directory + name_file ) == False:
                fs_syn = get_response_for_bar(to_plot=False, syn=True, binary=True, cut_nr_neurons=nr)[1]
                with open(directory + name_file, 'wb') as fout:
                    pickle.dump(fs_syn, fout)
            else:
                print("Yeey! {} datafile already exists!".format(name_file))


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