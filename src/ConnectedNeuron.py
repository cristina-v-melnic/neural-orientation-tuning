import numpy as np
from parameters import *
from tuning_distributions import *
from scipy.stats import truncnorm

class ConnectedNeuron:
    '''
        Class containing the stimulus-independent properties of the single neuron and
         its afferents, such as nr of afferents, their synaptic weights, and the individual
         tuning of the afferents (distribution of PO per afferent).
    '''

    def __init__(self, connectivity_type, PO = np.pi/4, PO_distrib = "continuous", to_plot = False):
        '''
        :param connectivity_type: ["weight", "number"] - based.
        :param PO: the desired preferred orientation of the postsynaptic neuron (rad).
        :param PO_distrib: ["binary", "continuous"] distribution of afferent POs.
        '''

        # Nr afferents.
        self.N_e = N_excit_synapses
        self.N_i = N_inhib_synapses

        # Weights properties.
        # Mean and std of lognormal distribution.
        self.w_mu = mu
        self.w_sigma = sigma
        # Bounds of the lognormal distribution.
        self.lower = lower
        self.upper = upper
        # Initialisation of the weights arrays.
        self.W_e = np.zeros(self.N_e)
        self.W_i = np.zeros(self.N_i)

        # Connectivity type upon which weight and PO distributions depend.
        self.connectivity_type = connectivity_type # ["weight", "number"]
        self.PO = PO # desired PO of the postsynaptic neuron (rad)
        self.PO_distrib = PO_distrib # ["binary", "continuous"]
        self.PO_e = np.zeros(self.N_e)
        self.PO_i = np.zeros(self.N_i)

        # Visualisation
        self.to_plot = to_plot

        self.generate_weights()
        self.tune_afferents()

    def generate_weights(self):
        '''
                Lognormal sampling of synaptic weights for each neuron
                from both excitatory and inhibitory populations.
        '''

        X = truncnorm((self.lower - self.w_mu) / self.w_sigma, (self.upper - self.w_mu) / self.w_sigma, loc=self.w_mu, scale=self.w_sigma)

        self.W_e = np.log(X.rvs(N_excit_synapses))
        self.W_i = np.log(X.rvs(N_inhib_synapses))

        if self.connectivity_type == "weight":
            # Sort the excitatory array of weights from large to small.
            self.W_e = np.flip(np.sort(self.W_e))

    def tune_afferents(self):
        '''
                Set a preffered orientation to each afferent for
                both E and I populations depending on the kind of PO
                distribution per neuron ["binary", "continuous"]
                and the kind of connectivity type ["weight", "number"].
        '''
        if self.PO_distrib == "binary":

            if self.connectivity_type == "number":
                self.PO_e, self.PO_i = tune_binary(N_excit_synapses = self.N_e,
                                                   N_inhib_synapses = self.N_i,
                                                   to_plot=self.to_plot)
            elif self.connectivity_type == "weight":
                self.PO_e, self.PO_i = tune_binary_synaptic(N_excit_synapses = self.N_e,
                                                            N_inhib_synapses = self.N_i,
                                                            to_plot=self.to_plot)
        elif self.PO_distrib == "continuous":

            if self.connectivity_type == "number":
                self.PO_e, self.PO_i = cont_tuning(N_excit_synapses = self.N_e,
                                                   N_inhib_synapses = self.N_i,
                                                   to_plot=self.to_plot)
            elif self.connectivity_type == "weight":
                self.PO_e, self.PO_i = tune_cont_synaptic(N_excit_synapses = self.N_e,
                                                          N_inhib_synapses = self.N_i,
                                                          weight_profiles = self.W_e,
                                                          to_plot=self.to_plot)


