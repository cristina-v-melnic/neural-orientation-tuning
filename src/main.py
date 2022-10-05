from robustness_experiment import *
from ConnectedNeuron import *
from SpikyLIF import *
from TunedResponse import *


if __name__ == '__main__':
    neuron1 = ConnectedNeuron(connectivity_type="number", PO_distrib="binary")
    #neuron2 = ConnectedNeuron(connectivity_type="weight")
    #print(neuron1.W_e)
    #print(neuron2.W_e)


    #LIF1 = SpikyLIF(neuron1, to_plot=True, show=True)
    #print(LIF1.f_post)


    Response_1_bar = TunedResponse(neuron1, trials=1, nr_bars=1, theta = [np.pi/4])
    Response_1_bar.plot_figs()

    #get_response_for_bar(trials=5, to_plot=True, mean = np.pi/4)
    #check_robustness()
    #get_input_response_for_bar()
    #get_input_stats()
    #get_output_bkg_stats()

