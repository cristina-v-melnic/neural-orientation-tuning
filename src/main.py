from robustness_experiment import *
from ConnectedNeuron import *
from SpikyLIF import *


if __name__ == '__main__':
    neuron1 = ConnectedNeuron(connectivity_type="number")
    neuron2 = ConnectedNeuron(connectivity_type="weight")
    print(neuron1.W_e)
    print(neuron2.W_e)


    LIF1 = SpikyLIF(neuron1, to_plot=True, show=True)
    print(LIF1.f_post)


    #get_response_for_bar(trials=5, to_plot=True, mean = np.pi/4)
    #check_robustness()
    #get_input_response_for_bar()
    #get_input_stats()
    #get_output_bkg_stats()

