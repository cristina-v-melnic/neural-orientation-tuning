# neural-orientation-tuning
 Code for "Orientation selectivity can arise from distinct connectivity patterns" project.

## Application description
Model of a single orientation-selective neuron receiving inputs from afferents with
different preferred orientations. The project shows what properties the postsynaptic neuron has
and how they depend on the input. It aims to investigate two types of connectivity of neurons
in the primary visual cortex, i.e., weight-based and number-based.


### Code structure:
parameters.py -> plotting_setup.py -> plots.py -> postsynaptic_train.py -> utils_stimulus_sweep.py 
-> stimulus_sweep.py -> robustness_experiment.py -> main.py 

#### To get plots of a tuning curve:
`get_response_for_bar(trials=5, to_plot=True, mean = np.pi/4)`

#### To see impact of connectivity on robustness:
`check_robustness()`


## More about the project 
### "Orientation selectivity can arise from distinct connectivity patterns"

#### Authors:
Cristina Melnic, Douglas Feitosa Tomé, Tim P Vogels

#### Abstract:
Neurons in the primary visual cortex (V1) are known to selectively respond to stimuli of spatial orientations, whereby groups of neurons are tuned to distinct preferred orientation angles. It was established that neurons with similar preferred orientation have a higher probability of being connected, which was interpreted in classical theories as larger synaptic weights. A recent experimental study of single neurons with all their afferents [1], however, found the individual synaptic strength of afferents to be independent of the similarity to the postsynaptic preferred orientation. 

Here, we theoretically investigate a neuron model with orientation selective afferents in the two scenarios of correlated and independent individual synaptic strength with respect to preferred orientation.  We observe that in both cases the shape of receptive fields arises from the cumulative synaptic weight of the active afferents, independently of the specific connectivity. The model is thus consistent with findings of higher probability between neurons of similar functionality and offers a mathematical framework for further investigations of factors that give rise to orientation selectivity.

[1] Scholl et al. 2021, “Cortical response selectivity derives from strength in numbers of synapses”, Nature

#### Credits
This project was realised during a summer internship (June - August 2022) in the Vogels lab at IST Austria, funded by OeAD and ISTA
as a part of the "ISTernship" program. It was closely supervised by Dr. Douglas Feitosa Tomé and Prof. Dr. Tim P Vogels and other members of the "Computational Neuroscience and Neurotheory" group at ISTA.
