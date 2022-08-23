import numpy as np
from scipy.stats import truncnorm

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

# Inhibition parameters
tau_i = 10 # ms
E_inh = -90 #  mV
g_0_i = 0.0

# Input parameters
N_synapses = 250
N_inhib_synapses = int(N_synapses/5)
N_excit_synapses = N_synapses - N_inhib_synapses

# The above specific mean and sigma values are chosen s.t. a reasonable background
# firing rate of the output neuron is obtained. (more details search_bg_weights())
f_background = 5 # Hz
f_max = 10 # Hz
f_inhib = 5 # Hz


# Integration parameters
dt = 0.1 # ms
nr_seconds = 20
N_steps = int(nr_seconds * 1000 / dt) # integration time steps
t_start = 0 # ms
t_end = nr_seconds * 1000 # ms
#dt = (t_end - t_start)/N_steps # time step width
time_range = np.linspace(t_start, t_end, N_steps)
save_interval = 10 # for dt = 0.1 save every 100 ms
save_zoom_interval = 500 #
tau_ref = int(50/dt)

# Spontaneuos firing mode parameters.
burn_seconds = int(nr_seconds/2)
burn_steps = int(burn_seconds * 1000 / dt) # steps or 20 seconds or 20000 ms
stimulus_seconds = 10
stimulus_time_steps = int(stimulus_seconds * 1000 / dt)
silence_seconds = nr_seconds - burn_seconds - stimulus_seconds
silence_time_steps = int(silence_seconds * 1000 / dt)

# Spike trains pre-stimulus and post-stimulus.
spikes_pre = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, burn_steps))
spikes_post = np.random.poisson(lam = f_background * 10e-4 * dt, size = (N_excit_synapses, silence_time_steps))
spikes_inh_pre = np.random.poisson(lam = f_inhib * 10e-4 * dt, size =  (N_inhib_synapses, burn_steps))
spikes_inh_post = np.random.poisson(lam = f_inhib * 10e-4 * dt, size =  (N_inhib_synapses, silence_time_steps))

# Lognormal weight distribution setup.
mu = 0.29
sigma = np.sqrt(mu)
lower, upper = 1.2, 1.5

normal_lower = np.log(lower)
normal_upper = np.log(upper)
normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
normal_mean = np.log(mu) - normal_std**2 / 2

X = truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc = mu, scale = sigma)


weight_profiles = np.log(X.rvs(N_excit_synapses))
w_inh = np.log(X.rvs(N_inhib_synapses))

# Other distributions I've tried.

# Uniform distrib for weights has worked.
#mu = 0.25
#weight_profiles = np.ones(N_excit_synapses) * mu
#w_inh = np.ones(N_inhib_synapses) * mu

# Gaussian distrib for weights has worked as well.
#mu = 0.07
#sigma = np.sqrt(mu)
#weight_profiles = np.random.normal( loc = mu, scale = sigma, size = N_excit_synapses)
#w_inh = np.random.normal(loc = mu, scale = sigma, size = N_inhib_synapses)