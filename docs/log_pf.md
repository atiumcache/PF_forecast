---
title: Log Particle Filter
---
<script src="./assets/mathjax_settings.js" async></script>

# Log-Domain Particle Filter

A particle filter represent the PDF of some state vector $X_t$ at time $t$. In our case, the state vector is:

$$X_t = [S, I, R, H, \text{new } H]$$

## ParticleCloud Class
We represent the particles in a `ParticleCloud` class. A `ParticleCloud` has the following attributes:
- settings: Global settings for the filter, an instance of `InitSettings`. See below for more info.
- states:
- weights:
- betas:

### Parameters
#### Filter Parameters

```python
@dataclass
class InitSettings:
    num_particles: int
    population: int
    location_code: str
    dt: float = field(default_factory=lambda: 1.0)
    beta_prior: Tuple[float, float] = field(default_factory=lambda: (0.10, 0.15))
    seed_size: float = field(default_factory=lambda: 0.005)
```
The `population` is the population of the location/territory that we are currently working with. A `location_code` corresponds to the location codes found in `/datasets/locations.csv`. 

The `dt` attribute determines the number of intermediary steps that the ODE/SDE system will be solved. The default is 1, which solves the system exactly one time per day. A lower value for `dt` will increase the numerical accuracy, while also increasing the computational cost of updating a particle. 

The `beta_prior` is our prior belief about an initial beta value. Each particle's \beta value is a uniform draw from this range. 

The `seed_size` determines the initial proportion of infected individuals in the population. 

#### Model Parameters
A `ModelParameters` class is passed into `Transition` object, which defines the SDE system for our particles. 

All parameters have default values. Interpretations of the parameters are below.

```python
@dataclass
class ModelParameters:
    gamma: float = field(default_factory=lambda: 0.06)
    mu: float = field(default_factory=lambda: 0.004)
    q: float = field(default_factory=lambda: 0.1)
    eta: float = field(default_factory=lambda: 0.1)
    std: float = field(default_factory=lambda: 10.0)
    R: float = field(default_factory=lambda: 50.0)
    hosp: int = field(default_factory=lambda: 10)
    L: int = field(default_factory=lambda: 90)
    D: int = field(default_factory=lambda: 7)
```

- gamma: The recovery rate, representing the fraction of infected individuals who recover per unit time.                                         
- mu: The mortality rate, indicating the fraction of infected individuals who die per unit time.                                                 
- q: The hospitalization rate, representing the fraction of infected individuals who require hospitalization per unit time.                      
- eta: The rate at which hospitalized individuals recover and are discharged from the hospital.                                                  
- std: The standard deviation of the process noise, which accounts for the variability in the model's predictions.                               
- R:                                                                                                                   
- hosp: The initial number of hospitalized individuals at the start of the simulation.                                                           
- L: The latency period, representing the average time between exposure to the virus and the onset of infectiousness.                            
- D: The duration of the infectious period, indicating the average time an individual remains infectious.                                        

### Update / Propagate
At each time step, we propagate the particles forward one time step based on our state transition function. 

The update is split into two functions to allow us to compute the gradient at each time step for sensitivity analysis. 

#### Update Single Particle
The `update_single_particle` method takes in a particle's current state, and returns a new, updated state. 

The method calls out to our transition model's SDE system, which is broken up into deterministic and stochastic components. If `dt = 1`, then this process occurs once. Otherwise, finer granularity can be achieved by decreasing `dt`, but this comes with computational cost. 

#### Update All Particles
The `update_all_particles` method calls out to `update_single_particle` for each particle. We use `jax.vmap` to map the function to a collection of iterable arguments: the current state vectors and beta values (t and dt remain constant for all particles). This is a vectorized function mapping.  


### Calculating Weights
We calculate the weight

### Normalizing Weights
We use a Jacobian algorithm to normalize weights. We will denote the log-normalization factor at time k as $W'_k$.
For contrast, we denote the linear-normalization factors as $W_k$. 

The normalization factor is the sum of all weights. In the linear domain, we just take the sum of all weights:

$$W_k = \sum_{i=1}^N w_k^i$$

So, we could calculate the log-normalization factor as follows:

$$W'_k = \ln(\sum_{i=1}^N e^{w'_k^{i}})$$

However, this requires a move from log to linear domain, and then back to log again. This could lead to numerical error. 

Thus, we utilize the Jacobian algorithm defined in [this paper](https://www.researchgate.net/publication/323521063_Log-PF_Particle_Filtering_in_Logarithm_Domain). 

The Jacobian defines the log of the sum of $n$ exponentials as follows:

$$\ln(e^{w'_1} + ... + e^{w'_n}) = \max(ln(\Delta), w'_n) + \ln(1 + e^{-|\ln(\Delta) - w'_n|)$$

Where $\Delta = e^{w'_1} + ... + e^{w'_{n-1}}$. 

The implemented algorithm is as follows:

```python
def jacobian(input_array: ArrayLike) -> Array:
    """
    The Jacobian algorithm, used in log likelihood normalization and
    resampling processes.

    Args:
        input_array: An array of values to sum.

    Returns:
        The vector of partial sums of the input array.
    """
    n = len(input_array)
    delta = jnp.zeros(n)
    delta = delta.at[0].set(input_array[0])
    for i in range(1, n):
        delta_i = max(input_array[i], delta[i - 1]) + jnp.log(
            1 + jnp.exp(-1 * jnp.abs(input_array[i] - delta[i - 1]))
        )
        delta = delta.at[i].set(delta_i)
    return delta
```

We see that the Jacobian function is an iterative process to build up the array of partial sums. 
Eventually, we get the full sum with $\Delta_n$, where $W'_k = \Delta_n$. 

### Resampling


### Perturbations
