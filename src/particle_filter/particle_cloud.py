from dataclasses import dataclass

import jax
import numpy as np
from jax import Array, float0
from jax import numpy as jnp
from jax import random as random
from jax.scipy.stats import nbinom as nbinom
from jax.scipy.stats import norm as normal
from jax.typing import ArrayLike

from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.transition import Transition


KeyArray = Array


@dataclass
class ParticleCloud:
    """Represents a cloud of particles. Includes methods for updating
    particles, computing weights, resampling, and perturbing variables.

    Attributes:
        settings: Global filter settings for initialization.
        states: An (N, S, T) array of system states at each time step, where
            N is number of particles,
            S is size of state vector,
            T is length of time series.
        weights: An (N, T) of particle weights, where
            N is number of particles,
            T is length of time series.
        hosp_estimates: An array of hospital estimates. Length = num_particles.

    Examples:
        weights[i, t] will return the ith particle's weight at time t.
        states[:, 4, t] will return all particles' new_H counts at time t.
    """

    def __init__(self, settings: GlobalSettings, transition: Transition):
        self.settings = settings
        self.model = transition

        seed = 43
        self.key = random.PRNGKey(seed)

        self.key, *initial_state_keys = random.split(
            self.key, self.settings.num_particles + 1
        )
        initial_states = jnp.array(
            [self._get_initial_state(k) for k in initial_state_keys]
        )

        self.states = jnp.zeros(
            (
                self.settings.num_particles,
                initial_states.shape[-1],
                self.settings.runtime,
            )
        )

        self.states = self.states.at[:, :, 0].set(initial_states)
        self.weights = jnp.zeros((self.settings.num_particles, self.settings.runtime))
        self.hosp_estimates = jnp.zeros(self.settings.num_particles)
        self.all_resamples = jnp.zeros(
            (
                self.settings.num_particles,
                self.settings.num_particles,
                self.settings.runtime,
            )
        )

    def _get_initial_state(self, key: KeyArray) -> Array:
        """Gets an initial state for one particle.

        The entire population is susceptible. Then, we draw from uniform
        random to infect some portion of the susceptible population.

        Args:
            key: A JAX PRNG key

        Returns:
            Initial state vector.
        """
        key1, key2 = random.split(key, 2)
        population = self.settings.population

        # state = [S, I, R, H, new_H, beta, ll_var]
        state = [population, 0, 0, 0, 0, 0]

        # Infect a portion of S compartment
        infected_seed = random.uniform(
            key=key1, minval=1, maxval=self.settings.seed_size * population
        )
        state[1] += infected_seed
        state[0] -= infected_seed

        # Initialize beta based on prior
        beta_prior = self.settings.beta_prior
        initial_beta = random.uniform(
            key=key2, shape=(), dtype=float, minval=beta_prior[0], maxval=beta_prior[1]
        )
        state[5] = initial_beta

        return jnp.array(state)

    def _update_single_particle(self, state: ArrayLike, t: int) -> Array:
        """For a single particle, step the state forward 1 discrete time step.

        Helper function for update_all_particles. Each particle's update is
        performed separately to accommodate for individual gradient/sensitivity
        calculations.

        Args:
            state: the current state of the particle at time t.
            t: the current time step.

        Returns:
            New state vector for a single particle.
        """
        num_steps = int(1 / self.settings.dt)
        for _ in range(num_steps):
            det_update = self.model.det_component(state, t) * self.settings.dt
            self.key, subkey = random.split(self.key)
            sto_update = self.model.sto_component(state, self.settings.dt, subkey)
            state += det_update + sto_update
        return state

    def update_all_particles(self, t: int) -> None:
        """Propagate all particle state vectors forward one time step.

        Args:
            t: current time step

        Returns:
            None. We update the instance states directly.
        """
        # Map each particle's previous state to the update_single_particle function.
        # We iterate over the 0th axes of states.
        # Thus, we pass our function the state vector for each particle at t - 1.
        new_states = jax.vmap(self._update_single_particle, in_axes=(0, None))(
            self.states[:, :, t - 1], t
        )

        self.states = self.states.at[:, :, t].set(new_states)

        new_hosp_estimates = self.states[:, 4, t] - self.states[:, 4, t]
        self.hosp_estimates = new_hosp_estimates

    def _compute_single_weight(
        self, reported_data: int, particle_estimate: float | int
    ) -> float:
        """Computes the un-normalized weight of a single particle.
        Helper function for compute_all_weights.

        Args:
            reported_data: Reported new hospitalization case counts at
                current time step.

        Returns:
            An un-normalized weight for a single particle.
        """
        particle_estimate = round(particle_estimate)

        weight = normal.logpdf(x=reported_data, loc=particle_estimate, scale=100)
        """ 
        weight = nbinom.logpmf(
            k=reported_data,
            loc=particle_estimate,
            n=self.settings.likelihood_r,
            p=self.settings.likelihood_p,
        ) """
        return float(weight)

    def compute_all_weights_vmap(self, reported_data: int | float, t: int) -> None:
        new_weights = jax.vmap(self._compute_single_weight, in_axes=(None, 0, None))(
            reported_data, self.hosp_estimates, t
        )
        self.weights = self.weights.at[:, t].set(new_weights)

    def compute_all_weights(self, reported_data: int | float, t: int) -> None:
        """Update the weights for every particle.

        Args:
            reported_data: Reported new hospitalization case counts at
                current time step.
            t: current time step.

        Returns:
            None. Updates the instance weights directly.
        """
        new_weights = jnp.zeros(self.settings.num_particles)

        for p in range(self.settings.num_particles):
            hosp_estimate = self.hosp_estimates[p]
            new_weight = self._compute_single_weight(
                reported_data, float(hosp_estimate)
            )
            new_weights = new_weights.at[p].set(new_weight)
        self.weights = self.weights.at[:, t].set(new_weights)

    def normalize_weights(self, t: int) -> None:
        """Normalize the weights using the Jacobian algorithm.
        Updates the instance weights directly.

        Args:
            t: current time step

        Returns:
            None. Directly updates the instance weights.
        """
        norm_weights = log_norm(self.weights[:, t])
        self.weights = self.weights.at[:, t].set(norm_weights)

    def resample(self, t: int) -> None:
        """Systematic resampling algorithm.

        Args:
            t: current time step

        Returns:
            None. Directly updates the instance states and beta values.
        """
        resampling_indices = jnp.zeros(self.settings.num_particles, dtype=int)
        cdf_log = jacobian(self.weights[:, t])

        u = np.random.uniform(0, 1 / self.settings.num_particles)

        i = 0
        for j in range(self.settings.num_particles):
            r_log = jnp.log(u + (1 / self.settings.num_particles) * j)

            # Ensure that we do not go out of bounds
            while i < len(cdf_log) - 1 and r_log > cdf_log[i]:
                i += 1
            resampling_indices = resampling_indices.at[j].set(i)

        self.all_resamples = self.all_resamples.at[:, :, t].set(resampling_indices)
        self.states = self.states.at[:, :, t].set(self.states[resampling_indices, :, t])

    def compute_marginal_likelihood(self):
        """Returns the marginal likelihood, to be used by MCMC."""
        sums = jacobian(self.weights[:, -1])
        return sums[-1]


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


def log_norm(log_weights: ArrayLike) -> Array:
    """
    Normalizes the probability space using the Jacobian algorithm as
    defined in jacobian().

    The Jacobian outputs an array of partial sums, where the
    last element is the sum of all inputs. Thus, the normalization
    factor is this last element.
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
