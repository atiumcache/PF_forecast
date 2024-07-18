from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp, random as random, Array
from jax._src.basearray import ArrayLike
from jax.scipy.stats import norm as norm

from filter_forecast.particle_filter.init_settings import InitSettings
from filter_forecast.particle_filter.transition import Transition


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
            T is number of time steps.
        betas: An (N, T) array of beta values, where
            N is number of particles,
            T is length of time series.
        hosp_estimates: An array of hospital estimates. Length = num_particles.

    Examples:
        weights[i, t] will return the ith particle's weight at time t.
        betas[:, t] will return all particles beta values at time t.
    """

    def __init__(self, settings: InitSettings, transition: Transition):
        self.settings = settings
        self.model = transition

        seed = 43
        self.key = random.PRNGKey(seed)

        initial_states = jnp.array(
            [self._get_initial_state() for _ in range(self.settings.num_particles)]
        )
        # Expand dimensions to accommodate the runtime
        self.states = jnp.zeros(
            (
                self.settings.num_particles,
                initial_states.shape[-1],
                self.settings.runtime,
            )
        )
        print(initial_states[0])
        # Initialize the first time step with initial states
        self.states = self.states.at[:, :, 0].set(initial_states)

        self.weights = jnp.zeros((self.settings.num_particles, self.settings.runtime))

        # Generate betas for each particle and each time step
        beta_prior = self.settings.beta_prior
        betas_initial = np.random.uniform(beta_prior[0], beta_prior[1],
                                          size=self.settings.num_particles)
        print(betas_initial)
        betas = jnp.tile(betas_initial[:, None], (1, self.settings.runtime))
        # Assign the betas array to self.betas
        self.betas = betas

        self.hosp_estimates = jnp.zeros(self.settings.num_particles)

    def _get_initial_state(self):
        """Gets an initial state for one particle.

        The entire population is susceptible. Then, we draw from uniform
        random to infect some portion of the susceptible population."""
        population = self.settings.population

        # state = [S, I, R, H, new_H]
        state = [population, 0, 0, 0, 0]
        infected_seed = np.random.uniform(0, self.settings.seed_size * population)
        # Move
        state[1] += infected_seed
        state[0] -= infected_seed
        return state

    def _update_single_particle(
        self, state: ArrayLike, t: int, beta: float, dt: float
    ) -> Array:
        """For a single particle, step the state forward 1 time step.

        Helper function for update_all_particles. Each particle's update is
        performed separately to accommodate for gradient/sensitivity
        calculations in the future.

        Args:
             state: the current state of the particle at time t.
             t: the current time step
             beta: the current beta value for the particle
             dt: granularity for numerical integration

        Returns:
            New state vector for a single particle.
        """
        num_steps = int(1 / dt)
        for _ in range(num_steps):
            state += self.model.det_component(state, t, beta) * dt
            state += self.model.sto_component(state, dt, self.key)
        return state

    def update_all_particles(self, t: int) -> Array:
        """Propagate all particle state vectors forward one time step.

        Args:
            t: current time step

        Returns:
            MxN array of states for each particle at time t, where M is
            dimension of state vector, N is number of particles.
        """
        # Map each particle's state and beta to the update_single_particle function.
        # We iterate over the 0th axis of states and betas.
        # Thus, we get the state vector for each particle at time t.
        # And we get the beta value for each particle at time t.
        new_states = jax.vmap(self._update_single_particle, in_axes=(0, None, 0, None))(
            self.states[:, :, t], t, self.betas[:, t], self.settings.dt
        )
        return new_states

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
        weight = norm.logpdf(reported_data, particle_estimate, self.model.params.R)
        return float(weight)

    def compute_all_weights(self, reported_data: int | float) -> Array:
        """Update the weights for every particle.

        Args:
            reported_data: Reported new hospitalization case counts at
                current time step.

        Returns:
            NDArray of new weights, where N is number of particles.
        """
        new_weights = jnp.zeros(self.settings.num_particles)
        for i in range(self.settings.num_particles):
            hosp_estimate = self.hosp_estimates[i]
            new_weight = self._compute_single_weight(reported_data, hosp_estimate)
            new_weights = new_weights.at[i].set(new_weight)
        return new_weights

    def normalize_weights(self):
        """Normalize the weights using the Jacobian algorithm."""
        self.weights = log_norm(self.weights)

    def resample(self):
        """Systematic resampling algorithm."""
        resampling_indices = jnp.zeros(self.settings.num_particles, dtype=int)
        cdf_log = jacobian(self.weights)

        u = np.random.uniform(0, 1 / self.settings.num_particles)

        i = 0
        for j in range(self.settings.num_particles):
            r_log = np.log(u + (1 / self.settings.num_particles) * j)
            while r_log > cdf_log[i]:
                i += 1
            resampling_indices = resampling_indices.at[j].set(i)

        self.states = self.states[resampling_indices]
        self.betas = self.betas[resampling_indices]
        self.weights = jnp.zeros(self.settings.num_particles)

    def perturb_betas(self, scale_factor: float = 0.001):
        """Perturbs the beta values by adding gaussian noise."""
        noise: Array = random.normal(self.key, shape=self.betas.shape) * scale_factor
        self.betas += noise


def jacobian(little_delta: ArrayLike) -> Array:
    """
    The Jacobian algorithm, used in log likelihood normalization and
    resampling processes.

    Args:
        little_delta: An array of values to sum

    Returns:
        The vector of partial sums of little_delta.
    """
    n = len(little_delta)
    big_delta = jnp.zeros(n)
    big_delta = big_delta.at[0].set(little_delta[0])
    for i in range(1, n):
        big_delta_i = max(little_delta[i], big_delta[i - 1]) + jnp.log(
            1 + jnp.exp(-1 * jnp.abs(little_delta[i] - big_delta[i - 1]))
        )
        big_delta = big_delta.at[i].set(big_delta_i)
    return big_delta


def log_norm(log_weights: ArrayLike) -> Array:
    """
    Normalizes the probability space using the Jacobian algorithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
