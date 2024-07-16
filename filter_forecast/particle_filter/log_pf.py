from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.stats.norm as norm
from jax.typing import ArrayLike

from filter_forecast.particle_filter.output_handler import OutputHandler
from filter_forecast.particle_filter.parameters import ModelParameters
from filter_forecast.particle_filter.setup_pf import get_logger
from filter_forecast.particle_filter.init_settings import InitSettings


@dataclass
class ObservationData:
    """Stores the observed/reported data (Hospitalization case counts)."""
    observations: ArrayLike

    def get_observation(self, t: int) -> int:
        """Returns the observation at time t.

        An observation is the new hospitalizations case count
        on day t.
        """
        return self.observations[t]


@dataclass
class ParticleCloud:
    """Represents a cloud of particles.

    Attributes:
        settings: Global filter settings for initialization.
        params: Model parameters for the transition function.
        states: A NxT array of system states at each time step, where
            N is size of state vector, T is length of time series.
        weights: A 1-D array of particle weights. Length is number of particles.
        betas: An NxT array of beta values. N is number of particles, T is length of time series.
    """

    settings: InitSettings
    params: ModelParameters = field(init=False)
    states: ArrayLike = field(init=False)
    weights: ArrayLike = field(init=False)
    betas: ArrayLike = field(init=False)
    hosp_estimates: ArrayLike = field(init=False)

    def __post_init__(self):
        self.params = ModelParameters()
        self.states = jnp.array(
            [self.get_initial_state() for _ in range(self.settings.num_particles)]
        )

        self.weights = jnp.zeros(self.settings.num_particles)

        betas = [
            np.random.uniform(self.settings.beta_prior[0], self.settings.beta_prior[1])
            for _ in range(self.settings.num_particles)
        ]

        self.betas = jnp.array(betas)

    def get_initial_state(self):
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
    ) -> ArrayLike:
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
            New state at time (t + 1).
        """
        num_steps = int(1 / dt)
        for _ in range(num_steps):
            state += self.state_transition(state, t, beta) * dt
        return state

    def update_all_particles(self, t: int) -> None:
        """Propagate all particles forward one time step.

        Args:
            t: current time step
        """
        # Vectorizing the update function
        new_states = jax.vmap(self._update_single_particle, in_axes=(0, None, 0, None))(
            self.states, t, self.betas, self.settings.dt
        )
        self.states = new_states

    def _compute_single_weight(
        self, reported_data: int, particle_estimate: float | int
    ) -> float:
        """Computes the un-normalized weight of a single particle.

        Args:
            reported_data: Reported new hospitalization case counts at
                current time step.

        Returns:
            An un-normalized weight for a single particle.
        """
        weight = norm.logpdf(reported_data, particle_estimate, self.params.R)
        return float(weight)

    def compute_all_weights(self, reported_data: int):
        """Update the weights for every particle."""
        self.weights = jnp.zeros(self.settings.num_particles)
        for i in range(self.settings.num_particles):
            hosp_estimate = self.hosp_estimates[i]
            new_weight = self._compute_single_weight(reported_data, hosp_estimate)
            self.weights = self.weights.at[i].set(new_weight)

    def normalize_weights(self):
        """Normalize the weights using the Jacobian algorithm."""
        self.weights = log_norm(self.weights)

    def resample(self):
        resampling_indices = jnp.zeros(self.settings.num_particles, dtype=int)
        cdf_log = jacobian(self.weights)

        u = np.random.uniform(0, 1 / self.settings.num_particles)

        i = 0
        for j in range(self.settings.num_particles):
            r_log = np.log(u + (1 / self.settings.num_particles) * j)
            while r_log > cdf_log[i]:
                i += 1
            resampling_indices[j] = i

        # TODO: finish resampling logic
        # copy? convert particles to classes so copying is easier?

    def state_transition(self, state: ArrayLike, t: int, beta: float):
        """
        Integrator for the SIRH model from Alex's SDH project.

        Args:
            t: A float value representing the current time point.
            state: A NDArray holding the current state of the system to integrate.
            beta: A float value representing the beta parameter.

        Returns:
            A NDArray of numerical derivatives of the state.
        """
        S, I, R, H, new_H = state  # unpack the state variables
        N = S + I + R + H  # compute the total population

        new_H = (1 / self.params.D) * self.params.gamma * I

        """The state transitions of the ODE model are below"""
        dS = -beta * (S * I) / N + (1 / self.params.L) * R
        dI = beta * S * I / N - (1 / self.params.D) * I
        dR = (
            (1 / self.params.hosp) * H
            + ((1 / self.params.D) * (1 - self.params.gamma) * I)
            - (1 / self.params.L) * R
        )
        dH = (1 / self.params.D) * self.params.gamma * I - (1 / self.params.hosp * H)

        return jnp.array([dS, dI, dR, dH, new_H])


def run_pf(settings: InitSettings, observation_data: ArrayLike, runtime: int) -> None:
    particles = ParticleCloud(settings)
    obs_data = ObservationData(observation_data)

    logger = get_logger()

    for t in range(runtime):

        print(f"Iteration: {t + 1} \r")

        if t != 0:
            particles.update_all_particles(t)

        reported_data = obs_data.get_observation(t)

        particles.compute_all_weights(reported_data=reported_data)
        particles.normalize_weights()
        particles.resample()

    output_handler = OutputHandler(settings, runtime)
    output_handler.set_destination_directory('./output/')
    output_handler.output_average_betas(particles.betas)


def jacobian(little_delta: ArrayLike) -> ArrayLike:
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


def log_norm(log_weights: ArrayLike) -> ArrayLike:
    """
    Normalizes the probability space using the Jacobian algorithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
