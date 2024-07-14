from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import jax.scipy.stats.norm as norm
from jax.typing import ArrayLike

from filter_forecast.particle_filter.setup_pf import get_logger
from filter_forecast.particle_filter.init_settings import InitSettings


@dataclass
class ModelParameters:
    """
    SIRH model parameters, for the RHS function.
    """

    gamma: float = field(default_factory=lambda: 0.06)
    mu: float = field(default_factory=lambda: 0.004)
    q: float = field(default_factory=lambda: 0.1)
    eta: float = field(default_factory=lambda: 0.1)
    std: float = field(default_factory=lambda: 10.0)
    R: float = field(default_factory=lambda: 50.0)
    hosp: int = field(default_factory=lambda: 10)
    L: int = field(default_factory=lambda: 90)
    D: int = field(default_factory=lambda: 10)


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
        separated out to accommodate for gradient calculations in the future.

        Args:
             state: the current state of the particle at time t.
             t: the current time step
             beta: the current beta value for the particle
             dt: granularity for numerical integration

        Returns:
            New state at time (t + 1).
        """
        total_change = 0
        for _ in range(int(1 / dt)):
            total_change += self.state_transition(state, t, beta) * dt
        return total_change

    def update_all_particles(self, t: int) -> None:
        """Propagate all particles forward one time step.

        Args:
            t: current time step
        """
        for i in range(self.settings.num_particles):
            state = self.states[i]
            beta = self.betas[i]
            state[i] = self._update_single_particle(state, t, beta, self.settings.dt)

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


def compute_weights():
    pass


def RHS_H(
    state: jnp.ndarray, t: float, params: ModelParameters, beta: float
) -> np.ndarray:
    """Integrator for the SIRH model from Alex's SDH project.

    Args:

    t: A float value representing the current time point.
    state: A NDArray holding the current state of the system to integrate.
    param: A dictionary of named parameters.
    beta: A float value representing the beta parameter.

    Returns:
    A NDArray of numerical derivatives of the state.

    """
    S, I, R, H, new_H = state  # unpack the state variables
    N = S + I + R + H  # compute the total population

    new_H = (1 / params.D) * params.gamma * I

    """The state transitions of the ODE model is below"""
    dS = -beta * (S * I) / N + (1 / params.L) * R
    dI = beta * S * I / N - (1 / params.D) * I
    dR = (
        (1 / params.hosp) * H
        + ((1 / params.D) * (1 - params.gamma) * I)
        - (1 / params.L) * R
    )
    dH = (1 / params.D) * params.gamma * I - (1 / params.hosp) * H

    return jnp.array([dS, dI, dR, dH, new_H])


def jacobian(δ: jnp.ndarray):
    """
    The jacobian logarithm, used in log likelihood normalization and
    resampling processes.

    Args:
        δ: An array of values to sum

    Returns:
        The vector of partial sums of δ.
    """
    n = len(δ)
    Δ = jnp.zeros(n)
    Δ[0] = δ[0]
    for i in range(1, n):
        Δ[i] = max(δ[i], Δ[i - 1]) + jnp.log(1 + jnp.exp(-1 * jnp.abs(δ[i] - Δ[i - 1])))
    return Δ


def log_norm(log_weights):
    """
    Normalizes the probability space using the jacobian logarithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
