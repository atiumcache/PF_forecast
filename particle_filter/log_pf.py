from dataclasses import dataclass, field
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import jax.scipy.stats.norm as norm
import jax
from jax.experimental.ode import odeint
from jax.typing import ArrayLike


@dataclass
class InitSettings:
    """Defines the initial settings of the particle filter.

    Attributes:
        num_particles: Number of particles in the filter.
        population: Population of the location.
        time_steps: How many days to run the particle filter.
        dt: Granularity of numerical integration.
        beta_prior: Range of values for each particle's initial beta
            to be sampled from.
        seed_size: Determines the ratio of population initially infected. See get_initial_state.
    """
    num_particles: int
    population: int
    time_steps: int
    dt: float = field(default_factory=lambda: 1.0)
    beta_prior: Tuple[float, float] = field(default_factory=lambda: (0.1,
                                                                     0.15))
    seed_size: float = field(default_factory=lambda: 0.005)


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
class ParticleCloud:
    settings: InitSettings
    params: ModelParameters = field(init=False)
    states: ArrayLike = field(init=False)
    weights: ArrayLike = field(init=False)
    betas: ArrayLike = field(init=False)
    observations: ArrayLike = field(init=False)

    def __post_init__(self):
        self.params = ModelParameters()
        self.states = jnp.array([self.get_initial_state() for _ in
                                 range(self.settings.num_particles)])

        self.weights = jnp.zeros(self.settings.num_particles)

        betas = [np.random.uniform(self.settings.beta_prior[0],
                                   self.settings.beta_prior[1])
                 for _ in range(self.settings.num_particles)]

        self.betas = jnp.array(betas)

    def get_initial_state(self):
        population = self.settings.population

        # state = [S, I, R, H, new_H]
        state = [population, 0, 0, 0, 0]
        infected_seed = np.random.uniform(0, self.settings.seed_size *
                                          population)
        # Move
        state[1] += infected_seed
        state[0] -= infected_seed

        return state

    def update_all_particles(self, t: int):
        """Propagate all particles forward one time step."""
        for state in self.states:
            state = self.update_single_particle(state, t, self.settings.dt)

    def update_single_particle(self, i: int, t: int, dt: float) -> ArrayLike:
        """For a single particle, step the state forward 1 time step.

        Helper function for update_all_particles."""
        total_change = 0
        for _ in range(int(1 / dt)):
            total_change += self.state_transition(self.states[i], t,
                                               self.betas[i]) * dt
        return total_change

    def compute_single_weight(self):
        pass

    def compute_all_weights(self):
        self.weights = jnp.zeros(self.settings.num_particles)


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


def run_pf(settings: InitSettings):
    particles = ParticleCloud(settings)
    model_params = ModelParameters()

    for t in range(settings.time_steps):

        print(f"Iteration: {t + 1} \r")

        if t != 0:
            for i in range(settings.num_particles):
                particles.update_all_particles(t)


def compute_weights():
    pass


def RHS_H(state: jnp.ndarray, t: float, params: ModelParameters, beta: float) -> np.ndarray:
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
        Δ[i] = max(δ[i], Δ[i - 1]) + jnp.log(1 + jnp.exp(-1 * jnp.abs(δ[i] -
                                                                      Δ[i - 1])))
    return Δ


def log_norm(log_weights):
    """
    Normalizes the probability space using the jacobian logarithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
