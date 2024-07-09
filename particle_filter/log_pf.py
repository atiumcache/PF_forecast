from dataclasses import dataclass, field
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import jax


@dataclass
class GlobalParameters:
    """Defines the global parameters of the particle filter.

    Attributes:
        num_particles: Number of particles in the filter.
        population: Population of the location.
        beta_prior_range: Range of values for each particle's initial beta
            to be sampled from."""
    num_particles: int
    population: int
    time_steps: int
    beta_prior: Tuple[float, float] = (0.1, 0.15)
    seed_size: float = 0.005


@dataclass
class ModelParameters:
    beta: float = 0.1
    gamma: float = 0.06
    mu: float = 0.004
    q: float = 0.1
    eta: float = 0.1
    std: float = 10
    R: float = 50
    hosp: int = 10
    L: int = 90
    D: int = 10


@dataclass
class Particles:
    gp: GlobalParameters
    states: jnp.ndarray = field(init=False)
    weights: jnp.ndarray = field(init=False)
    betas: jnp.ndarray = field(init=False)

    def __post_init__(self):
        self.states = jnp.array([self.get_initial_state() for _ in
                                 range(
            self.gp.num_particles)])

        self.weights = jnp.zeros(self.gp.num_particles)

        betas = [np.random.uniform(self.gp.beta_prior[0],
                                   self.gp.beta_prior[1])
                 for _ in range(self.gp.num_particles)]

        self.betas = jnp.array(betas)

    def get_initial_state(self):
        population = self.gp.population

        # state = [S, I, R, H]
        state = [population, 0, 0, 0]
        infected_seed = np.random.uniform(0, self.gp.seed_size *
                                          population)
        # Move
        state[1] += infected_seed
        state[0] -= infected_seed

        return state


def run_pf(gp: GlobalParameters):
    particles = Particles(gp)


    for t in range(gp.time_steps):

        print(f"Iteration: {t + 1} \r")

        if t != 0:
            pass

def update_particles():
    pass


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
