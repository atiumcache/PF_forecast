from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as random
from jax import Array, jit
from jax.typing import ArrayLike

from src.particle_filter.parameters import ModelParameters

# for typing hints
KeyArray = Array


class Transition(ABC):
    def __init__(self, config_file: str):
        """

        Args:
            config_file: Absolute path to config.toml file
        """
        self.params = ModelParameters(config_file)

    def det_component(self, state: ArrayLike, t: int) -> Array:
        """The deterministic component of the SDE model.

        Args:
            t: A float value representing the current time point.
            state: A NDArray holding the current state of the system to integrate.

        Returns:
            A NDArray of numerical derivatives of the state.
        """
        S, I, R, H, new_H, beta, ll_var = state  # unpack the state variables
        N = S + I + R + H  # compute the total population

        new_H = (1 / self.params.D) * self.params.gamma * I

        dS = -beta * (S * I) / N + (1 / self.params.L) * R
        dI = beta * S * I / N - (1 / self.params.D) * I
        dR = (
            (1 / self.params.hosp) * H
            + ((1 / self.params.D) * (1 - self.params.gamma) * I)
            - (1 / self.params.L) * R
        )
        dH = new_H - (1 / self.params.hosp * H)

        # OU process for beta
        d_beta = self.params.beta_theta * (self.params.beta_mu - beta)

        # OU process for likelihood variance
        d_var = self.params.ll_var_theta * (self.params.ll_var_mu - ll_var)

        return jnp.array([dS, dI, dR, dH, new_H, d_beta, d_var])

    @abstractmethod
    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """The stochastic component of the SDE model.

        Must be defined by each concrete implementation of the Transition class."""
        raise NotImplementedError


class GaussianNoiseModel(Transition):
    def __init__(self, model_params: ModelParameters, sigma: float = 0.001):
        super().__init__(model_params)
        self.sigma = sigma  # Volatility of the noise

    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """A simple gaussian noise is added to each element of the state vector.
        The magnitude of the noise is controlled by sigma.

        Args:
            state: A NDArray holding the current state of the system.
            dt: A float value representing the time step.
            key: A PRNGKey for random number generation.

        Return:
            A NDArray of stochastic increments.
        """
        noise = self.sigma * random.normal(key, shape=jnp.shape(state))
        return noise * jnp.sqrt(dt)


class OUModel(Transition):
    def __init__(self, model_params: ModelParameters, theta=0.005, mu=0.0, sigma=0.01):
        super().__init__(model_params)
        self.theta = theta  # Speed of reversion
        self.mu = mu  # Long-term mean
        self.sigma = sigma  # Volatility

    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """The stochastic component of the SDE model.
        Utilizes Weiner process for state variables.
        Utilizes OU processes for time-variant parameters.

        Args:
            state: A NDArray holding the current state of the system.
            dt: A float value representing the time step.
            key: A PRNGKey for random number generation.

        Returns:
            A NDArray of stochastic increments.
        """
        S, I, R, H, new_H, beta, ll_var = state  # unpack the state variables

        # Generate random noise
        noise = random.normal(key, shape=(4,))
        dW = jnp.sqrt(dt) * noise

        # OU process for each state variable
        dS = 0.1 * dW[0] * S
        dI = 0.1 * dW[1] * I
        dR = 0.1 * dW[2] * R
        dH = 0.1 * dW[3] * H

        # Stochastic component for beta
        dW_beta = random.normal(key, shape=())  # single Wiener process for beta
        d_beta = self.params.beta_sigma * jnp.sqrt(dt) * dW_beta

        # Stochastic component for sigma2
        dW_ll_var = random.normal(key, shape=())  # single Wiener process for sigma2
        d_ll_var = self.params.sigma2_eta * jnp.sqrt(dt) * dW_ll_var

        # Note that new_H is derived from I, so we don't need to
        # perturb new_H --- dI already accounts for that.
        return jnp.array([dS, dI, dR, dH, 0, d_beta, d_ll_var])
