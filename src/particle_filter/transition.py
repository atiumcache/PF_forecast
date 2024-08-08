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
        self.params.update_all(key=3)
        S, I, R, H, new_H, beta = state  # unpack the state variables
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

        return jnp.array([dS, dI, dR, dH, new_H, 0])

    @abstractmethod
    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """The stochastic component of the SDE model.

        Must be defined by each concrete implementation of the Transition class."""
        raise NotImplementedError

    @abstractmethod
    def update_beta(self):
        raise NotImplementedError("Subclasses must implement.")


class OUModel(Transition):
    def __init__(self, config_file: str):
        super().__init__(config_file)

    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """The stochastic component of the SDE model.
        Utilizes Wiener process for state variables (S, I, R, H).

        Args:
            state: A NDArray holding the current state of the system.
            dt: A float value representing the time step.
            key: A PRNGKey for random number generation.

        Returns:
            A NDArray of stochastic increments.
        """
        S, I, R, H, new_H, beta = state  # unpack the state variables

        # Generate random noise
        noise = random.normal(key, shape=(5,))
        dW = jnp.sqrt(dt) * noise

        # Wiener process for each state variable
        dS = self.params.dW_volatility * dW[0] * S
        dI = self.params.dW_volatility * dW[1] * I
        dR = self.params.dW_volatility * dW[2] * R
        dH = self.params.dW_volatility * dW[3] * H
        new_H = self.params.dW_volatility * dW[4] * (1 / self.params.D) * self.params.gamma * I

        # Note that new_H is derived from I, so we do not
        # perturb new_H --- perturbations to I already account for that.
        return jnp.array([dS, dI, dR, dH, new_H, 0])

    def update_beta(self, beta: float, dt: float, t: int, key: KeyArray) -> Array:
        """OU Process update for beta.
        
        Args:
            state: A NDArray holding the current state of the system.
            dt: A float value representing the time step.
            key: A PRNGKey for random number generation.

        Returns:
            A NDArray of stochastic increments.
        """
        dW = random.normal(key, shape=())  # single Wiener process for beta
        d_beta = self.params.beta_sigma * jnp.sqrt(dt) * dW
        d_beta += self.params.beta_theta * (self.params.beta_mu - beta) * dt
        
        return d_beta
