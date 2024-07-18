from abc import ABC, abstractmethod

from filter_forecast.particle_filter.parameters import ModelParameters
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike
from jax import Array


# for typing hints
KeyArray = Array


class Transition(ABC):
    def __init__(self, model_params: ModelParameters):
        self.params = model_params

    def det_component(self, state: ArrayLike, t: int, beta: float) -> Array:
        """The deterministic component of the SDE model.
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

        dS = -beta * (S * I) / N + (1 / self.params.L) * R
        dI = beta * S * I / N - (1 / self.params.D) * I
        dR = (
            (1 / self.params.hosp) * H
            + ((1 / self.params.D) * (1 - self.params.gamma) * I)
            - (1 / self.params.L) * R
        )
        dH = new_H - (1 / self.params.hosp * H)

        return jnp.array([dS, dI, dR, dH, new_H])

    @abstractmethod
    def sto_component(self, state: ArrayLike, dt: float, key: KeyArray) -> Array:
        """The stochastic component of the SDE model.

        Must be defined by each concrete implementation of the Transition class."""
        raise NotImplementedError


class GaussianNoiseModel(Transition):
    def __init__(self, model_params: ModelParameters, sigma=0.001):
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
        """The stochastic component of the SDE model using an OU process.

        Args:
            state: A NDArray holding the current state of the system.
            dt: A float value representing the time step.
            key: A PRNGKey for random number generation.

        Returns:
            A NDArray of stochastic increments.
        """
        S, I, R, H, new_H = state  # unpack the state variables

        # Generate random noise
        noise = random.normal(key, shape=state.shape)

        # OU process for each state variable
        dS = self.theta * (self.mu - S) * dt + self.sigma * jnp.sqrt(dt) * noise[0]
        dI = self.theta * (self.mu - I) * dt + self.sigma * jnp.sqrt(dt) * noise[1]
        dR = self.theta * (self.mu - R) * dt + self.sigma * jnp.sqrt(dt) * noise[2]
        dH = self.theta * (self.mu - H) * dt + self.sigma * jnp.sqrt(dt) * noise[3]
        dNew_H = (
            self.theta * (self.mu - new_H) * dt + self.sigma * jnp.sqrt(dt) * noise[4]
        )

        return jnp.array([dS, dI, dR, dH, dNew_H])
