from abc import ABC, abstractmethod

from filter_forecast.particle_filter.parameters import ModelParameters
import jax.numpy as jnp


class Transition(ABC):
    def __init__(self, model_params: ModelParameters):
        self.params = model_params

    def det_component(self, state, beta):
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

    @abstractmethod
    def sto_component(self):
        """The stochastic component of the SDE model."""
        raise NotImplementedError


class OUModel(Transition):
    def __init__(self, model_params: ModelParameters):
        super().__init__(model_params)

    def sto_component(self):
        pass

