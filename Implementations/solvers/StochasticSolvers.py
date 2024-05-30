"""Stochastic analog to the euler solver for Alex and Kayodes SIRH model"""

from typing import List

import numpy as np

from Abstract.Integrator import Integrator
from utilities.Utils import Context, Particle


class PoissonSolver(Integrator):
    """This class uses the tau-leaping method to compute the next state of the particle and the observations,
    i.e. a poisson stochastic propagation model"""

    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        """Propagates the state forward one step and returns an array of states and observations across the the integration period

        Args:
            particleArray: A list of particles, this will be self.particles from Algorithm.
            ctx: The Algorithm's context object is passed as well, in case algorithm metadata is needed.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """

        tau = 1

        for j, particle in enumerate(particleArray):

            particleArray[j].observation = np.array(
                [0 for _ in range(len(particleArray[j].observation))]
            )

            for _ in range(int(1 / tau)):

                state, hospitalized, new_hospitalized = self.RHS(
                    particleArray[j], ctx, tau
                )
                particleArray[j].state = state

                """TODO Why is the observation like this? Must be broken."""
                particleArray[j].observation += [hospitalized, new_hospitalized]

        return particleArray

    def RHS(self, particle: Particle, ctx: Context, tau: float):
        """RHS function for the tau-leaping methodology.

        Args:
            particle: A single particle in the particle list.
            ctx: The algorithm's Context, used to obtain the rng.
            tau: A parameter governing the length of time on which to generate the poisson draws.

        """

        S, I, R, H = particle.state
        N = S + I + R + H
        new_susceptibles = ctx.rng.poisson(((1 / particle.param["L"]) * R) * tau)
        new_infected = ctx.rng.poisson((((particle.param["beta"] * S * I) / N)) * tau)
        new_recovered_from_H = ctx.rng.poisson(((1 / particle.param["hosp"]) * H) * tau)
        new_recovered_from_I = ctx.rng.poisson(
            (((1 / particle.param["D"]) * (1 - particle.param["gamma"])) * I) * tau
        )
        new_hospitalized = ctx.rng.poisson(
            ((1 / particle.param["D"]) * particle.param["gamma"] * I) * tau
        )

        state = np.zeros_like(particle.state)
        """Enforces the minimum will be 0"""
        state[0] = max(0.0, S - new_infected + new_susceptibles)
        state[1] = max(
            0.0, I + new_infected - (new_hospitalized + new_recovered_from_I)
        )
        state[2] = max(
            0.0, R + new_recovered_from_H + new_recovered_from_I - new_susceptibles
        )
        state[3] = max(0.0, H + new_hospitalized - new_recovered_from_H)

        return state, int(state[3]), new_hospitalized
