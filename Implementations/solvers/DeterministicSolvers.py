from math import isnan
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint, solve_ivp

from Abstract.Integrator import Integrator
from utilities.Utils import Context, Particle

"""Integrators for various SIR style models and using various integrators, all deterministic."""


"""TODO The RHS functions floating out here and not inside a class are because scipy.integrate needs a function to integrate that lives outside the scope of the class
I forget why actually, this might be fixable. 

"""


def RHS_H(
    t: float, state: NDArray[np.float_], param: Dict[str, float]
) -> NDArray[np.float_]:
    """Integrator for the SIRH model from Alex's SDH project.

    Args:

    t: A float value representing the current time point.
    state: A NDArray holding the current state of the system to integrate.
    param: A dictionary of named parameters.

    Returns:
    A NDArray of numerical derivatives of the state.

    """
    S, I, R, H, new_H = state  # unpack the state variables
    N = S + I + R + H  # compute the total population

    new_H = (1 / param["D"]) * (param["gamma"]) * I

    """The state transitions of the ODE model is below"""
    dS = -param["beta"] * (S * I) / N + (1 / param["L"]) * R
    dI = param["beta"] * S * I / N - (1 / param["D"]) * I
    dR = (
        (1 / param["hosp"]) * H
        + ((1 / param["D"]) * (1 - (param["gamma"])) * I)
        - (1 / param["L"]) * R
    )
    dH = (1 / param["D"]) * (param["gamma"]) * I - (1 / param["hosp"]) * H

    return np.array([dS, dI, dR, dH, new_H])


def Jacobian(
    t: float, state: NDArray[np.float_], par: Dict[str, float]
) -> NDArray[np.float_]:
    """Jacobian for the SIRH model from Alex's SDH project.

    Args:

    t: A float value representing the current time point.
    state: A NDArray holding the current state of the system.
    param: A dictionary of named parameters.

    Returns:
    A 2-D NDArray of numerical partials of the state.

    """

    S, I, R, H, new_H = state
    N = S + I + R + H  # compute the total population
    return np.array(
        [
            [
                -I * par["beta"] / N,
                -S * par["beta"] / N,
                1 / par["L"],
                0,
                0,
            ],
            [
                I * par["beta"] / N,
                S * par["beta"] / N - 1 / par["D"],
                0,
                0,
                0,
            ],
            [
                0,
                (1 - par["gamma"]) / par["D"],
                -1 / par["L"],
                1 / par["hosp"],
                0,
            ],
            [
                0,
                par["gamma"] / par["D"],
                0,
                -1 / par["hosp"],
                0,
            ],
            [
                0,
                par["gamma"] / par["D"],
                0,
                0,
                0,
            ],
        ]
    )


def RHS_SEIARHD(
    t: float, state: NDArray[np.float_], param: Dict[str, float]
) -> NDArray[np.float_]:
    """Integator for the SIRH model from Alex's SDH project.

    Args:

    t: A float value representing the current time point.
    state: A NDArray holding the current state of the system to integrate.
    param: A dictionary of named parameters.

    Returns:
    A NDArray of numerical derivatives of the state.

    """
    S, E, A, I, H, R, D = state  # unpack the state variables

    N = S + E + A + I + R + H + D  # compute the total population

    kL = 0.25

    fA = 0.44

    fH = 0.054

    fR = 0.79

    cA = 0.26

    cI = 0.12

    cH = 0.17

    """The state transitions of the ODE model is below"""

    dS = -param["beta"] * (S * I) / N

    dE = param["beta"] * S * I / N - kL * E

    dA = kL * fA * E - cA * A

    dI = (
        kL * (1 - fA) * E - cI * I
    )  # compare the I compartment to the reported case number, this model works for case number comparison, may not work for hospitalization number

    dH = cI * fH * I - cH * fR * H

    dR = cA * A + cI * (1 - fH) * I + cH * fR * H

    dD = cH * (1 - fR) * H

    return np.array([dS, dE, dA, dI, dH, dR, dD])


def RHS_Calvetti(t: float, y: NDArray[np.float_], par: Dict[str, float]):
    """Integator for the SIRH model from Alex's SDH project.

    Args:

    t: A float value representing the current time point.
    y: A NDArray holding the current state of the system to integrate.
    par: A dictionary of named parameters.

    Returns:
    A NDArray of numerical derivatives of the state.

    """
    S, E, I, R, _ = y

    N = S + E + I + R

    dS = -par["beta"] * ((E + par["q"] * I) / N) * S
    dE = par["beta"] * ((E + par["q"] * I) / N) * S - par["eta"] * E - par["gamma"] * E
    dI = par["eta"] * E - par["gamma"] * I - par["mu"] * I
    dR = par["gamma"] * E + par["gamma"] * I

    d_newI = par["eta"] * E

    return np.array([dS, dE, dI, dR, d_newI])


class EulerSolver(Integrator):

    def __init__(self) -> None:
        """Integrator for one step of the SIRH model from Alex's SDH project."""
        super().__init__()

    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        """One step Euler integrator of the SIRH system, even though this is a very naive integration scheme, low accuracy integrators can work for PF due to the resampling dynamics.

        Args:
           particleArray: A list of particles, the Algorithm's self.particles list.
           ctx: The Algorithm's Context, in case metadata is necessary.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """

        dt = 1  # decrease for finer accuracy

        """This step is used for testing the forward estimation idea"""
        for particle in particleArray:
            particle.observation = np.array([0 for _ in range(ctx.forward_estimation)])

        for j, _ in enumerate(particleArray):

            # one step forward

            for _ in range(int(1 / dt)):

                """This loop runs over the particleArray, performing the integration in RHS for each one"""

                d_RHS, sim_obv = self.RHS_H(
                    particleArray[j].state, particleArray[j].param
                )

                particleArray[j].state += d_RHS * dt
                if np.any(np.isnan(particleArray[j].state)):
                    print(f"NaN state at particle: {j}")
                particleArray[j].observation[0] += d_RHS[3] * dt

            # additional loops

            # state = particleArray[j].state
            # for i in range(1,ctx.forward_estimation):
            #     for _ in range(int(1/dt)):

            #         d_RHS,sim_obv = self.RHS_H(state,particleArray[j].param)

            #         state += d_RHS*dt
            #         particleArray[j].observation[i] += sim_obv * dt

        return particleArray

    def RHS_H(
        self, state: NDArray[np.int_], param: Dict[str, int]
    ) -> NDArray[np.float_]:
        """RHS for the SIRH model from Alex's SDH project.

        Args:

        t: A float value representing the current time point.
        state: A NDArray holding the current state of the system to integrate.
        param: A dictionary of named parameters.

        Returns:
        A NDArray of numerical derivatives of the state.

        """
        # params has all the parameters – beta, gamma
        # state is a numpy array

        S, I, R, H = state  # unpack the state variables
        N = S + I + R + H  # compute the total population

        new_H = (
            (1 / param["D"]) * param["gamma"]
        ) * I  # our observation value for the particle

        """The state transitions of the ODE model is below"""
        dS = -param["beta"] * (S * I) / N + (1 / param["L"]) * R
        dI = param["beta"] * S * I / N - (1 / param["D"]) * I
        dR = (
            (1 / param["hosp"]) * H
            + ((1 / param["D"]) * (1 - (param["gamma"])) * I)
            - (1 / param["L"]) * R
        )
        dH = (1 / param["D"]) * (param["gamma"]) * I - (1 / param["hosp"]) * H

        return np.array([dS, dI, dR, dH]), new_H


class EulerSolver_SEAIRH(Integrator):

    def __init__(self) -> None:
        """Euler integrator for the SEIARHD model from Ye's paper."""
        super().__init__()

    """Propagates the state forward one step and returns an array of states and observations across the the integration period"""

    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        """One step Euler integrator of the SEIARHD system, even though this is a very naive integration scheme, low accuracy integrators can work for PF due to the resampling dynamics.

        Args:
           particleArray: A list of particles, the Algorithm's self.particles list.
           ctx: The Algorithm's Context, in case metadata is necessary.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """

        dt = 1

        # zero out the particleArray
        for particle in particleArray:
            particle.observation = np.array([0 for _ in range(ctx.forward_estimation)])

        for j, _ in enumerate(particleArray):

            # one step forward

            for _ in range(int(1 / dt)):

                """This loop runs over the particleArray, performing the integration in RHS for each one"""

                d_RHS, sim_obv = self.RHS(
                    particleArray[j].state, particleArray[j].param
                )

                particleArray[j].state += d_RHS * dt
                if np.any(np.isnan(particleArray[j].state)):
                    print(f"NaN state at particle: {j}")
                particleArray[j].observation[0] += sim_obv * dt

            # additional loops

            state = particleArray[j].state
            for i in range(1, ctx.forward_estimation):
                for _ in range(int(1 / dt)):

                    d_RHS, sim_obv = self.RHS(state, particleArray[j].param)

                    state += d_RHS * dt
                    particleArray[j].observation[i] += sim_obv

        return particleArray

    def RHS(t: float, state: NDArray, param: Dict[str, float]) -> NDArray[np.float_]:
        """RHS for the SIRH model from Alex's SDH project.

        Args:

        t: A float value representing the current time point.
        state: A NDArray holding the current state of the system to integrate.
        param: A dictionary of named parameters.

        Returns:
        A NDArray of numerical derivatives of the state.

        """

        # params has all the parameters – beta, gamma

        # state is a numpy array

        S, E, A, I, H, R, D = state  # unpack the state variables

        N = S + E + A + I + R + H + D  # compute the total population

        kL = 0.25

        fA = 0.44

        fH = 0.054

        fR = 0.79

        cA = 0.26

        cI = 0.12

        cH = 0.17

        """The state transitions of the ODE model is below"""

        dS = -param["beta"] * (S * I) / N

        dE = param["beta"] * S * I / N - kL * E

        dA = kL * fA * E - cA * A

        dI = (
            kL * (1 - fA) * E - cI * I
        )  # compare the I compartment to the reported case number, this model works for case number comparison, may not work for hospitalization number

        dH = cI * fH * I - cH * fR * H

        dR = cA * A + cI * (1 - fH) * I + cH * fR * H

        dD = cH * (1 - fR) * H

        new_I = kL * (1 - fA) * E
        return np.array([dS, dE, dA, dI, dH, dR, dD]), new_I


class EulerSolver_SIR(Integrator):

    def __init__(self) -> None:
        """One step integrator for the simple SIR model, used for testing purposes."""
        super().__init__()

    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        """One step Euler integrator of the SIR system, even though this is a very naive integration scheme, low accuracy integrators can work for PF due to the resampling dynamics.

        Args:
           particleArray: A list of particles, the Algorithm's self.particles list.
           ctx: The Algorithm's Context, in case metadata is necessary.

        Returns:
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the
            self.particles list in the Algorithm is updated via assignment.

        """

        dt = 1
        for particle in particleArray:
            particle.observation = np.array([0 for _ in range(ctx.forward_estimation)])

        for j, _ in enumerate(particleArray):
            # one step forward
            for _ in range(int(1 / dt)):

                """This loop runs over the particleArray, performing the integration in RHS for each one"""

                d_RHS, sim_obv = self.RHS(
                    particleArray[j].state, particleArray[j].param
                )

                particleArray[j].state += d_RHS * dt
                if np.any(np.isnan(particleArray[j].observation)):
                    print(f"NaN observation at particle: {j}")

                if isnan(sim_obv):
                    sim_obv = 0
                particleArray[j].observation[0] += sim_obv

            # additional loops
            state = particleArray[j].state
            for i in range(1, ctx.forward_estimation):
                for _ in range(int(1 / dt)):

                    d_RHS, sim_obv = self.RHS(state, particleArray[j].param)

                    state += d_RHS * dt
                    particleArray[j].observation[i] += sim_obv

        return particleArray

    def RHS(
        self, state: NDArray[np.float_], param: Dict[str, float]
    ) -> NDArray[np.float_]:
        """RHS for the SIR model.

        Args:

        t: A float value representing the current time point.
        state: A NDArray holding the current state of the system to integrate.
        param: A dictionary of named parameters.

        Returns:
        A NDArray of numerical derivatives of the state.

        """

        S, I, R = state
        N = S + I + R

        new_I = param["beta"] * S * I / N - param["gamma"] * I

        dS = -param["beta"] * S * I / N + param["eta"] * R
        dI = param["beta"] * S * I / N - param["gamma"] * I
        dR = param["gamma"] * I - param["eta"] * R

        return np.array([dS, dI, dR]), new_I


class LSODASolver(Integrator):

    def __init__(self) -> None:
        """A one step integrator of Alex's SIRH model using LSODA or RK45 if the jacobian is not available."""
        super().__init__()

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

        for i, particle in enumerate(particleArray):

            y0 = np.concatenate(
                (particle.state, particle.observation)
            )  # Initial state of the system

            t_span = [0.0, 1.0]
            sol = solve_ivp(
                fun=lambda t, y: RHS_H(t, y, particle.param),
                t_span=(0.0, 1.0),
                y0=y0,
                t_eval=t_span,
                method="RK45",
                rtol=1e-3,
                atol=1e-3,
            )

            particleArray[i].state = sol.y[: ctx.state_size, 1]
            particleArray[i].observation = np.array([sol.y[-1, 1] - sol.y[-1, 0]])
            # particleArray[i].observation = sol.y[3,1]

        return particleArray


class LSODASolverSEIARHD:

    def __init__(self) -> None:
        """A one step integrator of Ye's SEIARHD model using LSODA or RK45 if the jacobian is not available."""
        super().__init__()

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

        for i, particle in enumerate(particleArray):

            y0 = particle.state  # Initial state of the system

            t_span = [0.0, float(ctx.forward_estimation)]
            par = particle.param
            sol = solve_ivp(
                fun=lambda t, z: RHS_SEIARHD(t, z, par),
                t_span=t_span,
                y0=y0,
                t_eval=np.linspace(t_span[0], t_span[1], ctx.forward_estimation + 1),
                method="LSODA",
                rtol=1e-3,
                atol=1e-3,
            )

            particleArray[i].state = sol.y[: ctx.state_size, 1]

            # for j in range(ctx.forward_estimation):
            particleArray[i].observation = np.array(
                sol.y[3, 1 : ctx.forward_estimation + 1]
            )
            if np.any(np.isnan(particleArray[i].state)):
                print(f"NaN state at particle: {i}")

        return particleArray


class LSODACalvettiSolver(Integrator):

    def __init__(self) -> None:
        """A one step integrator for the SEIR model of Calvetti et. al. using LSODA."""
        super().__init__()

    """Elements of particleArray are of Particle class in utilities/Utils.py"""

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
        for i, particle in enumerate(particleArray):

            y0 = np.concatenate(
                (particle.state, particle.observation)
            )  # Initial state of the system

            t_span = [0.0, 1.0]
            par = particle.param
            sol = solve_ivp(
                fun=lambda t, z: RHS_Calvetti(t, z, par),
                t_span=(0.0, 1.0),
                y0=y0,
                t_eval=t_span,
                method="LSODA",
                rtol=1e-3,
                atol=1e-3,
            )

            particleArray[i].state = sol.y[: ctx.state_size, 1]
            # particleArray[i].observation = np.array([sol.y[3,1]])
            particleArray[i].observation = np.array([sol.y[-1, 1] - sol.y[-1, 0]])
            # particleArray[i].observation = np.array([sol.y[2,1]])

            if np.any(np.isnan(particleArray[i].state)):
                print(f"NaN state at particle: {i}")

        return particleArray
